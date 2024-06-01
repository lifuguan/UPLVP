'''
Author: lifuguan lifugan_10027@outlook.com
Date: 2022-06-01 12:00:58
LastEditors: lifuguan lifugan_10027@outlook.com
LastEditTime: 2022-06-01 17:28:44
FilePath: /research_workspace/custommd/models/detectors/upqueryinst.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import transforms
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import QueryInst
from mmdet.core.mask.structures import BitmapMasks

from ..oursnet.utils import mask2bbox, sem2ins_masks, matrix_nms, center_of_mass, GaussianBlur

@DETECTORS.register_module()
class UpQueryInstSup(QueryInst):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 num_proposals = 100,
                 query_shuffle=False,
                 is_kernel2query = True,
                 mask_ratio=0.05,
                 mask_assign_stride=4,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(UpQueryInstSup, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.mask_assign_stride = mask_assign_stride
        self.get_query_tarnsforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToPILImage(mode='RGB'),
            transforms.RandomApply([
                GaussianBlur([.1, 2.])
                ], p=0.5),
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(),  HorizontalFlip may cause the pretext too difficult, so we remove it
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        """build up-knet approach"""
        self.num_proposals = num_proposals
        self.is_kernel2query = is_kernel2query
        if self.is_kernel2query is True:
            self.query_shuffle = query_shuffle

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # pooling used for the query patch feature

            self.patch2query = nn.Linear(2048, 256)
            
            self.mask_ratio = mask_ratio

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """Forward function of SparseR-CNN and QueryInst in train stage.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (List[Tensor], optional) : Segmentation masks for
                each box. This is required to train QueryInst.
            proposals (List[Tensor], optional): override rpn proposals with
                custom proposals. Use when `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        assert proposals is None, 'Sparse R-CNN and QueryInst ' \
            'do not support external proposals'
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

        pad_H, pad_W = img_metas[0]['batch_input_shape']
        bs = len(img_metas)

        assign_H = pad_H // self.mask_assign_stride
        assign_W = pad_W // self.mask_assign_stride

        x, gt_masks, gt_labels, gt_bboxes = self.extract_feat_train(img, img_metas)
        gt_masks_tensor = []
        for i, gt_mask in enumerate(gt_masks):
            mask_tensor = gt_mask
            if mask_tensor.shape[0] == 0:
                gt_masks_tensor.append(
                    mask_tensor.new_zeros(
                        (mask_tensor.size(0), assign_H, assign_W)))
            else:
                mask_numpy = F.interpolate(
                        mask_tensor[None], (assign_H, assign_W),
                        mode='bilinear',
                        align_corners=False)[0].to('cpu').detach().numpy()
                gt_masks_tensor.append(BitmapMasks(mask_numpy, mask_numpy.shape[-2], mask_numpy.shape[-1]))
        gt_masks = gt_masks_tensor
        
        if self.is_kernel2query is True:
            patch_feats = self.extract_patches(img, gt_masks, img_metas)

        proposal_boxes, proposal_feats, imgs_whwh = \
            self.rpn_head.forward_train(x, img_metas)

        if self.is_kernel2query is True:
            patch_feature_gt = self.avgpool(patch_feats).flatten(1)
            patch_feats = self.patch2query(patch_feature_gt) \
                .view(bs, 20, -1) \
                .repeat_interleave(self.num_proposals // 20, dim=1) \
                .permute(1, 0, 2) \
                .contiguous()


            # if object query shuffle, we shuffle the index of object query embedding,
            # which simulate to adding patch feature to object query randomly.
            idx = torch.randperm(self.num_proposals) if self.query_shuffle else torch.arange(self.num_proposals)

            # NOTE query 
            # for training, it uses fixed number of query patches.
            mask_query_patch = (torch.rand(self.num_proposals, bs, 1, device=patch_feats.device) > self.mask_ratio).float()
            # NOTE mask some query patch and add query embedding
            refine_patch_feats = patch_feats * mask_query_patch
            proposal_feats = proposal_feats + refine_patch_feats[idx, ...] \
                .permute(1,0,2).contiguous()

        roi_losses = self.roi_head.forward_train(
            x,
            proposal_boxes,
            proposal_feats,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_masks=gt_masks,
            imgs_whwh=imgs_whwh)
        return roi_losses

    def extract_feat_train(self, imgs, img_metas):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(imgs)
        key_feats = x[-1]
        if self.with_neck:
            x = self.neck(x)

        B, C, H, W = key_feats.shape

        ins_per_img, pusedo_masks, pusedo_kernels, pusedo_labels, pusedo_boxes = [],[],[],[],[]
        for keys, img in zip(key_feats, imgs):
            scale_factors = [1, 0.5, 0.25]
            queries_list = []
            for scale_factor in scale_factors:
                cur_queries = F.interpolate(keys[None, ...], scale_factor=scale_factor, mode='bilinear')[0].reshape(keys.shape[0], -1).permute(1, 0)
                num_q = len(cur_queries)
                queries_list.append(cur_queries)
            queries = torch.cat(queries_list)
            _, H, W = keys.shape
            keys = keys / keys.norm(dim=0, keepdim=True)
            queries = queries / queries.norm(dim=1, keepdim=True)
            attn = queries @ keys.reshape(keys.shape[0], -1)
            # min max normalize
            attn -= attn.min(-1, keepdim=True)[0]
            attn /= attn.max(-1, keepdim=True)[0]

            attn = attn.reshape(attn.shape[0], H, W)


            soft_masks = attn
            masks = soft_masks >= 0.5

            sum_masks = masks.sum((1,2))
            keep = sum_masks > 1
            if keep.sum() == 0:
                continue
            masks = masks[keep]
            soft_masks = soft_masks[keep]
            sum_masks = sum_masks[keep]
            queries = queries[keep]

            # Matrix NMS
            maskness = (soft_masks * masks.float()).sum((1, 2)) / sum_masks
            sort_inds = torch.argsort(maskness, descending=True)
            maskness = maskness[sort_inds]
            masks = masks[sort_inds]
            sum_masks = sum_masks[sort_inds]
            soft_masks = soft_masks[sort_inds]
            queries = queries[sort_inds]
            
            pusedo_label = maskness * 0 # 赋值假标签
            # scores, label, _, keep_inds = mask_matrix_nms(
            #     masks,
            #     pusedo_label,
            #     maskness,
            #     mask_area=sum_masks,
            #     max_num=20,
            #     sigma=4.0,
            #     filter_thr=0.6
            #     )

            maskness = matrix_nms(masks, maskness*0, maskness, sigma=2, kernel='gaussian', sum_masks=sum_masks)

            keep_inds = torch.argsort(maskness, descending=True)
            if len(keep_inds) > 20:
                keep_inds = keep_inds[:20]

            pusedo_label = pusedo_label[keep_inds]
            masks = masks[keep_inds]
            maskness = maskness[keep_inds]
            soft_masks = soft_masks[keep_inds]
            queries = queries[keep_inds]

            soft_masks = F.interpolate(soft_masks[None, ...], size=(H, W), mode='bilinear')[0]
            masks = (soft_masks >= 0.5).float()
            # masks = median_blur(masks.unsqueeze(0), [3,3]).squeeze(0)
            # masks = (masks >= 0.5).float()

            # mask to box
            width_proj = masks.max(1)[0]
            height_proj = masks.max(2)[0]
            center_ws, _ = center_of_mass(width_proj[:, None, :])
            _, center_hs = center_of_mass(height_proj[:, :, None])
            boxes = mask2bbox(masks.bool())
            #boxes = []
            #for mask in masks.cpu().numpy():
            #    ys, xs = np.where(mask)
            #    box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            #    boxes.append(box)
            #boxes = torch.tensor(boxes, device = maskness.device)

            # filter masks on the top border or with large width
            keep = center_hs > 0.1 * H
            keep_1 = (boxes[:, 3] - boxes[:, 1]) > 0.8 * H
            keep_2 = (boxes[:, 2] - boxes[:, 0]) > 0.8 * W
            keep_3 = maskness >= 0.7
            keep = keep & ~(keep_1 & keep_2) & keep_3
            #
            if keep.sum() == 0:
                # continue
                keep[0] = True

            masks = masks[keep]
            boxes = boxes[keep]
            boxes = boxes.float() * 32
            pusedo_label = pusedo_label[keep].long()

            pusedo_labels.append(pusedo_label.detach())  
            pusedo_masks.append(masks)
            ins_per_img.append(masks.shape[0])
            pusedo_boxes.append(boxes)


        return x, pusedo_masks, pusedo_labels, pusedo_boxes

    def extract_patches(self, imgs, gt_masks, img_metas):
        """Directly extract features from the backbone+neck."""

        ins_per_img, pusedo_kernels = [], []

        for per_img_mask, img in zip(gt_masks, imgs):
            mask_tensor = per_img_mask.to_tensor(torch.float, img.device)
            boxes = mask2bbox(mask_tensor.bool()).long()
            for box in boxes:
                patch = img[:, box[1]:box[3], box[0]:box[2]]

                patch = self.get_query_tarnsforms(patch)[None,...].to(device=mask_tensor.device)
                patch_feat = self.backbone(patch)[-1]
                pusedo_kernels.append(patch_feat)
            ins_per_img.append(len(mask_tensor))

        pusedo_kernels = torch.cat(pusedo_kernels, dim=0)


        begin_i, expend_kernels = 0, []
        for end_i in ins_per_img:
            single_patch_feats = pusedo_kernels[begin_i:end_i] \
                .repeat_interleave(20 // end_i, dim=0)
            
            if 20 % end_i != 0:
                single_patch_feats = torch.cat((single_patch_feats, \
                    pusedo_kernels[begin_i:20 % end_i]), dim=0)
            # 将kernels扩充到20个
            expend_kernels.append(single_patch_feats)

        expend_kernels = torch.cat(expend_kernels, dim=0)


        return expend_kernels.detach()