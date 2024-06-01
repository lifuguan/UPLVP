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

from ..oursnet.utils import mask2bbox, sem2ins_masks, matrix_nms, center_of_mass, GaussianBlur

@DETECTORS.register_module()
class UpQueryInstPrompt(QueryInst):
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
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(UpQueryInstPrompt, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

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
            
            self.mask_ratio = mask_ratio
            print("is_kernel2query is True.")
        else:
            print("is_kernel2query is False.")
            
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

        bs = len(img_metas)

        x = self.extract_feat(img)
        patch_feats = self.extract_patches(img, gt_masks, img_metas, x[-1])

        proposal_boxes, proposal_feats, imgs_whwh = \
            self.rpn_head.forward_train(x, img_metas)


        if self.is_kernel2query is True:
            patch_feats = patch_feats.view(bs, 20, -1) \
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
    
    def extract_patches(self, imgs, gt_masks, img_metas, feats):
        """Directly extract features from the backbone+neck."""

        ins_per_img, pusedo_kernels = [], []

        for i, (per_img_mask, img) in enumerate(zip(gt_masks, imgs)):
            if len(per_img_mask) == 0:
                print(img_metas[i])
                patch_feat_refine = torch.zeros((1,256), device=img.device)
                pusedo_kernels.append(patch_feat_refine)
                ins_per_img.append(1)
            else:
                mask_tensor = per_img_mask.to_tensor(torch.float, img.device)
                boxes = (mask2bbox(mask_tensor.bool()) / 32).long()
                for box in boxes:
                    if box[1] == box[3]:
                        box[3] += 2
                    if box[0] == box[2]:
                        box[2] += 2
                    patch_feat = feats[i, :, box[1]:box[3], box[0]:box[2]]
                    patch_feat_refine = self.avgpool(patch_feat).flatten(1).permute(1, 0)
                    if torch.isnan(patch_feat_refine).any():
                        print("danger")
                    pusedo_kernels.append(patch_feat_refine)
                ins_per_img.append(len(mask_tensor))
        # if torch.isnan(pusedo_kernels).any():
        #     print("danger")
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