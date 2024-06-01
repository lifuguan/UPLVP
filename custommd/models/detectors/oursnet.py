import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import transforms
from mmdet.models.builder import DETECTORS

from mmdet.models.detectors import TwoStageDetector
from mmdet.utils import get_root_logger

from ..oursnet.utils import mask2bbox, sem2ins_masks, matrix_nms, center_of_mass, GaussianBlur
from ..detectors.knet import KNet
@DETECTORS.register_module()
class OursNet(KNet):

    def __init__(self,
                 *args,
                 num_proposals = 100,
                 query_shuffle=False,
                 is_kernel2query = True,
                 is_loss_emb = True,
                 mask_ratio=0.1, 
                 **kwargs):
        super(OursNet, self).__init__(*args, **kwargs)
        """build free mask approach"""
        self.free_scale_factors = 4
        self.maskness_thr = 0.5
        self.out_mask_thr = 0.7
        self.minial_area = 20

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
        """use semantic embedding loss"""
        self.is_loss_emb = is_loss_emb

        """build up-knet approach"""
        self.num_proposals = num_proposals
        self.is_kernel2query = is_kernel2query
        if self.is_kernel2query is True:
            self.query_shuffle = query_shuffle

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # pooling used for the query patch feature

            self.patch2query = nn.Linear(2048, 256)
            self.gen_gt_kernel = nn.Linear(2048, 256)
            
            self.mask_ratio = mask_ratio


    def extract_feat_train(self, imgs, img_metas):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(imgs)
        key_feats = x[-1]
        if self.with_neck:
            x = self.neck(x)

        B, C, H, W = key_feats.shape

        ins_per_img, pusedo_masks, pusedo_kernels, pusedo_labels, pseudo_queries = [],[],[],[],[]

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

            pusedo_label = pusedo_label[keep_inds].long()
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
            keep_1 = maskness >= 0.7
            keep = keep & keep_1
            #
            if keep.sum() == 0:
                # continue
                keep[0] = True

            masks = masks[keep]
            boxes = boxes[keep]
            queries = queries[keep]
            boxes = boxes.long() * 32
            for box in boxes:
                patch = img[:, box[1]:box[3], box[0]:box[2]]

                patch = self.get_query_tarnsforms(patch)[None,...].to(device=masks.device)
                patch_feat = self.backbone(patch)[-1]
                pusedo_kernels.append(patch_feat)

            pusedo_label = pusedo_label[keep]
            pusedo_labels.append(pusedo_label.detach())  
            pusedo_masks.append(masks)
            pseudo_queries.append(queries)
            ins_per_img.append(masks.shape[0])

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


        return x, expend_kernels.detach(), pusedo_masks, pusedo_labels, pseudo_queries

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      gt_semantic_seg=None,
                      **kwargs):
        super(TwoStageDetector, self).forward_train(img, img_metas)
        x, patch_feats, gt_masks, gt_labels, target_kernels = self.extract_feat_train(img, img_metas)
        assert proposals is None, 'OursNet does not support' \
                                  ' external proposals'
        assert gt_masks is not None

        # gt_masks and gt_semantic_seg are not padded when forming batch
        gt_masks_tensor = []
        gt_sem_seg = []
        gt_sem_cls = []
        # batch_input_shape shoud be the same across images
        pad_H, pad_W = img_metas[0]['batch_input_shape']
        bs = len(img_metas)
        assign_H = pad_H // self.mask_assign_stride
        assign_W = pad_W // self.mask_assign_stride

        for i, gt_mask in enumerate(gt_masks):
            mask_tensor = gt_mask

            if gt_semantic_seg is not None:
                # gt_semantic seg is padded by 255 and
                # zero indicating the first class
                sem_labels, sem_seg = sem2ins_masks(
                    gt_semantic_seg[i],
                    num_thing_classes=self.num_thing_classes)
                if sem_seg.shape[0] == 0:
                    gt_sem_seg.append(
                        mask_tensor.new_zeros(
                            (mask_tensor.size(0), assign_H, assign_W)))
                else:
                    gt_sem_seg.append(
                        F.interpolate(
                            sem_seg[None], (assign_H, assign_W),
                            mode='bilinear',
                            align_corners=False)[0])
                gt_sem_cls.append(sem_labels)

            else:
                gt_sem_seg = None
                gt_sem_cls = None

            if mask_tensor.shape[0] == 0:
                gt_masks_tensor.append(
                    mask_tensor.new_zeros(
                        (mask_tensor.size(0), assign_H, assign_W)))
            else:
                gt_masks_tensor.append(
                    F.interpolate(
                        mask_tensor[None], (assign_H, assign_W),
                        mode='bilinear',
                        align_corners=False)[0])

        gt_masks = gt_masks_tensor
        rpn_results = self.rpn_head.forward_train(x, img_metas, gt_masks,
                                                  gt_labels, gt_sem_seg,
                                                  gt_sem_cls)
        (rpn_losses, proposal_feats, x_feats, mask_preds,
         cls_scores) = rpn_results

        if self.is_kernel2query is True:
            patch_feature_gt = self.avgpool(patch_feats).flatten(1)
            patch_feats = self.patch2query(patch_feature_gt) \
                .view(bs, 20, -1) \
                .repeat_interleave(self.num_proposals // 20, dim=1) \
                .permute(1, 0, 2) \
                .contiguous()


            # NOTE query 
            # for training, it uses fixed number of query patches.
            mask_query_patch = (torch.rand(self.num_proposals, bs, 1, device=patch_feats.device) > self.mask_ratio).float()
            # NOTE mask some query patch and add query embedding
            refine_patch_feats = patch_feats * mask_query_patch
            proposal_feats = proposal_feats + refine_patch_feats \
                .permute(1,0,2).contiguous().unsqueeze(-1).unsqueeze(-1)

        if self.is_loss_emb == True: 
            gt_kernels = [self.gen_gt_kernel(gt_kernel) for gt_kernel in target_kernels] # 压缩后所有query都长的差不多了
        else:
            gt_kernels =  None
        # kernel_iter_update.py 
        losses = self.roi_head.forward_train(
            x_feats,
            proposal_feats,  # 叠加进patches的proposal_feats
            mask_preds,
            cls_scores,
            img_metas,
            gt_masks,
            gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_bboxes=gt_bboxes,
            gt_sem_seg=gt_sem_seg,
            gt_sem_cls=gt_sem_cls,
            imgs_whwh=None,
            gt_kernels=gt_kernels)

        losses.update(rpn_losses)
        return losses