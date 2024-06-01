import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import TwoStageDetector
from mmdet.utils import get_root_logger

from ..detectors.knet import KNet
from ..oursnet.utils import mask2bbox, sem2ins_masks, GaussianBlur
from .positional_encoding import PositionEmbeddingSine

@DETECTORS.register_module()
class UPKNet(KNet):

    def __init__(self,
                 *args,
                 num_proposals = 100,
                 query_shuffle=False,
                 is_kernel2query = True,
                 mask_ratio=0.1, 
                 **kwargs):
        super(UPKNet, self).__init__(*args, **kwargs)

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
            print("is_kernel2query is True. Use feature prompt.")
        else:
            print("is_kernel2query is False.")

        self.positional_encoder = PositionEmbeddingSine(num_pos_feats=256 // 2)

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

        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

        x = self.extract_feat(img)
        patch_feats, patch_pos, gt_poses = self.extract_patches(img, gt_masks, img_metas, x[-1])

        assert proposals is None, 'KNet does not support' \
                                  ' external proposals'
        assert gt_masks is not None

        # gt_masks and gt_semantic_seg are not padded when forming batch
        gt_masks_tensor = []
        gt_poses_tensor = []
        gt_sem_seg = []
        gt_sem_cls = []
        # batch_input_shape shoud be the same across images
        pad_H, pad_W = img_metas[0]['batch_input_shape']
        assign_H = pad_H // self.mask_assign_stride
        assign_W = pad_W // self.mask_assign_stride

        bs = len(img_metas)

        for i, (gt_mask, gt_pos) in enumerate(zip(gt_masks, gt_poses)):
            mask_tensor = gt_mask.to_tensor(torch.float, gt_labels[0].device)
            pos_tensor = gt_pos
            # if gt_mask.width != pad_W or gt_mask.height != pad_H:
            #     pad_wh = (0, pad_W - gt_mask.width, 0, pad_H - gt_mask.height)
            #     mask_tensor = F.pad(mask_tensor, pad_wh, value=0)

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

            if pos_tensor.shape[0] == 0:
                gt_poses_tensor.append(
                    pos_tensor.new_zeros(
                        (pos_tensor.size(0), assign_H, assign_W)))
            else:
                gt_poses_tensor.append(
                    F.interpolate(
                        pos_tensor[None], (assign_H, assign_W),
                        mode='bilinear',
                        align_corners=False)[0])
                
        gt_masks = gt_masks_tensor
        gt_poses = gt_poses_tensor
        rpn_results = self.rpn_head.forward_train(x, img_metas, gt_masks,
                                                  gt_labels, gt_poses, gt_sem_seg,
                                                  gt_sem_cls)
        (rpn_losses, proposal_feats, x_feats, mask_preds,
         cls_scores, pos_preds) = rpn_results

        if self.is_kernel2query is True:
            patch_feats = patch_feats.view(bs, 20, -1) \
                .repeat_interleave(self.num_proposals // 20, dim=1) \
                .permute(1, 0, 2) \
                .contiguous()
            patch_pos = patch_pos.view(bs, 20, -1) \
                .repeat_interleave(self.num_proposals // 20, dim=1) \
                .permute(1, 0, 2) \
                .contiguous()

            idx = torch.randperm(self.num_proposals*2) if self.query_shuffle else torch.arange(self.num_proposals*2)

            # NOTE query 
            # for training, it uses fixed number of query patches.
            mask_query_patch = (torch.rand(self.num_proposals, bs, 1, device=patch_feats.device) > self.mask_ratio).float()
            # NOTE mask some query patch and add query embedding
            refine_patch_feats = patch_feats * mask_query_patch
            refine_patch_pos = patch_pos * mask_query_patch
            refine_patch_feats = torch.cat([refine_patch_feats, refine_patch_pos], dim=0) # shape [200,bs,256]
            proposal_feats = proposal_feats + refine_patch_feats[idx, ...] \
                .permute(1,0,2).contiguous().unsqueeze(-1).unsqueeze(-1)

                

        # kernel_iter_update.py 
        losses = self.roi_head.forward_train(
            x_feats,
            proposal_feats,
            mask_preds,
            pos_preds,
            cls_scores,
            img_metas,
            gt_masks,
            gt_poses,
            gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_bboxes=gt_bboxes,
            gt_sem_seg=gt_sem_seg,
            gt_sem_cls=gt_sem_cls,
            imgs_whwh=None)

        losses.update(rpn_losses)
        return losses

  
    def extract_patches(self, imgs, gt_masks, img_metas, feats):
        """Directly extract features from the backbone+neck."""

        pos = self.positional_encoder(feats) # feats shape [bs,256,H,W]

        ins_per_img, pusedo_kernels, pos_kernels, gt_poses = [], [], [], []

        for i, (per_img_mask, img) in enumerate(zip(gt_masks, imgs)):
            if len(per_img_mask) == 0:
                print(img_metas[i])
                patch_feat_refine = torch.zeros((1,256), device=img.device)
                patch_pos_refine = torch.zeros((1,256), device=img.device)
                pusedo_kernels.append(patch_feat_refine)
                pos_kernels.append(patch_pos_refine)
                ins_per_img.append(1)
            else:
                mask_tensor = per_img_mask.to_tensor(torch.float, img.device)
                gt_pos = torch.zeros_like(mask_tensor)
                gt_boxes = mask2bbox(mask_tensor.bool()).long()
                boxes = (mask2bbox(mask_tensor.bool()) / 32).long()
                for j, (box, gt_box) in enumerate(zip(boxes, gt_boxes)):
                    if box[1] == box[3]:
                        box[3] += 2
                    if box[0] == box[2]:
                        box[2] += 2
                    patch_feat = feats[i, :, box[1]:box[3], box[0]:box[2]]
                    gt_pos[j, gt_box[1]:gt_box[3], gt_box[0]:gt_box[2]] = 1
                    patch_pos = pos[i, :, box[1]:box[3], box[0]:box[2]]

                    patch_feat_refine = self.avgpool(patch_feat).flatten(1).permute(1, 0)
                    patch_pos_refine = self.avgpool(patch_pos).flatten(1).permute(1, 0)
                    if torch.isnan(patch_feat_refine).any():
                        print("danger")
                    pusedo_kernels.append(patch_feat_refine)
                    pos_kernels.append(patch_pos_refine)

                ins_per_img.append(len(mask_tensor))
                gt_poses.append(gt_pos)
        # if torch.isnan(pusedo_kernels).any():
        #     print("danger")
        pusedo_kernels = torch.cat(pusedo_kernels, dim=0)
        pos_kernels = torch.cat(pos_kernels, dim=0)


        begin_i, expend_kernels, expend_kernels_pos = 0, [], []
        for end_i in ins_per_img:
            single_patch_feats = pusedo_kernels[begin_i:(begin_i+end_i)] \
                .repeat_interleave(20 // end_i, dim=0)
            single_patch_pos = pos_kernels[begin_i:(begin_i+end_i)] \
                .repeat_interleave(20 // end_i, dim=0)

            if 20 % end_i != 0:
                single_patch_feats = torch.cat((single_patch_feats, \
                    pusedo_kernels[begin_i:(begin_i+20 % end_i)]), dim=0)
                single_patch_pos = torch.cat((single_patch_pos, \
                    pos_kernels[begin_i:(begin_i+20 % end_i)]), dim=0)

            # 将kernels扩充到20个
            expend_kernels.append(single_patch_feats)
            expend_kernels_pos.append(single_patch_pos)
            begin_i += end_i
            
        expend_kernels = torch.cat(expend_kernels, dim=0)
        expend_kernels_pos = torch.cat(expend_kernels_pos, dim=0)


        return expend_kernels.detach(), expend_kernels_pos.detach(), gt_poses