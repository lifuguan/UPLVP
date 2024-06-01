import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import transforms
from mmdet.models.builder import DETECTORS

from mmdet.models.detectors import TwoStageDetector
from mmdet.utils import get_root_logger

from ..oursnet.utils import mask2bbox, sem2ins_masks, matrix_nms, center_of_mass, GaussianBlur

# @ALGORITHMS.register_module()
@DETECTORS.register_module()
class UPKNet(TwoStageDetector):

    def __init__(self,
                 *args,
                 num_thing_classes=80,
                 num_stuff_classes=53,
                 mask_assign_stride=4,
                 thing_label_in_seg=0,
                 text_repr_type='quad',
                 num_proposals,
                 query_shuffle=False,
                 is_kernel2query = True,
                 mask_ratio=0.1, 
                 **kwargs):
        super(UPKNet, self).__init__(*args, **kwargs)
        assert self.with_rpn, 'OursNet does not support external proposals'
        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.mask_assign_stride = mask_assign_stride
        self.thing_label_in_seg = thing_label_in_seg
        logger = get_root_logger()
        logger.info(f'Model: \n{self}')
        assert text_repr_type in ['quad', 'poly']
        self.text_repr_type = text_repr_type

        """build guided kernel approach"""
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


    def extract_feat_train(self, imgs, gt_masks, img_metas):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(imgs)
        if self.with_neck:
            x = self.neck(x)

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


        return x, expend_kernels.detach()

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
        
        x, patch_feats = self.extract_feat_train(img, gt_masks, img_metas)
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
            mask_tensor = gt_mask.to_tensor(torch.float, img.device)

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


            # if object query shuffle, we shuffle the index of object query embedding,
            # which simulate to adding patch feature to object query randomly.
            idx = torch.randperm(self.num_proposals) if self.query_shuffle else torch.arange(self.num_proposals)

            # NOTE query 
            # for training, it uses fixed number of query patches.
            mask_query_patch = (torch.rand(self.num_proposals, bs, 1, device=patch_feats.device) > self.mask_ratio).float()
            # NOTE mask some query patch and add query embedding
            refine_patch_feats = patch_feats * mask_query_patch
            proposal_feats = proposal_feats + refine_patch_feats[idx, ...] \
                .permute(1,0,2).contiguous().unsqueeze(-1).unsqueeze(-1)

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
            imgs_whwh=None)

        losses.update(rpn_losses)
        return losses

    def extract_feat_test(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)

    def simple_test(self, img, img_metas, rescale=False):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        bs = len(img_metas)

        # x = self.extract_feat_train(img)
        # backbone的输出；kernel的初始化
        rpn_results = self.rpn_head.simple_test_rpn(x, img_metas)
        (proposal_feats, x_feats, mask_preds, cls_scores,
         seg_preds) = rpn_results

        # if self.is_kernel2query is True:
        #     patch_feature_gt = self.avgpool(patch_feats).flatten(1)
        #     patch_feats = self.patch2query(patch_feature_gt) \
        #         .view(bs, 20, -1) \
        #         .repeat_interleave(self.num_proposals // 20, dim=1) \
        #         .permute(1, 0, 2) \
        #         .contiguous()


        #     # if object query shuffle, we shuffle the index of object query embedding,
        #     # which simulate to adding patch feature to object query randomly.
        #     idx = torch.randperm(self.num_proposals) if self.query_shuffle else torch.arange(self.num_proposals)

        #     # NOTE query 
        #     # for training, it uses fixed number of query patches.
        #     mask_query_patch = (torch.rand(self.num_proposals, bs, 1, device=patch_feats.device) > self.mask_ratio).float()
        #     # NOTE mask some query patch and add query embedding
        #     refine_patch_feats = patch_feats * mask_query_patch
        #     proposal_feats = proposal_feats + refine_patch_feats[idx, ...] \
        #         .permute(1,0,2).contiguous().unsqueeze(-1).unsqueeze(-1)
        # 关键点
        segm_results = self.roi_head.simple_test(
            x_feats,
            proposal_feats,
            mask_preds,
            cls_scores,
            img_metas,
            imgs_whwh=None,
            rescale=rescale)
        return segm_results

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        # backbone
        x = self.extract_feat(img)
        # rpn
        num_imgs = len(img)
        dummy_img_metas = [
            dict(img_shape=(800, 1333, 3)) for _ in range(num_imgs)
        ]
        rpn_results = self.rpn_head.simple_test_rpn(x, dummy_img_metas)
        (proposal_feats, x_feats, mask_preds, cls_scores,
         seg_preds) = rpn_results
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x_feats, proposal_feats,
                                               dummy_img_metas)
        return roi_outs
