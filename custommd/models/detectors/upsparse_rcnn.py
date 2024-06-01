import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import transforms
from mmdet.models.builder import DETECTORS

from mmdet.models.detectors import TwoStageDetector

from ..oursnet.utils import mask2bbox, sem2ins_masks, matrix_nms, center_of_mass, GaussianBlur

@DETECTORS.register_module()
class UPSparseRCNN(TwoStageDetector):
    r"""Implementation of `Sparse R-CNN: End-to-End Object Detection with
    Learnable Proposals <https://arxiv.org/abs/2011.12450>`_"""

    def __init__(self, *args, **kwargs):
        super(UPSparseRCNN, self).__init__(*args, **kwargs)
        assert self.with_rpn, 'Sparse R-CNN and QueryInst ' \
            'do not support external proposals'

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
        self.num_proposals = 100
        self.is_kernel2query = True
        if self.is_kernel2query is True:
            self.query_shuffle = False

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # pooling used for the query patch feature

            self.patch2query = nn.Linear(2048, 256)
            
            self.mask_ratio = 0.1

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
        patch_feats = self.extract_patches(img, gt_masks, img_metas)

        proposal_boxes, proposal_features, imgs_whwh = \
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
            proposal_features = proposal_features + refine_patch_feats[idx, ...].permute(1,0,2).contiguous()

        roi_losses = self.roi_head.forward_train(
            x,
            proposal_boxes,
            proposal_features,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_masks=gt_masks,
            imgs_whwh=imgs_whwh)
        return roi_losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.simple_test_rpn(x, img_metas)
        results = self.roi_head.simple_test(
            x,
            proposal_boxes,
            proposal_features,
            img_metas,
            imgs_whwh=imgs_whwh,
            rescale=rescale)
        return results

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        # backbone
        x = self.extract_feat(img)
        # rpn
        num_imgs = len(img)
        dummy_img_metas = [
            dict(img_shape=(800, 1333, 3)) for _ in range(num_imgs)
        ]
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.simple_test_rpn(x, dummy_img_metas)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposal_boxes,
                                               proposal_features,
                                               dummy_img_metas)
        return roi_outs
