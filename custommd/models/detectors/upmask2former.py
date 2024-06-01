'''
Author: lifuguan lifugan_10027@outlook.com
Date: 2022-07-14 16:29:23
LastEditors: lifuguan 1002732355@qq.com
LastEditTime: 2024-01-10 15:18:18
FilePath: /research_workspace/custommd/models/detectors/upmask2former.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn
from torchvision.transforms import transforms

from mmdet.models.builder import DETECTORS
from .mask2former import Mask2Former
from mmdet.models.detectors.single_stage import SingleStageDetector

from ..oursnet.utils import mask2bbox, GaussianBlur


@DETECTORS.register_module()
class UpMask2Former(Mask2Former):
    r"""Implementation of `Masked-attention Mask
    Transformer for Universal Image Segmentation
    <https://arxiv.org/pdf/2112.01527>`_."""

    def __init__(self,
                 backbone,
                 neck=None,
                 panoptic_head=None,
                 panoptic_fusion_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super().__init__(
            backbone,
            neck=neck,
            panoptic_head=panoptic_head,
            panoptic_fusion_head=panoptic_fusion_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
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

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # pooling used for the query patch feature

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_semantic_seg=None,
                      gt_bboxes_ignore=None,
                      **kargs):

        # add batch_input_shape in img_metas
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)

        patch_feats = self.extract_patches(img, gt_masks, img_metas, x[-1])

        losses = self.panoptic_head.forward_train(x, img_metas, gt_bboxes,
                                                  gt_labels, gt_masks,
                                                  gt_semantic_seg,
                                                  patch_feats,
                                                  gt_bboxes_ignore)

        return losses


    def extract_patches(self, imgs, gt_masks, img_metas, feats):
        """Directly extract features from the backbone+neck."""

        ins_per_img, pusedo_kernels = [], []

        for i, (per_img_mask, img) in enumerate(zip(gt_masks, imgs)):
            if len(per_img_mask) == 0:
                print(img_metas[i])
                patch_feat_refine = torch.zeros((1,2048), device=img.device)
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
            single_patch_feats = pusedo_kernels[begin_i:(begin_i+end_i)] \
                .repeat_interleave(20 // end_i, dim=0)
            
            if 20 % end_i != 0:
                single_patch_feats = torch.cat((single_patch_feats, \
                    pusedo_kernels[begin_i:(begin_i+20 % end_i)]), dim=0)
            # 将kernels扩充到20个
            expend_kernels.append(single_patch_feats)
            begin_i += end_i
            
        expend_kernels = torch.cat(expend_kernels, dim=0)


        return expend_kernels.detach()




    # def extract_patches(self, imgs, gt_masks, img_metas):
    #     """Directly extract features from the backbone+neck."""

    #     ins_per_img, pusedo_kernels = [], []

    #     for per_img_mask, img in zip(gt_masks, imgs):
    #         mask_tensor = per_img_mask.to_tensor(torch.float, img.device)
    #         boxes = mask2bbox(mask_tensor.bool()).long()
    #         for box in boxes:
    #             patch = img[:, box[1]:box[3], box[0]:box[2]]

    #             patch = self.get_query_tarnsforms(patch)[None,...].to(device=mask_tensor.device)
    #             patch_feat = self.backbone(patch)[-1]
    #             pusedo_kernels.append(patch_feat)
    #         ins_per_img.append(len(mask_tensor))

    #     pusedo_kernels = torch.cat(pusedo_kernels, dim=0)


    #     begin_i, expend_kernels = 0, []
    #     for end_i in ins_per_img:
    #         single_patch_feats = pusedo_kernels[begin_i:end_i] \
    #             .repeat_interleave(20 // end_i, dim=0)
            
    #         if 20 % end_i != 0:
    #             single_patch_feats = torch.cat((single_patch_feats, \
    #                 pusedo_kernels[begin_i:20 % end_i]), dim=0)
    #         # 将kernels扩充到20个
    #         expend_kernels.append(single_patch_feats)

    #     expend_kernels = torch.cat(expend_kernels, dim=0)


    #     return expend_kernels.detach()