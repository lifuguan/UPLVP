'''
Author: lifuguan lifugan_10027@outlook.com
Date: 2022-06-08 17:12:20
LastEditors: lifuguan lifugan_10027@outlook.com
LastEditTime: 2022-06-19 12:44:25
FilePath: /research_workspace/custommd/models/knet/seg/iter.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import transforms


from mmseg.models.builder import HEADS, build_head
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from .utils import mask2bbox, GaussianBlur

@HEADS.register_module()
class UpIterativeDecodeHead(BaseDecodeHead):
    """Rethinking Atrous Convolution for Semantic Image Segmentation.

    This head is the implementation of `DeepLabV3
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        dilations (tuple[int]): Dilation rates for ASPP module.
            Default: (1, 6, 12, 18).
    """

    def __init__(self, num_stages, kernel_generate_head, kernel_update_head,
                 num_proposals = 150,
                 query_shuffle = False,
                 is_kernel2query = True,
                 **kwargs):
        super(BaseDecodeHead, self).__init__(**kwargs)
        assert num_stages == len(kernel_update_head)
        self.num_stages = num_stages
        self.kernel_generate_head = build_head(kernel_generate_head)
        self.kernel_update_head = nn.ModuleList()
        self.align_corners = self.kernel_generate_head.align_corners
        self.num_classes = self.kernel_generate_head.num_classes
        self.input_transform = self.kernel_generate_head.input_transform
        self.ignore_index = self.kernel_generate_head.ignore_index

        for head_cfg in kernel_update_head:
            self.kernel_update_head.append(build_head(head_cfg))

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

            self.patch2query = nn.Linear(2048, 512)
            
            self.mask_ratio = 0.1

    def forward_train(self, backbone, inputs, img, img_metas, gt_semantic_seg, train_cfg):
        self.backbone = backbone

        seg_logits = self.forward(inputs, img, gt_semantic_seg)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def mask2kernel(self, imgs, gt_masks):
        ins_per_img, pusedo_kernels = [], []

        for per_img_mask, img in zip(gt_masks, imgs):
            mask_tensor = per_img_mask.float()
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
                .repeat_interleave(10 // end_i, dim=0)
            
            if 10 % end_i != 0:
                single_patch_feats = torch.cat((single_patch_feats, \
                    pusedo_kernels[begin_i:10 % end_i]), dim=0)
            # 将kernels扩充到10个
            expend_kernels.append(single_patch_feats)

        expend_kernels = torch.cat(expend_kernels, dim=0)

        return expend_kernels.detach()



    def forward(self, inputs, img, gt_masks):
        """Forward function."""
        feats = self.kernel_generate_head._forward_feature(inputs)
        sem_seg = self.kernel_generate_head.cls_seg(feats)
        seg_kernels = self.kernel_generate_head.conv_seg.weight.clone()
        seg_kernels = seg_kernels[None].expand(
            feats.size(0), *seg_kernels.size())


        stage_segs = [sem_seg]
        for i in range(self.num_stages):
            sem_seg, seg_kernels = self.kernel_update_head[i](feats,
                                                              seg_kernels,
                                                              sem_seg)
            stage_segs.append(sem_seg)
        if self.training:
            return stage_segs
        # only return the prediction of the last stage during testing
        return stage_segs[-1]

    def losses(self, seg_logit, seg_label):
        losses = dict()
        for i, logit in enumerate(seg_logit):
            loss = self.kernel_generate_head.losses(logit, seg_label)
            for k, v in loss.items():
                losses[f'{k}.s{i}'] = v

        return losses
