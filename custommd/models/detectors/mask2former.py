'''
Author: lifuguan 1002732355@qq.com
Date: 2024-01-10 12:46:54
LastEditors: lifuguan 1002732355@qq.com
LastEditTime: 2024-01-10 13:00:39
FilePath: /research_workspace/custommd/models/detectors/mask2former.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.builder import DETECTORS
from .maskformer import MaskFormer


@DETECTORS.register_module()
class Mask2Former(MaskFormer):
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
