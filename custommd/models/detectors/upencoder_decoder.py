'''
Author: lifuguan lifugan_10027@outlook.com
Date: 2022-06-09 13:45:02
LastEditors: lifuguan lifugan_10027@outlook.com
LastEditTime: 2022-06-19 12:30:25
FilePath: /research_workspace/custommd/models/detectors/upencoder_decoder.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import torch

from mmdet.models.builder import DETECTORS

from mmseg.models.segmentors import EncoderDecoder
from mmseg.core import add_prefix


@DETECTORS.register_module()
class UpEncoderDecoder(EncoderDecoder):
    def __init__(self, *args, **kwargs):
        super(UpEncoderDecoder, self).__init__(*args, **kwargs)

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
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
        dict[str, Tensor]: a dictionary of loss components
        """
        gt_sem_seg = torch.zeros(
            (len(gt_masks), 1, gt_masks[0].height, gt_masks[0].width), 
             dtype=torch.int64, device=img.device)
        for i, gt_segs in enumerate(gt_masks):
            for j, gt_seg in enumerate(gt_segs):
                mask_sum = gt_seg * (j*10+1)
                gt_sem_seg[i] += torch.tensor(mask_sum).to(img.device)
        
        x = self.extract_feat(img)


        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img, img_metas,
                                                      gt_sem_seg)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_sem_seg)
            losses.update(loss_aux)

        return losses

    def _decode_head_forward_train(self, x, img, img_metas, gt_masks):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(self.backbone, x, img, img_metas,
                                                     gt_masks,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses