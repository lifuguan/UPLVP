'''
Author: your name
Date: 2022-04-11 10:00:53
LastEditTime: 2022-04-11 11:01:37
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/custommd/models/oursnet/loss.py
'''


import imp
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES
from .utils import unfold_wo_center

@LOSSES.register_module()
class PairwiseLoss(nn.Module):

    def __init__(self, loss_weight=1.0, pairwise_dilation=2):
        super(PairwiseLoss, self).__init__()
        self.loss_weight = loss_weight
        self.pairwise_dilation = pairwise_dilation

    def forward(self, mask_logits, pairwise_size):
        assert mask_logits.dim() == 4

        log_fg_prob = F.logsigmoid(mask_logits)
        log_bg_prob = F.logsigmoid(-mask_logits)

        log_fg_prob_unfold = unfold_wo_center(
            log_fg_prob, kernel_size=pairwise_size,
            dilation=self.pairwise_dilation
        )
        log_bg_prob_unfold = unfold_wo_center(
            log_bg_prob, kernel_size=pairwise_size,
            dilation=self.pairwise_dilation
        )

        # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
        # we compute the the probability in log space to avoid numerical instability
        log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
        log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

        max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
        log_same_prob = torch.log(
            torch.exp(log_same_fg_prob - max_) +
            torch.exp(log_same_bg_prob - max_)
        ) + max_

            # loss = -log(prob)
        return -log_same_prob[:, 0] * self.loss_weight

