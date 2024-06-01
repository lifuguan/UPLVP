'''
Author: your name
Date: 2022-04-11 10:47:45
LastEditTime: 2022-04-13 20:11:51
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/custommd/models/oursnet/projection_loss.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class ProjectLoss(nn.Module):

    def __init__(self, loss_weight=1.0):
        super(ProjectLoss, self).__init__()

        self.loss_weight = loss_weight
    
    def forward(self, mask_scores, gt_bitmasks):
        mask_scores = mask_scores.sigmoid()
        mask_losses_y = dice_coefficient(
            mask_scores.max(dim=1, keepdim=True)[0],
            gt_bitmasks.max(dim=1, keepdim=True)[0]
        )
        mask_losses_x = dice_coefficient(
            mask_scores.max(dim=2, keepdim=True)[0],
            gt_bitmasks.max(dim=2, keepdim=True)[0]
        )
        return (mask_losses_x + mask_losses_y).mean() * self.loss_weight

def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss

@LOSSES.register_module()
class AvgLoss(nn.Module):

    def __init__(self, loss_weight=1.0):
        super(AvgLoss, self).__init__()

        self.loss_weight = loss_weight
    
    def forward(self, mask_scores, gt_bitmasks):
        mask_scores = mask_scores.sigmoid()
        mask_losses_y = dice_coefficient(
            mask_scores.mean(dim=1, keepdim=True)[0],
            gt_bitmasks.mean(dim=1, keepdim=True)[0]
        )
        mask_losses_x = dice_coefficient(
            mask_scores.mean(dim=2, keepdim=True)[0],
            gt_bitmasks.mean(dim=2, keepdim=True)[0]
        )
        return (mask_losses_x + mask_losses_y).mean() * self.loss_weight

def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss