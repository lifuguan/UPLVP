
import torch
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from PIL import ImageFilter


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def median_blur(input: torch.Tensor, kernel_size: Tuple[int, int]) -> torch.Tensor:
    """Blur an image using the median filter."""
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

    padding: Tuple[int, int] = _compute_zero_padding(kernel_size)

    # prepare kernel
    kernel: torch.Tensor = get_binary_kernel2d(kernel_size).to(input)
    b, c, h, w = input.shape

    # map the local window to single vector
    features: torch.Tensor = F.conv2d(input.reshape(b * c, 1, h, w), kernel, padding=padding, stride=1)
    features = features.view(b, c, -1, h, w)  # BxCx(K_h * K_w)xHxW

    # compute the median along the feature axis
    median: torch.Tensor = torch.median(features, dim=2)[0]

    return median

def _compute_zero_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
    r"""Utility function that computes zero padding tuple."""
    computed: List[int] = [(k - 1) // 2 for k in kernel_size]
    return computed[0], computed[1]

def get_binary_kernel2d(window_size: Tuple[int, int]) -> torch.Tensor:
    r"""Create a binary kernel to extract the patches. If the window size
    is HxW will create a (H*W)xHxW kernel.
    """
    window_range: int = window_size[0] * window_size[1]
    kernel: torch.Tensor = torch.zeros(window_range, window_range)
    for i in range(window_range):
        kernel[i, i] += 1.0
    return kernel.view(window_range, 1, window_size[0], window_size[1])
    
def sem2ins_masks(gt_sem_seg,
                  num_thing_classes=80):
    """Convert semantic segmentation mask to binary masks

    Args:
        gt_sem_seg (torch.Tensor): Semantic masks to be converted.
            [0, num_thing_classes-1] is the classes of things,
            [num_thing_classes:] is the classes of stuff.
        num_thing_classes (int, optional): Number of thing classes.
            Defaults to 80.

    Returns:
        tuple[torch.Tensor]: (mask_labels, bin_masks).
            Mask labels and binary masks of stuff classes.
    """
    # gt_sem_seg is zero-started, where zero indicates the first class
    # since mmdet>=2.17.0, see more discussion in
    # https://mmdetection.readthedocs.io/en/latest/conventions.html#coco-panoptic-dataset  # noqa
    classes = torch.unique(gt_sem_seg)
    # classes ranges from 0 - N-1, where the class IDs in
    # [0, num_thing_classes - 1] are IDs of thing classes
    masks = []
    labels = []

    for i in classes:
        # skip ignore class 255 and "thing classes" in semantic seg
        if i == 255 or i < num_thing_classes:
            continue
        labels.append(i)
        masks.append(gt_sem_seg == i)

    if len(labels) > 0:
        labels = torch.stack(labels)
        masks = torch.cat(masks)
    else:
        labels = gt_sem_seg.new_zeros(size=[0])
        masks = gt_sem_seg.new_zeros(
            size=[0, gt_sem_seg.shape[-2], gt_sem_seg.shape[-1]])
    return labels.long(), masks.float()


# Copyright (c) OpenMMLab. All rights reserved.
import torch


def mask_matrix_nms(masks,
                    labels,
                    scores,
                    filter_thr=-1,
                    nms_pre=-1,
                    max_num=-1,
                    kernel='gaussian',
                    sigma=2.0,
                    mask_area=None):
    """Matrix NMS for multi-class masks.

    Args:
        masks (Tensor): Has shape (num_instances, h, w)
        labels (Tensor): Labels of corresponding masks,
            has shape (num_instances,).
        scores (Tensor): Mask scores of corresponding masks,
            has shape (num_instances).
        filter_thr (float): Score threshold to filter the masks
            after matrix nms. Default: -1, which means do not
            use filter_thr.
        nms_pre (int): The max number of instances to do the matrix nms.
            Default: -1, which means do not use nms_pre.
        max_num (int, optional): If there are more than max_num masks after
            matrix, only top max_num will be kept. Default: -1, which means
            do not use max_num.
        kernel (str): 'linear' or 'gaussian'.
        sigma (float): std in gaussian method.
        mask_area (Tensor): The sum of seg_masks.

    Returns:
        tuple(Tensor): Processed mask results.

            - scores (Tensor): Updated scores, has shape (n,).
            - labels (Tensor): Remained labels, has shape (n,).
            - masks (Tensor): Remained masks, has shape (n, w, h).
            - keep_inds (Tensor): The indices number of
                the remaining mask in the input mask, has shape (n,).
    """
    assert len(labels) == len(masks) == len(scores)
    if len(labels) == 0:
        return scores.new_zeros(0), labels.new_zeros(0), masks.new_zeros(
            0, *masks.shape[-2:]), labels.new_zeros(0)
    if mask_area is None:
        mask_area = masks.sum((1, 2)).float()
    else:
        assert len(masks) == len(mask_area)

    # sort and keep top nms_pre
    scores, sort_inds = torch.sort(scores, descending=True)

    keep_inds = sort_inds
    if nms_pre > 0 and len(sort_inds) > nms_pre:
        sort_inds = sort_inds[:nms_pre]
        keep_inds = keep_inds[:nms_pre]
        scores = scores[:nms_pre]
    masks = masks[sort_inds]
    mask_area = mask_area[sort_inds]
    labels = labels[sort_inds]

    num_masks = len(labels)
    flatten_masks = masks.reshape(num_masks, -1).float()
    # inter.
    inter_matrix = torch.mm(flatten_masks, flatten_masks.transpose(1, 0))
    expanded_mask_area = mask_area.expand(num_masks, num_masks)
    # Upper triangle iou matrix.
    iou_matrix = (inter_matrix /
                  (expanded_mask_area + expanded_mask_area.transpose(1, 0) -
                   inter_matrix)).triu(diagonal=1)
    # label_specific matrix.
    expanded_labels = labels.expand(num_masks, num_masks)
    # Upper triangle label matrix.
    label_matrix = (expanded_labels == expanded_labels.transpose(
        1, 0)).triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(num_masks,
                                           num_masks).transpose(1, 0)

    # IoU decay
    decay_iou = iou_matrix * label_matrix

    # Calculate the decay_coefficient
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou**2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou**2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError(
            f'{kernel} kernel is not supported in matrix nms!')
    # update the score.
    scores = scores * decay_coefficient


    # sort and keep top max_num
    scores, sort_inds = torch.sort(scores, descending=True)
    keep_inds = keep_inds[sort_inds]
    if max_num > 0 and len(sort_inds) > max_num:
        sort_inds = sort_inds[:max_num]
        keep_inds = keep_inds[:max_num]
        scores = scores[:max_num]
    masks = masks[sort_inds]
    labels = labels[sort_inds]

    if filter_thr > 0:
        keep = scores >= filter_thr
        if not keep.any():
            return scores[0].unsqueeze(0), labels[0].unsqueeze(0), masks[0].unsqueeze(0), keep_inds[0].unsqueeze(0)  # 使得蒙版若没有高于阈值则输出最大值
        keep_inds = keep_inds[keep]
        masks = masks[keep]
        scores = scores[keep]
        labels = labels[keep]

    return scores, labels, masks, keep_inds

def matrix_nms(seg_masks, cate_labels, cate_scores, kernel='gaussian', sigma=2.0, sum_masks=None):
    """Matrix NMS for multi-class masks.
    Args:
        seg_masks (Tensor): shape (n, h, w)
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gauss' 
        sigma (float): std in gaussian method
        sum_masks (Tensor): The sum of seg_masks
    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []
    if sum_masks is None:
        sum_masks = seg_masks.sum((1, 2)).float()
    seg_masks = seg_masks.reshape(n_samples, -1).float()
    # inter.
    inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
    # union.
    sum_masks_x = sum_masks.expand(n_samples, n_samples)
    # iou.
    iou_matrix = (inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)).triu(diagonal=1)
    # label_specific matrix.
    cate_labels_x = cate_labels.expand(n_samples, n_samples)
    label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

    # IoU decay 
    decay_iou = iou_matrix * label_matrix

    # matrix nms
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1-decay_iou)/(1-compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError

    # update the score.
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update


def center_of_mass(bitmasks):
    _, h, w = bitmasks.size()
    ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
    xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

    m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
    m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
    m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
    center_x = m10 / m00
    center_y = m01 / m00
    return center_x, center_y
    
def unfold_wo_center(x, kernel_size, dilation):
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )

    # remove the center pixels
    size = kernel_size ** 2
    unfolded_x = torch.cat((
        unfolded_x[:, :, :size // 2],
        unfolded_x[:, :, size // 2 + 1:]
    ), dim=2)

    return unfolded_x


def mask2bbox(masks):
    """Obtain tight bounding boxes of binary masks.

    Args:
        masks (Tensor): Binary mask of shape (n, h, w).

    Returns:
        Tensor: Bboxe with shape (n, 4) of \
            positive region in binary mask.
    """
    N = masks.shape[0]
    bboxes = masks.new_zeros((N, 4), dtype=torch.float32)
    x_any = torch.any(masks, dim=1)
    y_any = torch.any(masks, dim=2)
    for i in range(N):
        x = torch.where(x_any[i, :])[0]
        y = torch.where(y_any[i, :])[0]
        if len(x) > 0 and len(y) > 0:
            bboxes[i, :] = bboxes.new_tensor(
                [x[0], y[0], x[-1] + 1, y[-1] + 1])

    return bboxes