import numpy as np
import torch
import torch.nn.functional as F

from mmdet.core import AssignResult, BaseAssigner
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs.builder import MATCH_COST, build_match_cost

from .utils import unfold_wo_center

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

@MATCH_COST.register_module()
class DiceCost(object):
    """DiceCost.

     Args:
         weight (int | float, optional): loss_weight
         pred_act (bool): Whether to activate the prediction
            before calculating cost

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import BBoxL1Cost
         >>> import torch
         >>> self = BBoxL1Cost()
         >>> bbox_pred = torch.rand(1, 4)
         >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(bbox_pred, gt_bboxes, factor)
         tensor([[1.6172, 1.6422]])
    """

    def __init__(self,
                 weight=1.,
                 pred_act=False,
                 act_mode='sigmoid',
                 eps=1e-3):
        self.weight = weight
        self.pred_act = pred_act
        self.act_mode = act_mode
        self.eps = eps

    def dice_loss(cls, input, target, eps=1e-3):
        input = input.reshape(input.size()[0], -1)
        target = target.reshape(target.size()[0], -1).float()
        # einsum saves 10x memory
        # a = torch.sum(input[:, None] * target[None, ...], -1)
        a = torch.einsum('nh,mh->nm', input, target) # (100,200,304) * (2,200,304) -> (100, 2)
        b = torch.sum(input * input, 1) + eps
        c = torch.sum(target * target, 1) + eps
        d = (2 * a) / (b[:, None] + c[None, ...])
        # 1 is a constance that will not affect the matching, so ommitted
        return -d

    def __call__(self, mask_preds, gt_masks):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].

        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        if self.pred_act and self.act_mode == 'sigmoid':
            mask_preds = mask_preds.sigmoid()
        elif self.pred_act:
            mask_preds = mask_preds.softmax(dim=0)
        dice_cost = self.dice_loss(mask_preds, gt_masks, self.eps)
        return dice_cost * self.weight



@BBOX_ASSIGNERS.register_module()
class MaskHungarianAssigner(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classfication cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 boundary_cost=None,
                 dice_cost=dict(),
                 pairwise_dilation = 2,
                 project_weight=5.0,
                 avg_weight=0.5,
                 topk=1):
        self.cls_cost = build_match_cost(cls_cost)
        if boundary_cost is not None:
            self.boundary_cost = build_match_cost(boundary_cost)
        else:
            self.boundary_cost = None
        self.topk = topk

        self.dice_cost = build_match_cost(dice_cost)
        self.pairwise_dilation = pairwise_dilation
        self.project_weight = project_weight
        self.avg_weight = avg_weight

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               img_meta=None,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # classification and bboxcost.
        if self.cls_cost.weight != 0 and cls_pred is not None:
            cls_cost = self.cls_cost(cls_pred, gt_labels)
        else:
            cls_cost = 0
        if self.avg_weight != 0:
            # 从mask cost修改成avg
            reg_cost = self.compute_avg_term(bbox_pred, gt_bboxes) * self.avg_weight
        else:
            reg_cost = 0
        if self.project_weight != 0:
            # 从dice修改成projection
            dice_cost = self.compute_project_term(bbox_pred, gt_bboxes) * self.project_weight
        else:
            dice_cost = 0
        if self.boundary_cost is not None and self.boundary_cost.weight != 0:
            b_cost = self.boundary_cost(bbox_pred, gt_bboxes)
        else:
            b_cost = 0
        cost = cls_cost + reg_cost + dice_cost + b_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        if self.topk == 1:
            matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        else:
            topk_matched_row_inds = []
            topk_matched_col_inds = []
            for i in range(self.topk):
                matched_row_inds, matched_col_inds = linear_sum_assignment(
                    cost)
                topk_matched_row_inds.append(matched_row_inds)
                topk_matched_col_inds.append(matched_col_inds)
                cost[matched_row_inds] = 1e10
            matched_row_inds = np.concatenate(topk_matched_row_inds)
            matched_col_inds = np.concatenate(topk_matched_col_inds)

        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)


    def compute_project_term(self, mask_scores, gt_bitmasks):
        mask_scores = mask_scores.sigmoid()
        mask_losses_y = self.dice_cost(
            mask_scores.max(dim=1, keepdim=True)[0],
            gt_bitmasks.max(dim=1, keepdim=True)[0]
        )
        mask_losses_x = self.dice_cost(
            mask_scores.max(dim=2, keepdim=True)[0],
            gt_bitmasks.max(dim=2, keepdim=True)[0]
        )
        return (mask_losses_x + mask_losses_y)

    def compute_avg_term(self, mask_scores, gt_bitmasks):
        mask_scores = mask_scores.sigmoid()
        mask_losses_y = self.dice_cost(
            mask_scores.mean(dim=1, keepdim=True),
            gt_bitmasks.mean(dim=1, keepdim=True)
        )
        mask_losses_x = self.dice_cost(
            mask_scores.mean(dim=2, keepdim=True),
            gt_bitmasks.mean(dim=2, keepdim=True)
        )
        return (mask_losses_x + mask_losses_y)


    def compute_pairwise_term(self, mask_logits, pairwise_size):
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
        return -log_same_prob[:, 0]

    def dice_coefficient(self, x, target):
        eps = 1e-5
        n_inst = x.size(0)
        x = x.reshape(n_inst, -1)
        target = target.reshape(n_inst, -1)
        intersection = (x * target).sum(dim=1)
        union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
        loss = 1. - (2 * intersection / union)
        return loss

