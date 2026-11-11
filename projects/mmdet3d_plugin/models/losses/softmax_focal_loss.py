import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss


@LOSSES.register_module()
class SoftmaxFocalLoss(nn.Module):
    """Multi-class softmax focal loss: FL = (1 - p_y)^gamma * CE.

    CTRL-F control baseline (hard-example mining without any RiskOcc term).
    mmdet's FocalLoss is sigmoid-only, so the softmax modulation is written
    here directly. No alpha / class weighting: with gamma=0 this reduces
    EXACTLY to CrossEntropyLoss(use_sigmoid=False) under the same
    weight / avg_factor reduction.

    Signature matches what BEVOCCHead2D.loss passes:
        pred [N, C] logits, target [N] labels, weight [N] = camera mask,
        avg_factor.
    """

    def __init__(self, gamma=2.0, loss_weight=1.0, ignore_index=255):
        super().__init__()
        self.gamma = float(gamma)
        self.loss_weight = float(loss_weight)
        self.ignore_index = int(ignore_index)

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        reduction = reduction_override if reduction_override else 'mean'
        ce = F.cross_entropy(pred, target, reduction='none',
                             ignore_index=self.ignore_index)
        # p_y = exp(-CE) exactly (CE = -log softmax(pred)[y]); ignored targets
        # have ce=0 -> p_y=1 -> modulator 0 -> zero loss, matching CE's ignore.
        p_y = torch.exp(-ce)
        loss = (1.0 - p_y).pow(self.gamma) * ce
        if weight is not None:
            weight = weight.float()
        loss = weight_reduce_loss(loss, weight, reduction=reduction,
                                  avg_factor=avg_factor)
        return self.loss_weight * loss
