import torch.nn as nn

from mmdet.models.builder import LOSSES, build_loss


@LOSSES.register_module()
class CombinedCETverskyLoss(nn.Module):
    """CE + multi-class Tversky joint occupancy loss (thin wrapper).

    loss = w_ce * CrossEntropyLoss + w_tv * MultiClassTverskyLoss

    CE carries the 18-way semantic learning (pure multi-class Tversky alone tops
    out ~20 mIoU, 12 pts below CE); the multi-class Tversky adds region-overlap /
    recall shaping (beta_t=0.7 > alpha_t=0.3, FN weighted above FP for safety).
    This only *composes* the two existing loss classes -- it does NOT
    reimplement Tversky.

    Signature matches what BEVOCCHead2D.loss passes:
        pred [N, C] logits, target [N] labels, weight [N] = camera mask,
        avg_factor (forwarded to CE only; the Tversky term self-normalizes).
    """

    def __init__(self,
                 ce=dict(type='CrossEntropyLoss', use_sigmoid=False,
                         ignore_index=255, loss_weight=1.0),
                 tversky=dict(type='MultiClassTverskyLoss', alpha_t=0.3,
                              beta_t=0.7, gamma_t=1.0, eps=1e-6, num_classes=18,
                              ignore_index=255, loss_weight=1.0),
                 w_ce=1.0,
                 w_tv=1.0):
        super().__init__()
        self.ce = build_loss(ce)
        self.tversky = build_loss(tversky)
        self.w_ce = float(w_ce)
        self.w_tv = float(w_tv)

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        ce_loss = self.ce(pred, target, weight=weight, avg_factor=avg_factor,
                          reduction_override=reduction_override)
        tv_loss = self.tversky(pred, target, weight=weight)
        return self.w_ce * ce_loss + self.w_tv * tv_loss
