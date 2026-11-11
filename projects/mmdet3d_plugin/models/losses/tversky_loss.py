import torch
import torch.nn as nn

from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class BinaryTverskyLoss(nn.Module):
    """Occupied-vs-free Tversky loss for Occ3D logits.

    BEVOCCHead2D passes flattened logits of shape [N, C], flattened semantic
    labels of shape [N], and the camera mask as the sample weight. This loss
    follows the Stage 4 safety/evaluator operating point:

        p_occ = 1 - softmax(logits)[free_id]

    and computes the binary occupied/free Tversky index inside valid,
    camera-visible voxels.
    """

    def __init__(self,
                 alpha_t=0.3,
                 beta_t=0.7,
                 gamma_t=1.0,
                 eps=1e-6,
                 free_id=17,
                 ignore_index=255,
                 loss_weight=1.0):
        super().__init__()
        self.alpha_t = float(alpha_t)
        self.beta_t = float(beta_t)
        self.gamma_t = float(gamma_t)
        self.eps = float(eps)
        self.free_id = int(free_id)
        self.ignore_index = int(ignore_index)
        self.loss_weight = float(loss_weight)

    def _flatten_logits(self, pred):
        if pred.dim() == 2:
            return pred
        if pred.shape[-1] > self.free_id:
            return pred.reshape(-1, pred.shape[-1])
        if pred.dim() >= 3 and pred.shape[1] > self.free_id:
            order = [0] + list(range(2, pred.dim())) + [1]
            return pred.permute(*order).reshape(-1, pred.shape[1])
        raise ValueError(
            f'Cannot infer class dimension for BinaryTverskyLoss from '
            f'prediction shape {tuple(pred.shape)} with free_id={self.free_id}')

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        del avg_factor, reduction_override, kwargs

        logits = self._flatten_logits(pred)
        target = target.reshape(-1).to(device=logits.device).long()
        num_classes = logits.shape[-1]

        valid = (target != self.ignore_index) & (target >= 0) \
            & (target < num_classes)
        if weight is not None:
            cam_weight = weight.reshape(-1).to(device=logits.device)
            valid = valid & (cam_weight > 0)
            cam_weight = cam_weight[valid].to(dtype=logits.dtype)
        else:
            cam_weight = torch.ones(
                int(valid.sum().item()), device=logits.device,
                dtype=logits.dtype)

        if valid.sum() == 0:
            return logits.sum() * 0.0

        logits = logits[valid]
        target = target[valid]
        prob = torch.softmax(logits, dim=-1)
        p_occ = 1.0 - prob[:, self.free_id]

        gt_occ = (target != self.free_id).to(dtype=logits.dtype) * cam_weight
        gt_free = (target == self.free_id).to(dtype=logits.dtype) * cam_weight

        tp = (p_occ * gt_occ).sum()
        fp = (p_occ * gt_free).sum()
        fn = ((1.0 - p_occ) * gt_occ).sum()

        index = tp / (tp + self.alpha_t * fp + self.beta_t * fn + self.eps)
        loss = 1.0 - index
        if self.gamma_t != 1.0:
            loss = loss.clamp(min=0.0).pow(self.gamma_t)
        return loss * self.loss_weight


@LOSSES.register_module()
class MultiClassTverskyLoss(nn.Module):
    """Multi-class (generalized) region-overlap Tversky for Occ3D logits.

    Fixes the degenerate *binary* occupied/free Tversky (BinaryTverskyLoss),
    which is blind to the 18-way semantics and collapses the mIoU. Following
    Salehi 2017, a soft Tversky index is computed PER CLASS from the full
    softmax and averaged over the classes present in the batch:

        p_c = softmax(logits)[:, c]      y_c = 1[target == c]
        TP_c = sum(p_c * y_c * w)        FP_c = sum(p_c * (1 - y_c) * w)
        FN_c = sum((1 - p_c) * y_c * w)
        TI_c = TP_c / (TP_c + alpha_t*FP_c + beta_t*FN_c + eps)
        loss = 1 - mean_{c present} TI_c

    beta_t > alpha_t weights FN above FP (recall / safety, same direction as the
    binary version). Signature matches BEVOCCHead2D.loss: pred [N, C] logits,
    target [N] labels, weight [N] = camera mask. Only classes present in the GT
    of the batch are averaged, so absent classes (TI ~= 0) do not bias the mean.
    """

    def __init__(self,
                 alpha_t=0.3,
                 beta_t=0.7,
                 gamma_t=1.0,
                 eps=1e-6,
                 num_classes=18,
                 ignore_index=255,
                 loss_weight=1.0):
        super().__init__()
        self.alpha_t = float(alpha_t)
        self.beta_t = float(beta_t)
        self.gamma_t = float(gamma_t)
        self.eps = float(eps)
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.loss_weight = float(loss_weight)

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        del avg_factor, reduction_override, kwargs

        logits = pred.reshape(-1, pred.shape[-1])
        num_classes = logits.shape[-1]
        target = target.reshape(-1).to(device=logits.device).long()

        valid = (target != self.ignore_index) & (target >= 0) \
            & (target < num_classes)
        if weight is not None:
            cam_weight = weight.reshape(-1).to(device=logits.device)
            valid = valid & (cam_weight > 0)
        if valid.sum() == 0:
            return logits.sum() * 0.0

        logits = logits[valid]
        target = target[valid]
        if weight is not None:
            cam_weight = cam_weight[valid].to(dtype=logits.dtype).unsqueeze(1)
        else:
            cam_weight = torch.ones(
                target.numel(), 1, device=logits.device, dtype=logits.dtype)

        prob = torch.softmax(logits, dim=-1)                       # [M, C]
        onehot = torch.zeros_like(prob)
        onehot.scatter_(1, target.unsqueeze(1), 1.0)               # [M, C]

        tp = (prob * onehot * cam_weight).sum(dim=0)               # [C]
        fp = (prob * (1.0 - onehot) * cam_weight).sum(dim=0)       # [C]
        fn = ((1.0 - prob) * onehot * cam_weight).sum(dim=0)       # [C]
        ti = tp / (tp + self.alpha_t * fp + self.beta_t * fn + self.eps)

        present = onehot.sum(dim=0) > 0                            # [C]
        if present.sum() == 0:
            return logits.sum() * 0.0
        loss = 1.0 - ti[present].mean()
        if self.gamma_t != 1.0:
            loss = loss.clamp(min=0.0).pow(self.gamma_t)
        return loss * self.loss_weight
