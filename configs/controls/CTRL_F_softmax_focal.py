"""Multi-class softmax focal baseline.

Inherits the CE model's initialization and training schedule. Replaces CE
with (1-p_y)^2 * CE without class weighting or RiskOcc auxiliary terms."""

_base_ = ['../formal/T0_ce.py']

model = dict(
    occ_head=dict(
        loss_occ=dict(
            _delete_=True,
            type='SoftmaxFocalLoss',
            gamma=2.0,
            loss_weight=1.0,
            ignore_index=255,
        ),
    )
)
