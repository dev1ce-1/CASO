"""CE plus multi-class Tversky baseline.

Inherits the CE baseline's detection pretraining, schedule, optimizer and EMA.
The occupancy loss combines cross-entropy and Tversky with unit coefficients.
Tversky uses alpha_t=0.3, beta_t=0.7 and gamma_t=1.0. Performance is reported
from the archived experimental results, not inferred from this configuration."""

_base_ = ['./T0_ce.py']

model = dict(
    occ_head=dict(
        loss_occ=dict(
            _delete_=True,
            type='CombinedCETverskyLoss',
            w_ce=1.0,
            w_tv=1.0,
            ce=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                ignore_index=255,
                loss_weight=1.0,
            ),
            tversky=dict(
                type='MultiClassTverskyLoss',
                alpha_t=0.3,
                beta_t=0.7,
                gamma_t=1.0,
                eps=1e-6,
                num_classes=18,
                ignore_index=255,
                loss_weight=1.0,
            ),
        ),
    )
)
