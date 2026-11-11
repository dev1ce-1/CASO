"""Formal T6: RiskOcc alpha=1.0, beta=0.4.

Only the occupancy head type and RiskOcc loss-term hyperparameters differ
from T0. The FlashOCC schedule, optimizer, runner, and EMA hook are inherited
unchanged.
"""

_base_ = ['./T0_ce.py']

model = dict(
    occ_head=dict(
        type='BEVOCCHead2D_RiskOcc',
        risk_alpha=1.0,
        risk_beta=0.4,
        risk_gamma=1.0,
        risk_lambda_distance=0.5,
        risk_lambda_class=0.5,
    )
)
