"""Formal E5: RiskOcc ablation with beta=0 (no false-positive term).

Identical to E6 (T6_riskocc: alpha=1.0, beta=0.4) except ``risk_beta=0.0``,
which removes the near-field free-space false-positive penalty
(``loss_risk_fp``). All other RiskOcc terms (alpha=1.0, gamma=1.0,
lambda_distance=0.5, lambda_class=0.5), the det-pretrain load_from, schedule,
optimizer, runner, and EMA hook are inherited from T6_riskocc unchanged.
"""

_base_ = ['./T6_riskocc.py']

model = dict(
    occ_head=dict(
        risk_beta=0.0,
    )
)
