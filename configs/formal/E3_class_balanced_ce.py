"""Formal E3: class-balanced CE baseline.

Identical to E1 (T0_ce) except the occupancy head enables ``class_balance``,
which weights the CE loss by ``1/log(nusc_class_frequencies + 0.001)``
(inverse-frequency balancing) and uses the matching weighted ``avg_factor``.
The det-pretrain ``load_from``, FlashOCC schedule, optimizer, runner, and EMA
hook are inherited from T0_ce unchanged. Only the loss balancing differs.
"""

_base_ = ['./T0_ce.py']

model = dict(
    occ_head=dict(
        class_balance=True,
    )
)
