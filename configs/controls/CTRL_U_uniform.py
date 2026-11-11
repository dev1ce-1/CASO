"""Uniform RiskOcc control.

Inherits the formal RiskOcc model and coefficients, replacing the contextual
field with R(v)=1. This shares coefficients with RiskOcc; it does not match
the mean magnitude of the occupied-voxel penalty."""

_base_ = ['../formal/T6_riskocc.py']

model = dict(
    occ_head=dict(
        risk_mode='uniform',
    )
)
