"""Temporal RiskOcc: same architecture and initialization as the temporal CE baseline."""

_base_ = ['./E1_T_ce.py']

model = dict(
    occ_head=dict(
        _delete_=True,
        type='BEVOCCHead2D_RiskOcc',
        in_dim=256,
        out_dim=256,
        Dz=16,
        use_mask=True,
        num_classes=18,
        use_predicter=True,
        class_balance=False,
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            ignore_index=255,
            loss_weight=1.0),
        risk_alpha=1.0,
        risk_beta=0.4,
        risk_gamma=1.0,
        risk_lambda_distance=0.5,
        risk_lambda_class=0.5,
        risk_free_id=17,
        risk_near_distance=20.0,
        risk_distance_max=40.0,
        risk_pc_range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
        risk_voxel_size=0.4,
    )
)
