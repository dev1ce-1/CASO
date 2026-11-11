"""Temporal CE baseline (non-stereo BEVDepth4D), detection-pretrained."""

_base_ = ['../../projects/configs/flashocc/flashocc-r50-4d-stereo.py']

grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 45.0, 0.5],
}
numC_Trans = 80

# --- stereo -> non-stereo depth-temporal (architecture matched to
#     ckpts/bevdet-r50-4d-depth-cbgs.pth: out_indices=(2,3), FPN out=512, VT in=512) ---
model = dict(
    type='BEVDepth4DOCC',
    # stereo needs stage-0 for the cost volume; non-stereo drops it
    img_backbone=dict(out_indices=(2, 3)),
    # depth pretrain uses a 512-wide FPN / depthnet
    img_neck=dict(out_channels=512),
    img_view_transformer=dict(
        _delete_=True,
        type='LSSViewTransformerBEVDepth',
        grid_config=grid_config,
        input_size=(256, 704),
        in_channels=512,
        out_channels=numC_Trans,
        sid=False,
        loss_depth_weight=0.05,
        depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=96),
        downsample=16),
    # occ_head inherited from base = BEVOCCHead2D (CrossEntropyLoss, use_mask=True) -> pure CE
)

# dataset: turn OFF stereo reference-frame loading (temporal-only)
data = dict(
    train=dict(stereo=False),
    val=dict(stereo=False),
    test=dict(stereo=False),
)

load_from = 'ckpts/bevdet-r50-4d-depth-cbgs.pth'
resume_from = None
checkpoint_config = dict(interval=1, max_keep_ckpts=2)
