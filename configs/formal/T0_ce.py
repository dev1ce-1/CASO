"""FlashOCC CE baseline initialized from BEVDet detection pretraining.
The 24-epoch schedule, optimizer and EMA hook are inherited unchanged.
"""

_base_ = ['../../projects/configs/flashocc/flashocc-r50.py']

load_from = "ckpts/bevdet-r50-cbgs.pth"
