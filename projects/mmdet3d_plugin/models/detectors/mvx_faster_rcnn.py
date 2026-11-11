# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet3d.models.builder import DETECTORS
from .mvx_two_stage import MVXTwoStageDetector_2


@DETECTORS.register_module()
class MVXFasterRCNN_2(MVXTwoStageDetector_2):
    """Multi-modality VoxelNet using Faster R-CNN."""

    def __init__(self, **kwargs):
        super(MVXFasterRCNN_2, self).__init__(**kwargs)


