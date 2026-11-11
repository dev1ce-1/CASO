from typing import List

import torch
from mmcv.cnn import ConvModule
from torch import nn

from mmdet3d.models.builder import FUSION_LAYERS

@FUSION_LAYERS.register_module()
class ConvFuser(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.module_list = nn.Sequential(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        self.prepare_pts_inputs = ConvModule(
            64*16,
            64*4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d', ),
            act_cfg=None)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        if len(inputs[1].shape) == 5:
            B, C, W, H, D = inputs[1].shape
            inputs[1] = self.prepare_pts_inputs(inputs[1].view(B, C*D ,W, H))
        return self.module_list(torch.cat(inputs, dim=1))
