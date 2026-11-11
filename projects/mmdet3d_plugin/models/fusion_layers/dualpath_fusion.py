import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from ..model_utils import BottleNeckASPP
from mmdet3d.models import FUSION_LAYERS
import pdb


@FUSION_LAYERS.register_module()
class DualpathFusion(BaseModule):
    def __init__(self,
                 in_channels,
                 stride=1,
                 norm_cfg=None,
                 init_cfg=None,
                 coeff_bias=True,
                 aspp_drop=0.1,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.stride = stride
        self.norm_cfg = norm_cfg
        self.kwargs = kwargs

        # build skip connection
        if self.stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, channels, kernel_size=1, stride=stride, bias=False),
                build_norm_layer(norm_cfg, channels)[1])
        else:
            self.downsample = nn.Identity()


        # aspp in global path
        self.aspp = BottleNeckASPP(inplanes=self.in_channels, norm_cfg=self.norm_cfg, dropout=aspp_drop)

        # soft weights for fusion
        self.combine_coeff = nn.Conv3d(self.in_channels, 1, kernel_size=1, bias=coeff_bias)

    def forward(self, x):
        # input_identity = x.clone()
        # x = self.input_conv(x)
        #
        # x_bev = x.mean(dim=-1)
        # batch_size = x_bev.shape[0]
        #
        # x = rearrange(x, 'b c x y z -> (b z) c x y')
        # x = torch.cat((x_bev, x), dim=0)
        # x = self.bev_encoder(x)  # relu output
        # x_bev, x = x[:batch_size], x[batch_size:]
        # x = rearrange(x, '(b z) c x y -> b c x y z', b=batch_size)

        x_bev = x[0]
        x_vox = x[1]
        input_identity = x[2]

        x_bev = self.aspp(x_bev)

        coeff = self.combine_coeff(x_vox).sigmoid()
        x = x_vox + coeff * x_bev.unsqueeze(2)

        return x + self.downsample(input_identity)