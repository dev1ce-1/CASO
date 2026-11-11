import math
from functools import partial
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule

import torch
import torch.nn as nn
import torch.nn.functional as F

import spconv.pytorch as spconv
from spconv.pytorch import functional as Fsp

from mmdet3d.models.builder import MIDDLE_ENCODERS, BACKBONES

import copy


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_cfg=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        build_norm_layer(norm_cfg, out_channels)[1],
        nn.ReLU(inplace=True),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):

    def __init__(self, inplanes, planes, stride=1, norm_cfg=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                              indice_key=indice_key),
            build_norm_layer(norm_cfg, planes)[1],
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                              indice_key=indice_key),
            build_norm_layer(norm_cfg, planes)[1],
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.net(x)
        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))

        return out

@BACKBONES.register_module()
class SparseResnet3D(nn.Module):
    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            with_cp=False,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            sparse_shape_xyz=[200,200,16]
    ):
        super().__init__()

        assert len(num_layer) == len(stride)
        num_channels = [numC_input * 2 ** (i + 1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids

        self.sparse_shape_xyz = sparse_shape_xyz


        block = post_act_block
        self.conv1 = spconv.SparseSequential(
            block(numC_input, num_channels[0], 3, norm_cfg=norm_cfg, stride=1, padding=1, indice_key='spconv2',
                  conv_type='spconv'),
            SparseBasicBlock(num_channels[0], num_channels[0], norm_cfg=norm_cfg, indice_key='res1'),
            SparseBasicBlock(num_channels[0], num_channels[0], norm_cfg=norm_cfg, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            block(num_channels[0], num_channels[1], 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2',
                  conv_type='spconv'),
            SparseBasicBlock(num_channels[1], num_channels[1], norm_cfg=norm_cfg, indice_key='res2'),
            SparseBasicBlock(num_channels[1], num_channels[1], norm_cfg=norm_cfg, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            block(num_channels[1], num_channels[2], 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv3',
                  conv_type='spconv'),
            SparseBasicBlock(num_channels[2], num_channels[2], norm_cfg=norm_cfg, indice_key='res3'),
            SparseBasicBlock(num_channels[2], num_channels[2], norm_cfg=norm_cfg, indice_key='res3'),
        )

    def forward(self, x, coors, batch_size):
        """

        Args:
            x: 经过lss后未splat的voxel特征， (B, C, Dz, Dy, Dx)
            coors:
            batch_size:

        Returns:

        """
        # spconv encoding
        coors = coors.int()  # zyx 与 points的输入(xyz)相反
        # FIXME bs=1 hardcode
        input_sp_tensor = spconv.SparseConvTensor(x, coors, self.sparse_shape_xyz[::-1], batch_size)
        feats = []
        x_conv1 = self.conv1(input_sp_tensor)
        feats.append(x_conv1.dense())
        x_conv2 = self.conv2(x_conv1)
        feats.append(x_conv2.dense())
        x_conv3 = self.conv3(x_conv2)
        feats.append(x_conv3.dense())
        return feats
