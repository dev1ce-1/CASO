import warnings
from copy import deepcopy

from mmdet3d.models.builder import BACKBONES
from mmdet.models.backbones import SwinTransformer


import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer, trunc_normal_init, build_conv_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.cnn.utils.weight_init import constant_init
from mmcv.runner import _load_checkpoint
from mmcv.runner.base_module import BaseModule, ModuleList
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
import torch.utils.checkpoint as checkpoint

from mmseg.ops import resize
from mmdet3d.utils import get_root_logger
from mmdet3d.models.builder import BACKBONES
from mmcv.cnn.bricks.registry import ATTENTION
from torch.nn.modules.utils import _pair as to_2tuple
from collections import OrderedDict
@BACKBONES.register_module()
class GHH_Backbone(SwinTransformer):
    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=96,
                 patch_size=4,
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 strides=(4, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 pretrain_style='official',
                 pretrained=None,
                 init_cfg=None,
                 with_cp=True,
                 return_stereo_feat=False,
                 output_missing_index_as_none=False,
                 frozen_stages=-1,
                 **kwargs):
        super(GHH_Backbone, self).__init__()

        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        assert pretrain_style in ['official', 'mmcls'], 'We only support load '
        'official ckpt and mmcls ckpt.'

        if isinstance(pretrained, str) or pretrained is None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
        else:
            raise TypeError('pretrained must be a str or None')

        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed
        self.pretrain_style = pretrain_style
        self.pretrained = pretrained
        self.init_cfg = init_cfg

        self.frozen_stages = frozen_stages

        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=strides[0],
            pad_to_patch_size=True,
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)

        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, num_patches, embed_dims)))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # stochastic depth
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule

        self.stages = ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=strides[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None)
            else:
                downsample = None

            stage = SwinBlockSequence(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                feedforward_channels=mlp_ratio * in_channels,
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[:depths[i]],
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                init_cfg=None,
                with_cp=with_cp)
            self.stages.append(stage)

            dpr = dpr[depths[i]:]
            if downsample:
                in_channels = downsample.out_channels

        self.num_features = [int(embed_dims * 2 ** i) for i in range(num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)
        self.output_missing_index_as_none = output_missing_index_as_none

        self._freeze_stages()
        self.return_stereo_feat = return_stereo_feat

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.use_abs_pos_embed:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.drop_after_pos.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.stages[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self):
        if self.pretrained is None:
            super().init_weights()
            if self.use_abs_pos_embed:
                trunc_normal_init(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, Linear):
                    trunc_normal_init(m.weight, std=.02)
                    if m.bias is not None:
                        constant_init(m.bias, 0)
                elif isinstance(m, LayerNorm):
                    constant_init(m.bias, 0)
                    constant_init(m.weight, 1.0)
        elif isinstance(self.pretrained, str):
            logger = get_root_logger()
            ckpt = _load_checkpoint(
                self.pretrained, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt

            if self.pretrain_style == 'official':
                state_dict = swin_convert(state_dict)

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            # if list(state_dict.keys())[0].startswith('backbone.'):
            #     state_dict = {k[9:]: v for k, v in state_dict.items()}
            # reshape absolute position embedding
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    logger.warning('Error in loading absolute_pos_embed, pass')
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                        N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

            # interpolate position bias table if needed
            relative_position_bias_table_keys = [
                k for k in state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                L1, nH1 = table_pretrained.size()
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    logger.warning(f'Error in loading {table_key}, pass')
                else:
                    if L1 != L2:
                        S1 = int(L1 ** 0.5)
                        S2 = int(L2 ** 0.5)
                        table_pretrained_resized = resize(
                            table_pretrained.permute(1, 0).reshape(
                                1, nH1, S1, S1),
                            size=(S2, S2),
                            mode='bicubic')
                        state_dict[table_key] = table_pretrained_resized.view(
                            nH2, L2).permute(1, 0).contiguous()

            # load state_dict
            self.load_state_dict(state_dict, False)

    def forward(self, x):
        x = self.patch_embed(x)

        hw_shape = (self.patch_embed.DH, self.patch_embed.DW)
        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i == 0 and self.return_stereo_feat:
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)
            elif self.output_missing_index_as_none:
                outs.append(None)
        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(GHH_Backbone, self).train(mode)
        self._freeze_stages()