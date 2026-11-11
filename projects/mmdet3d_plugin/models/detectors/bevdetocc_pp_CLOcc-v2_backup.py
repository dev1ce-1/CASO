# Copyright (c) Phigent Robotics. All rights reserved.
from mmcv.cnn import ConvModule, normal_init
from mmcv.runner import force_fp32, BaseModule

from .bevdet_occ import BEVStereo4DOCC
from ...ops import TRTBEVPoolv2
from .bevdet import BEVDet
from .bevdepth import BEVDepth
from .bevdepth4d import BEVDepth4D
from .bevstereo4d import BEVStereo4D
from mmdet3d.models import DETECTORS
from mmdet3d.models.builder import build_head
import torch.nn.functional as F
from mmdet3d.core import bbox3d2result
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from ...ops import nearest_assign
# pool = ThreadPool(processes=4)  # �����̳߳�
from mmdet3d.models import builder
from torch import nn

from mmdet3d.models import CenterPoint

import warnings
from os import path as osp

import mmcv
import torch
from mmcv.ops import Voxelization
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from mmdet.core import multi_apply
from mmdet3d.models import builder
from mmdet3d.models.builder import DETECTORS
from mmdet3d.models.detectors import Base3DDetector

from einops import rearrange

from ..model_utils.ops.modules.ms_deform_attn import MSDeformAttn, LearnedPositionalEncoding3D

# for pano
grid_config_occ = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 45.0, 1.0],
}
# det
det_class_name = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
    'barrier']

# occ
occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

det_ind = [2, 3, 4, 5, 6, 7, 9, 10]
occ_ind = [5, 3, 0, 4, 6, 7, 2, 1]
detind2occind = {
    0:4,
    1:10,
    2:9,
    3:3,
    4:5,
    5:2,
    6:6,
    7:7,
    8:8,
    9:1,
}
occind2detind = {
    4:0,
    10:1,
    9:2,
    3:3,
    5:4,
    2:5,
    6:6,
    7:7,
    8:8,
    1:9,
}
occind2detind_cuda = [-1, -1, 5, 3, 0, 4, 6, 7, -1, 2, 1]

inst_occ = np.ones([200, 200, 16])*0

import torch
X1, Y1, Z1 = 200, 200, 16
coords_x = torch.arange(X1).float()
coords_y = torch.arange(Y1).float()
coords_z = torch.arange(Z1).float()
coords = torch.stack(torch.meshgrid([coords_x, coords_y, coords_z])).permute(1, 2, 3, 0)  # W, H, D, 3
# coords = coords.cpu().numpy()
st = [grid_config_occ['x'][0], grid_config_occ['y'][0], grid_config_occ['z'][0]]
sx = [grid_config_occ['x'][2], grid_config_occ['y'][2], 0.4]

@DETECTORS.register_module()
class BEVDetOCCPP(CenterPoint):
    """
    简单的pointpillars+flashocc的融合版本
    支持增加辅助检测头，以及对image和lidar的bev执行自定义融合
    """
    def __init__(self,
                 img_backbone,
                 img_neck,
                 img_view_transformer=None,
                 img_bev_encoder_backbone=None,
                 img_bev_encoder_neck=None,
                 fuse=None,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_backbone=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 occ_head=None,
                 upsample=False,
                 train_cfg=None,
                 test_cfg=None,
                 use_transfusion=False,
                 aux_centerness_head=None,
                 **kwargs):
        super(BEVDetOCCPP, self).__init__(**kwargs)
        if pts_voxel_layer:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        if pts_voxel_encoder:
            self.pts_voxel_encoder = builder.build_voxel_encoder(
                pts_voxel_encoder)
        if pts_middle_encoder:
            self.pts_middle_encoder = builder.build_middle_encoder(
                pts_middle_encoder)

        self.pts_backbone = builder.build_backbone(pts_backbone) if pts_backbone is not None else None
        self.pts_neck = builder.build_neck(pts_neck) if pts_neck is not None else None
        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)

        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)


        if fuse:
            self.fuse = builder.build_fusion_layer(fuse)

        if img_view_transformer:
            self.img_view_transformer = builder.build_neck(img_view_transformer)
        if img_bev_encoder_backbone:
            self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone)
        if img_bev_encoder_neck:
            self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)

        self.occ_head = builder.build_head(occ_head)

        # self.train_cfg = train_cfg
        # self.test_cfg = test_cfg
        self.upsample = upsample

        self.use_transfusion = use_transfusion

        self.aux_centerness_head=None
        if aux_centerness_head:
            # train_cfg = kwargs['train_cfg']
            # test_cfg = kwargs['test_cfg']
            pts_train_cfg = train_cfg.pts if train_cfg else None
            aux_centerness_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            aux_centerness_head.update(test_cfg=pts_test_cfg)
            self.aux_centerness_head = build_head(aux_centerness_head)
    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def extract_pts_feat(self, pts, img_feats, img_metas,):
        """Extract features of points."""
        # if not self.with_pts_bbox:
        #     return None
        # print(f"init:{pts}")
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors
                                                )
        batch_size = coors[-1, 0] + 1

        # for sparse_encoder
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        # print(f"after_middler:{x}")
        if self.pts_backbone is not None:
            x = self.pts_backbone(x)
        # print(f"after_backbone:{x}")
        if self.pts_neck is not None:
            x = self.pts_neck(x)
        # print(f"after_neck:{x}")
        return x


    def image_encoder(self, img, stereo=False):
        """
        Args:
            img: (B, N, 3, H, W)
            stereo: bool
        Returns:
            x: (B, N, C, fH, fW)
            stereo_feat: (B*N, C_stereo, fH_stereo, fW_stereo) / None
        """
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            x = x[1:]
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x, stereo_feat

    @force_fp32()
    def bev_encoder(self, x):
        """
        Args:
            x: (B, C, Dy, Dx)
        Returns:
            x: (B, C', 2*Dy, 2*Dx)
        """

        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x

    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        assert len(inputs) == 7
        B, N, C, H, W = inputs[0].shape
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs

        sensor2egos = sensor2egos.view(B, N, 4, 4)
        ego2globals = ego2globals.view(B, N, 4, 4)

        # calculate the transformation from adj sensor to key ego
        keyego2global = ego2globals[:, 0,  ...].unsqueeze(1)    # (B, 1, 4, 4)
        global2keyego = torch.inverse(keyego2global.double())   # (B, 1, 4, 4)
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()     # (B, N_views, 4, 4)
        sensor2keyegos = sensor2keyegos.float()

        return [imgs, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda]

    def extract_img_feat(self, img_inputs, img_metas, **kwargs):
        """ Extract features of images.
        img_inputs:
            imgs:  (B, N_views, 3, H, W)
            sensor2egos: (B, N_views, 4, 4)
            ego2globals: (B, N_views, 4, 4)
            intrins:     (B, N_views, 3, 3)
            post_rots:   (B, N_views, 3, 3)
            post_trans:  (B, N_views, 3)
            bda_rot:  (B, 3, 3)
        Returns:
            x: [(B, C', H', W'), ]
            depth: (B*N, D, fH, fW)
        """
        img_inputs = self.prepare_inputs(img_inputs)
        x, _ = self.image_encoder(img_inputs[0])    # x: (B, N, C, fH, fW)
        x, depth = self.img_view_transformer([x] + img_inputs[1:7])
        # print(f"x.shape: {x.shape}") # 4,64,200,200 for flashocc

        # x: (B, C, Dy, Dx) for flashocc   /    (B,C,Dz,Dy,Dx) for bevdet
        # depth: (B*N, D, fH, fW)

        return [x], depth

    def extract_feat(self, points, img_inputs, img_metas, **kwargs):
        """Extract features from images and points."""
        """
        points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
        img_inputs:
                imgs:  (B, N_views, 3, H, W)        
                sensor2egos: (B, N_views, 4, 4)
                ego2globals: (B, N_views, 4, 4)
                intrins:     (B, N_views, 3, 3)
                post_rots:   (B, N_views, 3, 3)
                post_trans:  (B, N_views, 3)
                bda_rot:  (B, 3, 3)
        """
        img_feats, depth = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        return img_feats, pts_feats, depth

    def custom_extract_feat(self, points, img_inputs, img_metas, **kwargs):
        img_inputs = self.prepare_inputs(img_inputs)
        x, _ = self.image_encoder(img_inputs[0])  # x: (B, N, C, fH, fW)
        img_feats = x
        B, N, C, fH, fW = img_feats.shape
        img_feats=img_feats.view(B* N, C, fH, fW)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        return img_feats,pts_feats

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        if not self.use_transfusion:
            img_feats, pts_feats, depth = self.extract_feat(
                points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)

            fuse_feats = [*img_feats, pts_feats]

            fuse_feats = self.fuse(fuse_feats)

        else:
            img_feats, pts_feats = self.custom_extract_feat(
                points, img_inputs=img_inputs, img_metas=img_metas, **kwargs
            )
            fuse_feats = self.fuse(pts_feats,img_feats,img_metas)

        fuse_feats = self.bev_encoder(fuse_feats)

        losses = dict()
        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)

        occ_bev_feature = fuse_feats
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        loss_occ = self.forward_occ_train(occ_bev_feature, voxel_semantics, mask_camera)
        losses.update(loss_occ)

        if self.aux_centerness_head is not None:
            losses_aux_centerness = self.forward_aux_centerness_train([occ_bev_feature], gt_bboxes_3d,
                                                                      gt_labels_3d, img_metas,
                                                                      gt_bboxes_ignore)
            losses.update(losses_aux_centerness)

        return losses

    def forward_occ_train(self, img_feats, voxel_semantics, mask_camera):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        outs = self.occ_head(img_feats)
        # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.occ_head.loss(
            outs,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
        )
        return loss_occ

    def forward_aux_centerness_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        outs = self.aux_centerness_head(pts_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.aux_centerness_head.loss(*loss_inputs)
        return losses

    def simple_test_aux_centerness(self, x, img_metas, rescale=False, **kwargs):
        """Test function of point cloud branch."""
        # outs = self.aux_centerness_head(x)
        tx = self.aux_centerness_head.shared_conv(x[0])     # (B, C'=share_conv_channel, H, W)
        outs_inst_center_reg = self.aux_centerness_head.task_heads[0].reg(tx)
        outs_inst_center_height = self.aux_centerness_head.task_heads[0].height(tx)
        outs_inst_center_heatmap = self.aux_centerness_head.task_heads[0].heatmap(tx)
        outs = ([{
            "reg" : outs_inst_center_reg,
            "height" : outs_inst_center_height,
            "heatmap" : outs_inst_center_heatmap,
        }],)

        # # bbox_list = self.aux_centerness_head.get_bboxes(
        # #     outs, img_metas, rescale=rescale)
        # # bbox_results = [
        # #     bbox3d2result(bboxes, scores, labels)
        # #     for bboxes, scores, labels in bbox_list
        # # ]
        ins_cen_list = self.aux_centerness_head.get_centers(
            outs, img_metas, rescale=rescale)
        # return bbox_results, ins_cen_list
        return None, ins_cen_list

    def forward_test(self,
                     points=None,
                     img_inputs=None,
                     img_metas=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)

        if not self.use_transfusion:
            img_feats, pts_feats, depth = self.extract_feat(
                points, img_inputs=img, img_metas=img_metas, **kwargs)

            fuse_feats = [*img_feats, pts_feats]

            fuse_feats = self.fuse(fuse_feats)

        else:
            img_feats, pts_feats = self.custom_extract_feat(
                points, img_inputs=img, img_metas=img_metas, **kwargs
            )
            fuse_feats = self.fuse(pts_feats, img_feats, img_metas)

        fuse_feats = self.bev_encoder(fuse_feats)

        occ_bev_feature = fuse_feats
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        occ_list = self.simple_test_occ(occ_bev_feature, img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list

    def simple_test_occ(self, img_feats, img_metas=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        outs = self.occ_head(img_feats)
        if not hasattr(self.occ_head, "get_occ_gpu"):
            occ_preds = self.occ_head.get_occ(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        else:
            occ_preds = self.occ_head.get_occ_gpu(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_preds

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        outs = self.occ_head(occ_bev_feature)
        return outs


class RadarConvFuser(BaseModule):
    def __init__(self, in_channels: int, out_channels: int, deconv_blocks: int) -> None:
        super(RadarConvFuser, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.fuse_conv_ghh = nn.Sequential(
            nn.Conv2d(sum(in_channels), out_channels*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU(True)
        )
        deconv = []
        # deconv_in = [sum(in_channels) + out_channels]
        deconv_in = [sum(in_channels)]
        deconv_out = [out_channels]
        for i in range(deconv_blocks - 1):
            deconv_in.append(out_channels)
            deconv_out.append(out_channels)
        for i in range(deconv_blocks):
            deconv.append(nn.Sequential(
                nn.Conv2d(deconv_in[i], deconv_out[i], 3, padding=1, bias=False),
                nn.BatchNorm2d(deconv_out[i]),
                nn.ReLU(True))
            )
        self.deconv = nn.ModuleList(deconv)

    def init_weights(self):
        super().init_weights()
        normal_init(self.fuse_conv, mean=0, std=0.001)
        for i in enumerate(self.deconv):
            normal_init(i, mean=0, std=0.001)

    def forward(self, input1, input2) -> torch.Tensor:  #img,pts
        res = torch.cat((input1, input2), dim=1)  #512
        res2 = res.clone() #512
        out = self.fuse_conv_ghh(res) #512
        # out = self.fuse_conv(res) #256
        out = out + res2  #512
        # out = torch.cat([out, res2], dim=1)
        for layer in self.deconv:
            out = layer(out)
        return out


@DETECTORS.register_module()
class RCFUSION(BEVDetOCCPP):
    """
    参考RCBEVDET的融合思路
    使用交叉注意力融合雷达和图像的bev特征
    """
    def __init__(self,
                 bev_size=128,
                 imc=256, rac=256,  # im ra 特征维度
                 freeze_img=False,
                 freeze_radar=False,
                 **kwargs):
        super(RCFUSION, self).__init__(**kwargs)
        # === 新增：dense BEV query，大小就是 (1, H*W, C) ===
        self.bev_query = nn.Parameter(
            torch.zeros(1, bev_size * bev_size, imc)
        )
        nn.init.xavier_uniform_(self.bev_query)  # 随便一个合理初始化

        # 下面是你原来的内容
        self.bev_size = bev_size

        self.DeformAttn1 = MSDeformAttn( d_model=256, n_levels=1, n_heads=8, n_points=8) #d_model=256, n_levels=1, n_heads=8, n_points=4
        self.DeformAttn2 = MSDeformAttn( d_model=256, n_levels=1, n_heads=8, n_points=8) #d_model=256, n_levels=1, n_heads=8, n_points=4
        self.DeformAttn3 = MSDeformAttn( d_model=256, n_levels=1, n_heads=8, n_points=8) #d_model=256, n_levels=1, n_heads=8, n_points=4
        self.DeformAttn4 = MSDeformAttn( d_model=256, n_levels=1, n_heads=8, n_points=8) #d_model=256, n_levels=1, n_heads=8, n_points=4
        patch_row = bev_size #// 4 pretrain_img_size[0] patch_size
        patch_col = bev_size #// 4
        num_patches = patch_row * patch_col
        self.LearnedPositionalEncoding1=LearnedPositionalEncoding3D(num_feats=imc//2, row_num_embed=bev_size, col_num_embed=bev_size)
        self.LearnedPositionalEncoding2=LearnedPositionalEncoding3D(num_feats=imc//2, row_num_embed=bev_size, col_num_embed=bev_size)
        self.LearnedPositionalEncoding3=LearnedPositionalEncoding3D(num_feats=imc//2, row_num_embed=bev_size, col_num_embed=bev_size)
        self.LearnedPositionalEncoding4=LearnedPositionalEncoding3D(num_feats=imc//2, row_num_embed=bev_size, col_num_embed=bev_size)
        self.tmp_deconv = nn.Sequential(
        nn.Conv2d(512, 256, 3, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(True)
        )


        self.radar_reduc_conv = ConvModule(
                rac,
                imc,  #rac change imc
                kernel_size=3,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='ReLU'),
                inplace=False)

        self.RadarConvFuser_fuse = RadarConvFuser(in_channels = (imc,imc),out_channels = imc,deconv_blocks = 3)

        self.freeze_img=freeze_img
        self.freeze_radar=freeze_radar

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='2d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    def extract_img_feat(self, img_inputs, img_metas, **kwargs):
        """ Extract features of images.
        img_inputs:
            imgs:  (B, N_views, 3, H, W)
            sensor2egos: (B, N_views, 4, 4)
            ego2globals: (B, N_views, 4, 4)
            intrins:     (B, N_views, 3, 3)
            post_rots:   (B, N_views, 3, 3)
            post_trans:  (B, N_views, 3)
            bda_rot:  (B, 3, 3)
        Returns:
            x: [(B, C', H', W'), ]
            depth: (B*N, D, fH, fW)
        """
        img_inputs = self.prepare_inputs(img_inputs)
        x, _ = self.image_encoder(img_inputs[0])    # x: (B, N, C, fH, fW)
        x, depth = self.img_view_transformer([x] + img_inputs[1:7])

        # x = self.bev_encoder(x)

        # print(f"x.shape: {x.shape}") # 4,64,200,200 for flashocc

        # x: (B, C, Dy, Dx) for flashocc   /    (B,C,Dz,Dy,Dx) for bevdet
        # depth: (B*N, D, fH, fW)

        return [x], depth

    def extract_feat(self, points, img_inputs, img_metas, **kwargs):
        img_feats, depth = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)

        if type(pts_feats) is not list:
            pts_feats = [pts_feats]

        fusion_feats = []
        bev_height = img_feats[0].shape[2]
        bev_width = img_feats[0].shape[3]

        for i in range(0, len(img_feats)):

            if hasattr(self, 'DeformAttn1') and self.DeformAttn1 is not None:
                pts_feats[i] = self.radar_reduc_conv(pts_feats[i])  # pts_feats 由 rac --> imc
                pts_feats_tmp = pts_feats[i].detach()
                img_feats_tmp = img_feats[i].detach()
                pts_feats = rearrange(pts_feats[i], 'b c h w -> b (h w) c')
                img_feats = rearrange(img_feats[i], 'b c h w -> b (h w) c')

                device = torch.device("cuda")  # Get the CUDA device
                mask = torch.zeros(1, 1, self.bev_size, self.bev_size).to(device)
                pos1 = self.LearnedPositionalEncoding1(mask)
                pos2 = self.LearnedPositionalEncoding2(mask)

                reference_point1 = self.get_reference_points(self.bev_size, self.bev_size)
                reference_point2 = self.get_reference_points(self.bev_size, self.bev_size)

                fusion_f1 = self.DeformAttn1(query=self.with_pos_embed(pts_feats, pos1),  #img
                                             reference_points=reference_point1,
                                             input_flatten=self.with_pos_embed(img_feats, pos2),
                                             input_spatial_shapes=torch.tensor([(self.bev_size, self.bev_size)]).to(
                                                 device),
                                             input_level_start_index=torch.tensor(
                                                 [0, self.bev_size * self.bev_size]).to(device),
                                             input_padding_mask=None)
                fusion_f2 = self.DeformAttn2(query=self.with_pos_embed(img_feats, pos2),   #pts
                                             reference_points=reference_point2,
                                             input_flatten=self.with_pos_embed(pts_feats, pos1),
                                             input_spatial_shapes=torch.tensor([(self.bev_size, self.bev_size)]).to(
                                                 device),
                                             input_level_start_index=torch.tensor(
                                                 [0, self.bev_size * self.bev_size]).to(device),
                                             input_padding_mask=None)
                fusion_f1 = rearrange(fusion_f1, 'b (h w) c -> b c h w', h=self.bev_size, w=self.bev_size)
                fusion_f2 = rearrange(fusion_f2, 'b (h w) c -> b c h w', h=self.bev_size, w=self.bev_size)
                pts_fusion_feature_tmp = torch.cat((pts_feats_tmp, fusion_f1), dim=1)
                img_fusion_feature_tmp = torch.cat((img_feats_tmp, fusion_f2), dim=1)
                pts_fusion_feature = self.tmp_deconv(pts_fusion_feature_tmp)
                img_fusion_feature = self.tmp_deconv(img_fusion_feature_tmp)
                fusion_f = self.RadarConvFuser_fuse(pts_fusion_feature, img_fusion_feature)

                # fusion_f = self.RadarConvFuser_fuse(fusion_f1, fusion_f2)
                """
                :param query                       (N, Length_{query}, C)
                :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                                or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
                :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
                :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
                :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
                :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

                :return output                     (N, Length_{query}, C)
                """
            fusion_feats.append(fusion_f)  # cat

        return fusion_feats, pts_feats, depth

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)

        fuse_feats, pts_feats, depth = self.extract_feat(
                points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        losses = dict()
        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)

        occ_bev_feature = fuse_feats[0]
        occ_bev_feature = self.bev_encoder(occ_bev_feature)
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        loss_occ = self.forward_occ_train(occ_bev_feature, voxel_semantics, mask_camera)
        losses.update(loss_occ)

        if self.aux_centerness_head is not None:
            losses_aux_centerness = self.forward_aux_centerness_train([occ_bev_feature], gt_bboxes_3d,
                                                                      gt_labels_3d, img_metas,
                                                                      gt_bboxes_ignore)
            losses.update(losses_aux_centerness)

        return losses


    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)

        fuse_feats, pts_feats, depth = self.extract_feat(
                points, img_inputs=img, img_metas=img_metas, **kwargs)

        occ_bev_feature = fuse_feats[0]
        occ_bev_feature = self.bev_encoder(occ_bev_feature)
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        occ_list = self.simple_test_occ(occ_bev_feature, img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list


@DETECTORS.register_module()
class RCFUSION4D(BEVStereo4DOCC):
    def __init__(self,
                 # img_backbone,
                 # img_neck,
                 # img_view_transformer=None,
                 # img_bev_encoder_backbone=None,
                 # img_bev_encoder_neck=None,
                 fuse=None,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_backbone=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 occ_head=None,
                 upsample=False,
                 train_cfg=None,
                 test_cfg=None,
                 use_transfusion=False,
                 aux_centerness_head=None,
                 bev_size=128,
                 imc=256, rac=256,  # im ra 特征维度
                 freeze_img=False,
                 freeze_radar=False,
                 **kwargs,
                 ):
        super().__init__(occ_head,upsample,
                         **kwargs)

        self.bev_size = bev_size

        self.DeformAttn1 = MSDeformAttn( d_model=imc, n_levels=1, n_heads=8, n_points=8) #d_model=256, n_levels=1, n_heads=8, n_points=4
        self.DeformAttn2 = MSDeformAttn( d_model=imc, n_levels=1, n_heads=8, n_points=8) #d_model=256, n_levels=1, n_heads=8, n_points=4
        self.DeformAttn3 = MSDeformAttn( d_model=imc, n_levels=1, n_heads=8, n_points=8) #d_model=256, n_levels=1, n_heads=8, n_points=4
        self.DeformAttn4 = MSDeformAttn( d_model=imc, n_levels=1, n_heads=8, n_points=8) #d_model=256, n_levels=1, n_heads=8, n_points=4
        patch_row = bev_size #// 4 pretrain_img_size[0] patch_size
        patch_col = bev_size #// 4
        num_patches = patch_row * patch_col
        self.LearnedPositionalEncoding1=LearnedPositionalEncoding3D(num_feats=imc//2, row_num_embed=bev_size, col_num_embed=bev_size)
        self.LearnedPositionalEncoding2=LearnedPositionalEncoding3D(num_feats=imc//2, row_num_embed=bev_size, col_num_embed=bev_size)
        self.LearnedPositionalEncoding3=LearnedPositionalEncoding3D(num_feats=imc//2, row_num_embed=bev_size, col_num_embed=bev_size)
        self.LearnedPositionalEncoding4=LearnedPositionalEncoding3D(num_feats=imc//2, row_num_embed=bev_size, col_num_embed=bev_size)

        self.radar_reduc_conv = ConvModule(
                rac,
                imc,  #rac change imc
                kernel_size=3,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='ReLU'),
                inplace=False)

        self.RadarConvFuser_fuse = RadarConvFuser(in_channels = (imc,imc),out_channels = 256,deconv_blocks = 3)

        self.freeze_img=freeze_img
        self.freeze_radar=freeze_radar

        if pts_voxel_layer:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        if pts_voxel_encoder:
            self.pts_voxel_encoder = builder.build_voxel_encoder(
                pts_voxel_encoder)
        if pts_middle_encoder:
            self.pts_middle_encoder = builder.build_middle_encoder(
                pts_middle_encoder)

        self.pts_backbone = builder.build_backbone(pts_backbone) if pts_backbone is not None else None
        self.pts_neck = builder.build_neck(pts_neck) if pts_neck is not None else None
        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)

        # if img_backbone:
        #     self.img_backbone = builder.build_backbone(img_backbone)
        # if img_neck is not None:
        #     self.img_neck = builder.build_neck(img_neck)


        if fuse:
            self.fuse = builder.build_fusion_layer(fuse)

        # if img_view_transformer:
        #     self.img_view_transformer = builder.build_neck(img_view_transformer)
        # if img_bev_encoder_backbone:
        #     self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone)
        # if img_bev_encoder_neck:
        #     self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)

        self.occ_head = builder.build_head(occ_head)

        # self.train_cfg = train_cfg
        # self.test_cfg = test_cfg
        self.upsample = upsample

        self.use_transfusion = use_transfusion

        self.aux_centerness_head=None
        if aux_centerness_head:
            # train_cfg = kwargs['train_cfg']
            # test_cfg = kwargs['test_cfg']
            pts_train_cfg = train_cfg.pts if train_cfg else None
            aux_centerness_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            aux_centerness_head.update(test_cfg=pts_test_cfg)
            self.aux_centerness_head = build_head(aux_centerness_head)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='2d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    def extract_pts_feat(self, pts, img_feats, img_metas,):
        """Extract features of points."""
        # if not self.with_pts_bbox:
        #     return None
        # print(f"init:{pts}")
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors
                                                )
        batch_size = coors[-1, 0] + 1

        # for sparse_encoder
        x = self.pts_middle_encoder(voxel_features, coors, batch_size,return_bev_feat=True)
        # print(f"after_middler:{x}")
        if self.pts_backbone is not None:
            x = self.pts_backbone(x)
        # print(f"after_backbone:{x}")
        if self.pts_neck is not None:
            x = self.pts_neck(x)
        # print(f"after_neck:{x}")
        return x

    def extract_feat(self, points, img_inputs, img_metas, **kwargs):
        img_feats, depth = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)

        if type(pts_feats) is not list:
            pts_feats = [pts_feats]

        fusion_feats = []
        bev_height = img_feats[0].shape[2]
        bev_width = img_feats[0].shape[3]

        for i in range(0, len(img_feats)):

            if hasattr(self, 'DeformAttn1') and self.DeformAttn1 is not None:
                pts_feats[i] = self.radar_reduc_conv(pts_feats[i])  # pts_feats 由 rac --> imc
                pts_feats = rearrange(pts_feats[i], 'b c h w -> b (h w) c')
                img_feats = rearrange(img_feats[i], 'b c h w -> b (h w) c')

                device = torch.device("cuda")  # Get the CUDA device
                mask = torch.zeros(1, 1, self.bev_size, self.bev_size).to(device)
                pos1 = self.LearnedPositionalEncoding1(mask)
                pos2 = self.LearnedPositionalEncoding2(mask)

                reference_point1 = self.get_reference_points(self.bev_size, self.bev_size)
                reference_point2 = self.get_reference_points(self.bev_size, self.bev_size)

                fusion_f1 = self.DeformAttn1(query=self.with_pos_embed(pts_feats, pos1),
                                             reference_points=reference_point1,
                                             input_flatten=self.with_pos_embed(img_feats, pos2),
                                             input_spatial_shapes=torch.tensor([(self.bev_size, self.bev_size)]).to(
                                                 device),
                                             input_level_start_index=torch.tensor(
                                                 [0, self.bev_size * self.bev_size]).to(device),
                                             input_padding_mask=None)
                fusion_f2 = self.DeformAttn2(query=self.with_pos_embed(img_feats, pos2),
                                             reference_points=reference_point2,
                                             input_flatten=self.with_pos_embed(pts_feats, pos1),
                                             input_spatial_shapes=torch.tensor([(self.bev_size, self.bev_size)]).to(
                                                 device),
                                             input_level_start_index=torch.tensor(
                                                 [0, self.bev_size * self.bev_size]).to(device),
                                             input_padding_mask=None)
                fusion_f1 = rearrange(fusion_f1, 'b (h w) c -> b c h w', h=self.bev_size, w=self.bev_size)
                fusion_f2 = rearrange(fusion_f2, 'b (h w) c -> b c h w', h=self.bev_size, w=self.bev_size)
                fusion_f = self.RadarConvFuser_fuse(fusion_f1, fusion_f2)
                """
                :param query                       (N, Length_{query}, C)
                :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                                or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
                :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
                :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
                :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
                :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

                :return output                     (N, Length_{query}, C)
                """
            fusion_feats.append(fusion_f)  # cat

        return fusion_feats, pts_feats, depth

    def forward_aux_centerness_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        outs = self.aux_centerness_head(pts_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.aux_centerness_head.loss(*loss_inputs)
        return losses

    def simple_test_aux_centerness(self, x, img_metas, rescale=False, **kwargs):
        """Test function of point cloud branch."""
        # outs = self.aux_centerness_head(x)
        tx = self.aux_centerness_head.shared_conv(x[0])     # (B, C'=share_conv_channel, H, W)
        outs_inst_center_reg = self.aux_centerness_head.task_heads[0].reg(tx)
        outs_inst_center_height = self.aux_centerness_head.task_heads[0].height(tx)
        outs_inst_center_heatmap = self.aux_centerness_head.task_heads[0].heatmap(tx)
        outs = ([{
            "reg" : outs_inst_center_reg,
            "height" : outs_inst_center_height,
            "heatmap" : outs_inst_center_heatmap,
        }],)

        # # bbox_list = self.aux_centerness_head.get_bboxes(
        # #     outs, img_metas, rescale=rescale)
        # # bbox_results = [
        # #     bbox3d2result(bboxes, scores, labels)
        # #     for bboxes, scores, labels in bbox_list
        # # ]
        ins_cen_list = self.aux_centerness_head.get_centers(
            outs, img_metas, rescale=rescale)
        # return bbox_results, ins_cen_list
        return None, ins_cen_list

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        fuse_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        gt_depth = kwargs['gt_depth']   # (B, N_views, img_H, img_W)
        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth

        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)
        occ_bev_feature = fuse_feats[0]
        loss_occ = self.forward_occ_train(occ_bev_feature, voxel_semantics, mask_camera)
        losses.update(loss_occ)

        if self.aux_centerness_head is not None:
            losses_aux_centerness = self.forward_aux_centerness_train([occ_bev_feature], gt_bboxes_3d,
                                                                      gt_labels_3d, img_metas,
                                                                      gt_bboxes_ignore)
            losses.update(losses_aux_centerness)

        return losses

    def forward_test(self,
                     points=None,
                     img_inputs=None,
                     img_metas=None,
                     **kwargs):
        for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)

        fuse_feats, pts_feats, depth = self.extract_feat(
                points, img_inputs=img, img_metas=img_metas, **kwargs)

        occ_bev_feature = fuse_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        occ_list = self.simple_test_occ(occ_bev_feature, img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list

    def simple_test_occ(self, img_feats, img_metas=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        outs = self.occ_head(img_feats)
        if not hasattr(self.occ_head, "get_occ_gpu"):
            occ_preds = self.occ_head.get_occ(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        else:
            occ_preds = self.occ_head.get_occ_gpu(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_preds

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        outs = self.occ_head(occ_bev_feature)
        return outs

@DETECTORS.register_module()
class CLRFusion(RCFUSION):
    def __init__(self,
                 radar_pts_voxel_layer=None,
                 radar_pts_voxel_encoder=None,
                 radar_pts_middle_encoder=None,
                 radar_pts_backbone=None,
                 radar_pts_neck=None,
                 imc=None,
                 rac=None,
                 **kwargs):

        super().__init__(**kwargs)
        if radar_pts_voxel_layer:
            self.radar_pts_voxel_layer = Voxelization(**radar_pts_voxel_layer)
        if radar_pts_voxel_encoder:
            self.radar_pts_voxel_encoder = builder.build_voxel_encoder(
                radar_pts_voxel_encoder)
        if radar_pts_middle_encoder:
            self.radar_pts_middle_encoder = builder.build_middle_encoder(
                radar_pts_middle_encoder)
        if radar_pts_backbone:
            self.radar_pts_backbone = builder.build_backbone(
                radar_pts_backbone)
        if radar_pts_neck:
            self.radar_pts_neck = builder.build_neck(
                radar_pts_neck)



    def extract_img_feat(self, img_inputs, img_metas, **kwargs):
        """ Extract features of images.
        img_inputs:
            imgs:  (B, N_views, 3, H, W)
            sensor2egos: (B, N_views, 4, 4)
            ego2globals: (B, N_views, 4, 4)
            intrins:     (B, N_views, 3, 3)
            post_rots:   (B, N_views, 3, 3)
            post_trans:  (B, N_views, 3)
            bda_rot:  (B, 3, 3)
        Returns:
            x: [(B, C', H', W'), ]
            depth: (B*N, D, fH, fW)
        """
        img_inputs = self.prepare_inputs(img_inputs)
        x, _ = self.image_encoder(img_inputs[0])    # x: (B, N, C, fH, fW)
        x, depth = self.img_view_transformer([x] + img_inputs[1:7])

        return [x], depth

    def extract_pts_feat(self, pts, img_feats, img_metas,):
        """Extract features of points."""
        # if not self.with_pts_bbox:
        #     return None
        # print(f"init:{pts}")
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors
                                                )
        batch_size = coors[-1, 0] + 1

        # for sparse_encoder
        x = self.pts_middle_encoder(voxel_features, coors, batch_size,return_bev_feat=True)
        # print(f"after_middler:{x}")

        return x

    def extract_radar_feat(self, pts, img_feats, img_metas,):
        """Extract features of radar_points."""
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.radar_pts_voxel_encoder(voxels, num_points, coors
                                                )
        batch_size = coors[-1, 0] + 1

        x = self.radar_pts_middle_encoder(voxel_features, coors, batch_size,return_bev_feat=True)

        if self.radar_pts_backbone is not None:
            x = self.radar_pts_backbone(x)

        if self.radar_pts_neck is not None:
            x = self.radar_pts_neck(x)

        return x
    def extract_feat(self, points, radar_points, img_inputs, img_metas, **kwargs):
        img_feats, depth = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        radar_feats = self.extract_radar_feat( radar_points, img_feats, img_metas)

        if type(pts_feats) is not list:
            pts_feats = [pts_feats]

        if type(radar_feats) is not list:
            radar_feats = [radar_feats]

        fusion_feats = []
        bev_height = img_feats[0].shape[2]
        bev_width = img_feats[0].shape[3]

        for i in range(0, len(img_feats)):

            if hasattr(self, 'DeformAttn1') and self.DeformAttn1 is not None:
                pts_feats[i] = self.radar_reduc_conv(pts_feats[i])  # pts_feats 由 rac --> imc
                radar_feats = rearrange(radar_feats[i], 'b c h w -> b (h w) c')
                pts_feats = rearrange(pts_feats[i], 'b c h w -> b (h w) c')
                # img_feats = rearrange(img_feats[i], 'b c h w -> b (h w) c')

                device = torch.device("cuda")  # Get the CUDA device
                mask = torch.zeros(1, 1, self.bev_size, self.bev_size).to(device)
                pos1 = self.LearnedPositionalEncoding1(mask)
                pos2 = self.LearnedPositionalEncoding2(mask)

                reference_point1 = self.get_reference_points(self.bev_size, self.bev_size)
                reference_point2 = self.get_reference_points(self.bev_size, self.bev_size)

                fusion_f1 = self.DeformAttn1(query=self.with_pos_embed(radar_feats, pos1),
                                             reference_points=reference_point1,
                                             input_flatten=self.with_pos_embed(pts_feats, pos2),
                                             input_spatial_shapes=torch.tensor([(self.bev_size, self.bev_size)]).to(
                                                 device),
                                             input_level_start_index=torch.tensor(
                                                 [0, self.bev_size * self.bev_size]).to(device),
                                             input_padding_mask=None)
                fusion_f2 = self.DeformAttn2(query=self.with_pos_embed(pts_feats, pos2),
                                             reference_points=reference_point2,
                                             input_flatten=self.with_pos_embed(radar_feats, pos1),
                                             input_spatial_shapes=torch.tensor([(self.bev_size, self.bev_size)]).to(
                                                 device),
                                             input_level_start_index=torch.tensor(
                                                 [0, self.bev_size * self.bev_size]).to(device),
                                             input_padding_mask=None)
                fusion_f1 = rearrange(fusion_f1, 'b (h w) c -> b c h w', h=self.bev_size, w=self.bev_size)
                fusion_f2 = rearrange(fusion_f2, 'b (h w) c -> b c h w', h=self.bev_size, w=self.bev_size)
                fusion_f = self.RadarConvFuser_fuse(fusion_f1, fusion_f2)
                """
                :param query                       (N, Length_{query}, C)
                :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                                or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
                :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
                :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
                :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
                :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

                :return output                     (N, Length_{query}, C)
                """
            fusion_feats.append(fusion_f)  # cat

        final_feats=[]
        for i in range(0, len(img_feats)):

            if hasattr(self, 'DeformAttn3') and self.DeformAttn3 is not None:
                img_feats[i] = self.radar_reduc_conv(img_feats[i])  # pts_feats 由 rac --> imc
                fusion_feats = rearrange(fusion_feats[i], 'b c h w -> b (h w) c')
                # pts_feats = rearrange(pts_feats[i], 'b c h w -> b (h w) c')
                img_feats = rearrange(img_feats[i], 'b c h w -> b (h w) c')

                device = torch.device("cuda")  # Get the CUDA device
                mask = torch.zeros(1, 1, self.bev_size, self.bev_size).to(device)
                pos3 = self.LearnedPositionalEncoding3(mask)
                pos4 = self.LearnedPositionalEncoding4(mask)

                reference_point3 = self.get_reference_points(self.bev_size, self.bev_size)
                reference_point4 = self.get_reference_points(self.bev_size, self.bev_size)

                fusion_f3 = self.DeformAttn3(query=self.with_pos_embed(fusion_feats, pos3),
                                             reference_points=reference_point3,
                                             input_flatten=self.with_pos_embed(img_feats, pos4),
                                             input_spatial_shapes=torch.tensor([(self.bev_size, self.bev_size)]).to(
                                                 device),
                                             input_level_start_index=torch.tensor(
                                                 [0, self.bev_size * self.bev_size]).to(device),
                                             input_padding_mask=None)
                fusion_f4 = self.DeformAttn4(query=self.with_pos_embed(img_feats, pos4),
                                             reference_points=reference_point4,
                                             input_flatten=self.with_pos_embed(fusion_feats, pos3),
                                             input_spatial_shapes=torch.tensor([(self.bev_size, self.bev_size)]).to(
                                                 device),
                                             input_level_start_index=torch.tensor(
                                                 [0, self.bev_size * self.bev_size]).to(device),
                                             input_padding_mask=None)
                fusion_f3 = rearrange(fusion_f3, 'b (h w) c -> b c h w', h=self.bev_size, w=self.bev_size)
                fusion_f4 = rearrange(fusion_f4, 'b (h w) c -> b c h w', h=self.bev_size, w=self.bev_size)
                fusion_f = self.RadarConvFuser_fuse(fusion_f3, fusion_f4)
                """
                :param query                       (N, Length_{query}, C)
                :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                                or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
                :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
                :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
                :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
                :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

                :return output                     (N, Length_{query}, C)
                """
            final_feats.append(fusion_f)  # cat


        return final_feats, pts_feats, depth

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      radar=None,
                      **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)

        fuse_feats, pts_feats, depth = self.extract_feat(
                points,radar_points=radar, img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        losses = dict()
        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)

        if type(fuse_feats) == list:
            occ_bev_feature = fuse_feats[0]
        occ_bev_feature = self.bev_encoder(occ_bev_feature)

        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        loss_occ = self.forward_occ_train(occ_bev_feature, voxel_semantics, mask_camera)
        losses.update(loss_occ)

        if self.aux_centerness_head is not None:
            losses_aux_centerness = self.forward_aux_centerness_train([occ_bev_feature], gt_bboxes_3d,
                                                                      gt_labels_3d, img_metas,
                                                                      gt_bboxes_ignore)
            losses.update(losses_aux_centerness)

        return losses

    def forward_test(self,
                     points=None,
                     img_inputs=None,
                     img_metas=None,
                     radar=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            radar = [radar] if radar is None else radar
            return self.simple_test(points[0], img_metas[0], img_inputs[0],radar=radar[0],
                                    **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    radar=None,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)

        fuse_feats, pts_feats, depth = self.extract_feat(
                points,radar_points=radar, img_inputs=img, img_metas=img_metas, **kwargs)

        if type(fuse_feats) == list:
            occ_bev_feature = fuse_feats[0]
        occ_bev_feature = self.bev_encoder(occ_bev_feature)

        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        occ_list = self.simple_test_occ(occ_bev_feature, img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list