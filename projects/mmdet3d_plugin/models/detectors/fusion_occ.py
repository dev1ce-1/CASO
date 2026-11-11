# Copyright (c) Phigent Robotics. All rights reserved.
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32

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

from mmcv.cnn import bias_init_with_prob

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
    0: 4,
    1: 10,
    2: 9,
    3: 3,
    4: 5,
    5: 2,
    6: 6,
    7: 7,
    8: 8,
    9: 1,
}
occind2detind = {
    4: 0,
    10: 1,
    9: 2,
    3: 3,
    5: 4,
    2: 5,
    6: 6,
    7: 7,
    8: 8,
    1: 9,
}
occind2detind_cuda = [-1, -1, 5, 3, 0, 4, 6, 7, -1, 2, 1]

inst_occ = np.ones([200, 200, 16]) * 0

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
class FUSIONOCCBase(CenterPoint):
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
                 radars_voxel_encoder=None,
                 radars_middle_encoder=None,
                 occ_head=None,
                 upsample=False,
                 train_cfg=None,
                 test_cfg=None,
                 use_transfusion=False,
                 **kwargs):
        super(FUSIONOCCBase, self).__init__(**kwargs)
        if pts_voxel_layer:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        if pts_voxel_encoder:
            self.pts_voxel_encoder = builder.build_voxel_encoder(
                pts_voxel_encoder)
        if pts_middle_encoder:
            self.pts_middle_encoder = builder.build_middle_encoder(
                pts_middle_encoder)
        if pts_backbone:
            self.pts_backbone = builder.build_backbone(pts_backbone)
        if pts_neck is not None:
            self.pts_neck = builder.build_neck(pts_neck)
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

        if img_bev_encoder_backbone:
            self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone)

        if img_bev_encoder_neck:
            self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)

        if radars_voxel_encoder:
            self.radars_voxel_encoder = builder.build_voxel_encoder(
                radars_voxel_encoder)
        if radars_middle_encoder:
            self.radars_middle_encoder = builder.build_middle_encoder(
                radars_middle_encoder)

        self.fuse = builder.build_fusion_layer(fuse) if fuse is not None else None

        if img_view_transformer:
            self.img_view_transformer = builder.build_neck(img_view_transformer)
        # self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone)
        # self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)

        self.occ_head = builder.build_head(occ_head)

        # self.train_cfg = train_cfg
        # self.test_cfg = test_cfg
        self.upsample = upsample

        self.use_transfusion = use_transfusion

        if self.pts_neck is not None:
            mid_channels = sum(self.pts_neck.out_channels)

        self.return_context = True
        self.return_occupancy = True

        if self.return_context:
            if 'out_channels_pts' in kwargs:
                out_channels = kwargs['out_channels_pts']
            else:
                out_channels = 192
            self.pred_context = nn.Sequential(
                nn.Conv2d(mid_channels,
                          mid_channels // 2,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          padding_mode='reflect'),
                nn.BatchNorm2d(mid_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels // 2,
                          out_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0),
            )

        if self.return_occupancy:
            self.pred_occupancy = nn.Sequential(
                nn.Conv2d(mid_channels,
                          mid_channels // 2,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          padding_mode='reflect'),
                nn.BatchNorm2d(mid_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels // 2,
                          1,
                          kernel_size=1,
                          stride=1,
                          padding=0),
            )

            if 'occupancy_init' in kwargs:
                occupancy_init = kwargs['occupancy_init']
            else:
                occupancy_init = 0.01
            self.pred_occupancy[-1].bias.data.fill_(bias_init_with_prob(occupancy_init))

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

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        # if not self.with_pts_bbox:
        #     return None
        # print(f"init:{pts}")
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                img_feats, img_metas)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        # print(f"after_middler:{x}")
        x = self.pts_backbone(x)
        # print(f"after_backbone:{x}")
        if self.with_pts_neck:
            x = self.pts_neck(x)
        # print(f"after_neck:{x}")
        return x

    def extract_radar_feat(self, radars):
        """Extract features of points."""
        # if not self.with_pts_bbox:
        #     return None
        # print(f"init:{pts}")
        B, N, P, F = radars.shape
        batch_size = B * N
        radars = radars.contiguous().view(B * N, P, F)
        voxels, num_points, coors = self.voxelize(radars)
        voxel_features = self.radars_voxel_encoder(voxels, num_points, coors)
        # batch_size = coors[-1, 0] + 1
        x = self.radars_middle_encoder(voxel_features, coors, batch_size)
        # print(f"after_middler:{x}")
        x = self.pts_backbone(x)
        # print(f"after_backbone:{x}")
        if self.with_pts_neck:
            x = self.pts_neck(x)
        # print(f"after_neck:{x}")

        x_context = None
        x_occupancy = None
        if self.return_context:
            x_context = self.pred_context(x[-1]).unsqueeze(1)
        if self.return_occupancy:
            x_occupancy = self.pred_occupancy(x[-1]).unsqueeze(1).sigmoid()

        return x_context, x_occupancy

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
        keyego2global = ego2globals[:, 0, ...].unsqueeze(1)  # (B, 1, 4, 4)
        global2keyego = torch.inverse(keyego2global.double())  # (B, 1, 4, 4)
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()  # (B, N_views, 4, 4)
        sensor2keyegos = sensor2keyegos.float()

        return [imgs, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda]

    def extract_img_feat(self, img_inputs, img_metas, radar_context=None, radar_occupancy=None):
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
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda = self.prepare_inputs(img_inputs)
        x, _ = self.image_encoder(imgs)  # x: (B, N, C, fH, fW)

        mlp_input = self.img_view_transformer.get_mlp_input(
            sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda)  # (B, N_views, 27)

        x, depth = self.img_view_transformer([x, sensor2keyegos, ego2globals, intrins, post_rots,
                                              post_trans, bda, mlp_input],
                                             radar_context=radar_context,
                                             radar_occupancy=radar_occupancy)
        # print(f"x.shape: {x.shape}") # 4,64,200,200 for flashocc

        # x: (B, C, Dy, Dx) for flashocc   /    (B,C,Dz,Dy,Dx) for bevdet
        # depth: (B*N, D, fH, fW)

        return [x], depth

    def extract_feat(self, points, img_inputs, img_metas, radar=None):
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
        radar_context = None
        radar_occupancy = None
        if radar is not None:
            radar_context, radar_occupancy = self.extract_radar_feat(radar)
        img_feats, depth = self.extract_img_feat(img_inputs, img_metas,
                                                 radar_context=radar_context,
                                                 radar_occupancy=radar_occupancy)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)

        return img_feats, pts_feats, radar_context, depth

    # for transfusion
    def custom_extract_feat(self, points, img_inputs, img_metas, **kwargs):
        img_inputs = self.prepare_inputs(img_inputs)
        x, _ = self.image_encoder(img_inputs[0])  # x: (B, N, C, fH, fW)
        img_feats = x
        B, N, C, fH, fW = img_feats.shape
        img_feats = img_feats.view(B * N, C, fH, fW)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        return img_feats, pts_feats

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

        # 判断是否使用radar
        radar = kwargs['radar'] if 'radar' in kwargs else None

        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        if not self.use_transfusion:
            img_feats, pts_feats, radar_feats, depth = self.extract_feat(
                points, img_inputs=img_inputs, img_metas=img_metas, radar=radar)

            fuse_feats = []
            if img_feats is not None:
                fuse_feats.append(img_feats)
            if pts_feats is not None:
                fuse_feats.append(pts_feats)
            if radar_feats is not None:
                fuse_feats.append(radar_feats)

            if len(fuse_feats) == 1:
                fuse_feats = fuse_feats[0]
            else:
                fuse_feats = self.fuse(fuse_feats)

        else:
            img_feats, pts_feats = self.custom_extract_feat(
                points, img_inputs=img_inputs, img_metas=img_metas, **kwargs
            )
            fuse_feats = self.fuse(pts_feats, img_feats, img_metas)

        fuse_feats = self.bev_encoder(fuse_feats)

        losses = dict()
        voxel_semantics = kwargs['voxel_semantics']  # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']  # (B, Dx, Dy, Dz)

        occ_bev_feature = fuse_feats
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        loss_occ = self.forward_occ_train(occ_bev_feature, voxel_semantics, mask_camera)
        losses.update(loss_occ)
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

        occ_list = self.simple_test_occ(occ_bev_feature, img_metas)  # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
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
            occ_preds = self.occ_head.get_occ(outs, img_metas)  # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        else:
            occ_preds = self.occ_head.get_occ_gpu(outs, img_metas)  # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
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
class RVT(FUSIONOCCBase):
    def __init__(self, **kwargs):
        super(RVT, self).__init__(**kwargs)

    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        batch_size, _, _ = points.shape
        points_list = [points[i] for i in range(batch_size)]

        for res in points_list:
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

    def extract_img_feat(self, img_inputs, img_metas, radar_context=None, radar_occupancy=None):
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
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda = self.prepare_inputs(img_inputs)
        x, _ = self.image_encoder(imgs)  # x: (B, N, C, fH, fW)

        mlp_input = self.img_view_transformer.get_mlp_input(
            sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda)  # (B, N_views, 27)

        x, depth = self.img_view_transformer([x, sensor2keyegos, ego2globals, intrins, post_rots,
                                              post_trans, bda, mlp_input],
                                             radar_context=radar_context[:,0,...],
                                             radar_occupancy=radar_occupancy[:,0,...])
        # print(f"x.shape: {x.shape}") # 4,64,200,200 for flashocc

        # x: (B, C, Dy, Dx) for flashocc   /    (B,C,Dz,Dy,Dx) for bevdet
        # depth: (B*N, D, fH, fW)

        return [x], depth

    def extract_radar_feat(self, radars):
        """Extract features of points."""
        # if not self.with_pts_bbox:
        #     return None
        # print(f"init:{pts}")
        B, N, P, F = radars.shape
        batch_size = B * N
        radars = radars.contiguous().view(B * N, P, F)
        voxels, num_points, coors = self.voxelize(radars)
        voxel_features = self.radars_voxel_encoder(voxels, num_points, coors)
        # batch_size = coors[-1, 0] + 1
        x = self.radars_middle_encoder(voxel_features, coors, batch_size)
        # print(f"after_middler:{x}")
        x = self.pts_backbone(x)
        # print(f"after_backbone:{x}")
        if self.with_pts_neck:
            x = self.pts_neck(x)
        # print(f"after_neck:{x}")
        # 6,192,88,44

        x_context = None
        x_occupancy = None
        if self.return_context:
            x_context = self.pred_context(x[-1]).unsqueeze(1)
        if self.return_occupancy:
            x_occupancy = self.pred_occupancy(x[-1]).unsqueeze(1).sigmoid()

        return x_context, x_occupancy
    def extract_feat(self, points, img_inputs, img_metas, radar=None):
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
        radar_context = None
        radar_occupancy = None
        if radar is not None:
            radar_context, radar_occupancy = self.extract_radar_feat(radar[:,0,...])
        img_feats, depth = self.extract_img_feat(img_inputs, img_metas,
                                                 radar_context=radar_context,
                                                 radar_occupancy=radar_occupancy)
        pts_feats = None

        return img_feats, pts_feats, radar_context, depth
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

        # 判断是否使用radar
        # radar = kwargs['radar'] if 'radar' in kwargs else None
        radar_pv = kwargs['radar_pv'] if 'radar_pv' in kwargs else None

        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)

        img_feats, pts_feats, radar_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, radar=radar_pv)

        fuse_feats = self.bev_encoder(img_feats[0]) if self.fuse is None else self.fuse(img_feats)

        losses = dict()
        gt_depth = kwargs['gt_depth']   # (B, N_views, img_H, img_W)
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth

        voxel_semantics = kwargs['voxel_semantics']  # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']  # (B, Dx, Dy, Dz)

        occ_bev_feature = fuse_feats
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        loss_occ = self.forward_occ_train(occ_bev_feature, voxel_semantics, mask_camera)
        losses.update(loss_occ)
        return losses

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
            # 判断是否使用radar
            # radar = kwargs['radar'] if 'radar' in kwargs else None
            radar_pv = kwargs['radar_pv'] if 'radar_pv' in kwargs else None
            return self.simple_test(points[0], img_metas[0], img_inputs[0],radar_pv=radar_pv[0])
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    radar_pv=None,
                    rescale=False,
                    ):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)



        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)

        img_feats, pts_feats, radar_feats, depth = self.extract_feat(
            points, img_inputs=img, img_metas=img_metas, radar=radar_pv)

        fuse_feats = img_feats[0] if self.fuse is None else self.fuse(img_feats)


        occ_bev_feature = fuse_feats
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        occ_list = self.simple_test_occ(occ_bev_feature, img_metas)  # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list
