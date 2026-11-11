# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
import numpy as np
from mmdet3d.models.builder import HEADS, build_loss
from ..losses.semkitti_loss import sem_scal_loss, geo_scal_loss
from ..losses.lovasz_softmax import lovasz_softmax


nusc_class_frequencies = np.array([
    944004,
    1897170,
    152386,
    2391677,
    16957802,
    724139,
    189027,
    2074468,
    413451,
    2384460,
    5916653,
    175883646,
    4275424,
    51393615,
    61411620,
    105975596,
    116424404,
    1892500630
])


@HEADS.register_module()
class BEVOCCHead3D(BaseModule):
    def __init__(self,
                 in_dim=32,
                 out_dim=32,
                 use_mask=True,
                 num_classes=18,
                 use_predicter=True,
                 class_balance=False,
                 loss_occ=None
                 ):
        super(BEVOCCHead3D, self).__init__()
        self.out_dim = 32
        out_channels = out_dim if use_predicter else num_classes
        self.final_conv = ConvModule(
            in_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv3d')
        )
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, num_classes),
            )

        self.num_classes = num_classes
        self.use_mask = use_mask
        self.class_balance = class_balance
        if self.class_balance:
            class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_classes] + 0.001))
            self.cls_weights = class_weights
            loss_occ['class_weight'] = class_weights

        self.loss_occ = build_loss(loss_occ)

    def forward(self, img_feats):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx)

        Returns:

        """
        # (B, C, Dz, Dy, Dx) --> (B, C, Dz, Dy, Dx) --> (B, Dx, Dy, Dz, C)
        occ_pred = self.final_conv(img_feats).permute(0, 4, 3, 2, 1)
        if self.use_predicter:
            # (B, Dx, Dy, Dz, C) --> (B, Dx, Dy, Dz, 2*C) --> (B, Dx, Dy, Dz, n_cls)
            occ_pred = self.predicter(occ_pred)

        return occ_pred

    def loss(self, occ_pred, voxel_semantics, mask_camera):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:

        """
        loss = dict()
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)   # (B, Dx, Dy, Dz)
            # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
            voxel_semantics = voxel_semantics.reshape(-1)
            # (B, Dx, Dy, Dz, n_cls) --> (B*Dx*Dy*Dz, n_cls)
            preds = occ_pred.reshape(-1, self.num_classes)
            # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
            mask_camera = mask_camera.reshape(-1)

            if self.class_balance:
                valid_voxels = voxel_semantics[mask_camera.bool()]
                num_total_samples = 0
                for i in range(self.num_classes):
                    num_total_samples += (valid_voxels == i).sum() * self.cls_weights[i]
            else:
                num_total_samples = mask_camera.sum()

            loss_occ = self.loss_occ(
                preds,      # (B*Dx*Dy*Dz, n_cls)
                voxel_semantics,    # (B*Dx*Dy*Dz, )
                mask_camera,        # (B*Dx*Dy*Dz, )
                avg_factor=num_total_samples
            )
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = occ_pred.reshape(-1, self.num_classes)

            if self.class_balance:
                num_total_samples = 0
                for i in range(self.num_classes):
                    num_total_samples += (voxel_semantics == i).sum() * self.cls_weights[i]
            else:
                num_total_samples = len(voxel_semantics)

            loss_occ = self.loss_occ(
                preds,
                voxel_semantics,
                avg_factor=num_total_samples
            )

        loss['loss_occ'] = loss_occ
        return loss

    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)     # (B, Dx, Dy, Dz)
        return list(occ_res)


@HEADS.register_module()
class BEVOCCHead2D(BaseModule):
    def __init__(self,
                 in_dim=256,
                 out_dim=256,
                 Dz=16,
                 use_mask=True,
                 num_classes=18,
                 use_predicter=True,
                 class_balance=False,
                 loss_occ=None,
                 **kwargs
                 ):
        super(BEVOCCHead2D, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz
        out_channels = out_dim if use_predicter else num_classes * Dz
        self.final_conv = ConvModule(
            self.in_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d')
        )
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, num_classes * Dz),
            )

        self.use_mask = use_mask
        self.num_classes = num_classes

        self.class_balance = class_balance
        if self.class_balance:
            class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_classes] + 0.001))
            self.cls_weights = class_weights
            loss_occ['class_weight'] = class_weights        # ce loss
        self.loss_occ = build_loss(loss_occ)

    def forward(self, img_feats):
        """
        Args:
            img_feats: (B, C, Dy, Dx)

        Returns:

        """
        # (B, C, Dy, Dx) --> (B, C, Dy, Dx) --> (B, Dx, Dy, C)
        occ_pred = self.final_conv(img_feats).permute(0, 3, 2, 1)
        bs, Dx, Dy = occ_pred.shape[:3]
        if self.use_predicter:
            # (B, Dx, Dy, C) --> (B, Dx, Dy, 2*C) --> (B, Dx, Dy, Dz*n_cls)
            occ_pred = self.predicter(occ_pred)
            occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_classes)

        return occ_pred

    def loss(self, occ_pred, voxel_semantics, mask_camera,mask_lidar=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:

        """
        loss = dict()
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            # mask_lidar = mask_lidar.to(torch.int32)   # (B, Dx, Dy, Dz)
            mask_camera = mask_camera.to(torch.int32)   # (B, Dx, Dy, Dz)
            # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
            voxel_semantics = voxel_semantics.reshape(-1)
            # (B, Dx, Dy, Dz, n_cls) --> (B*Dx*Dy*Dz, n_cls)
            preds = occ_pred.reshape(-1, self.num_classes)
            # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
            # mask_lidar = mask_lidar.reshape(-1)
            mask_camera = mask_camera.reshape(-1)

            if self.class_balance:
                valid_voxels = voxel_semantics[mask_camera.bool()]
                # valid_voxels = voxel_semantics[mask_lidar.bool()]
                num_total_samples = 0
                for i in range(self.num_classes):
                    num_total_samples += (valid_voxels == i).sum() * self.cls_weights[i]
            else:
                # num_total_samples = mask_lidar.sum()
                num_total_samples = mask_camera.sum()

            loss_occ = self.loss_occ(
                preds,      # (B*Dx*Dy*Dz, n_cls)
                voxel_semantics,    # (B*Dx*Dy*Dz, )
                # mask_lidar,        # (B*Dx*Dy*Dz, )
                mask_camera,        # (B*Dx*Dy*Dz, )
                avg_factor=num_total_samples
            )
            loss['loss_occ'] = loss_occ
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = occ_pred.reshape(-1, self.num_classes)

            if self.class_balance:
                num_total_samples = 0
                for i in range(self.num_classes):
                    num_total_samples += (voxel_semantics == i).sum() * self.cls_weights[i]
            else:
                num_total_samples = len(voxel_semantics)

            loss_occ = self.loss_occ(
                preds,
                voxel_semantics,
                avg_factor=num_total_samples
            )

            loss['loss_occ'] = loss_occ
        return loss

    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)     # (B, Dx, Dy, Dz)
        return list(occ_res)

    def get_occ_gpu(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1).int()      # (B, Dx, Dy, Dz)
        return list(occ_res)


@HEADS.register_module()
class BEVOCCHead2D_RiskOcc(BEVOCCHead2D):
    """BEVOCCHead2D with the RiskOcc auxiliary training losses."""

    OCC3D_CLASS_NAMES = [
        'others', 'barrier', 'bicycle', 'bus', 'car',
        'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
        'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
        'terrain', 'manmade', 'vegetation', 'free'
    ]

    def __init__(self,
                 risk_alpha=0.5,
                 risk_beta=0.25,
                 risk_gamma=1.0,
                 risk_lambda_distance=0.5,
                 risk_lambda_class=0.5,
                 risk_free_id=17,
                 risk_near_distance=20.0,
                 risk_distance_max=40.0,
                 risk_pc_range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
                 risk_voxel_size=0.4,
                 risk_mode='field',
                 **kwargs):
        super().__init__(**kwargs)
        if risk_mode not in ('field', 'uniform'):
            raise ValueError(f"risk_mode must be 'field' or 'uniform', "
                             f"got {risk_mode!r}")
        self.risk_mode = risk_mode
        self.risk_alpha = float(risk_alpha)
        self.risk_beta = float(risk_beta)
        self.risk_gamma = float(risk_gamma)
        self.risk_lambda_distance = float(risk_lambda_distance)
        self.risk_lambda_class = float(risk_lambda_class)
        self.risk_free_id = int(risk_free_id)
        self.risk_near_distance = float(risk_near_distance)
        self.risk_distance_max = float(risk_distance_max)
        self.risk_pc_range = tuple(float(v) for v in risk_pc_range)
        self.risk_voxel_size = float(risk_voxel_size)

        rc_values = []
        vulnerable = {'pedestrian', 'bicycle', 'motorcycle'}
        dynamic = {
            'car', 'truck', 'bus', 'trailer', 'construction_vehicle'
        }
        for name in self.OCC3D_CLASS_NAMES[:self.num_classes]:
            if name in vulnerable:
                rc_values.append(1.0)
            elif name in dynamic:
                rc_values.append(0.6)
            elif name == 'free':
                rc_values.append(0.0)
            else:
                rc_values.append(0.3)
        self.register_buffer(
            'risk_class_weights',
            torch.tensor(rc_values, dtype=torch.float32),
            persistent=False)

        self._risk_grid_shape = None

    def _risk_distance_fields(self, occ_pred):
        grid_shape = tuple(int(v) for v in occ_pred.shape[1:4])
        device = occ_pred.device
        dtype = occ_pred.dtype
        cached_shape = self._risk_grid_shape
        if (cached_shape == grid_shape
                and hasattr(self, 'risk_distance')
                and self.risk_distance.device == device
                and self.risk_distance.dtype == dtype):
            return self.risk_distance, self.risk_near_mask

        x_count, y_count, z_count = grid_shape
        pr = self.risk_pc_range
        vs = self.risk_voxel_size
        xs = (torch.arange(x_count, dtype=dtype, device=device) + 0.5) \
            * vs + pr[0]
        ys = (torch.arange(y_count, dtype=dtype, device=device) + 0.5) \
            * vs + pr[1]
        x_grid, y_grid = torch.meshgrid(xs, ys, indexing='ij')
        bev_dist = torch.sqrt(x_grid.square() + y_grid.square())
        rd = torch.clamp((self.risk_distance_max - bev_dist)
                         / self.risk_distance_max, min=0.0, max=1.0)
        rd_3d = rd[:, :, None].expand(x_count, y_count, z_count)
        near_3d = (bev_dist <= self.risk_near_distance)[:, :, None].expand(
            x_count, y_count, z_count)
        self.risk_distance = rd_3d
        self.risk_near_mask = near_3d
        self._risk_grid_shape = grid_shape
        return self.risk_distance, self.risk_near_mask

    def loss(self, occ_pred, voxel_semantics, mask_camera, mask_lidar=None):
        loss = super().loss(occ_pred, voxel_semantics, mask_camera,
                            mask_lidar=mask_lidar)

        y = voxel_semantics.long()
        label_valid = (y >= 0) & (y < self.num_classes)
        if self.use_mask:
            valid = mask_camera.bool() & label_valid
        else:
            valid = label_valid

        occ_mask = valid & (y != self.risk_free_id)
        free_mask = valid & (y == self.risk_free_id)
        rd, near_mask = self._risk_distance_fields(occ_pred)

        if self.risk_mode == 'uniform':
            # CTRL-U control: R(v) == 1 on every voxel (uniform FN up-weight,
            # no spatial/class shape). Everything else identical to 'field'.
            risk = torch.ones_like(y, dtype=occ_pred.dtype)
        else:
            y_safe = y.clamp(min=0, max=self.num_classes - 1)
            rc = self.risk_class_weights.to(
                device=occ_pred.device, dtype=occ_pred.dtype)[y_safe]
            risk = self.risk_lambda_distance * rd + self.risk_lambda_class * rc

        prob = torch.softmax(occ_pred, dim=-1)
        p_free = prob[..., self.risk_free_id]
        p_occ = 1.0 - p_free

        denom_occ = occ_mask.sum().clamp(min=1).to(dtype=occ_pred.dtype)
        denom_free = (near_mask & free_mask).sum().clamp(min=1).to(
            dtype=occ_pred.dtype)

        loss_fn = (risk * occ_mask.to(dtype=occ_pred.dtype)
                   * p_free.pow(self.risk_gamma)).sum() / denom_occ
        loss_fp = ((near_mask & free_mask).to(dtype=occ_pred.dtype)
                   * p_occ.square()).sum() / denom_free

        loss['loss_risk_fn'] = loss_fn * self.risk_alpha
        loss['loss_risk_fp'] = loss_fp * self.risk_beta
        return loss


@HEADS.register_module()
class BEVOCCHead2D_V2(BaseModule):      # Use stronger loss setting
    def __init__(self,
                 in_dim=256,
                 out_dim=256,
                 Dz=16,
                 use_mask=True,
                 num_classes=18,
                 use_predicter=True,
                 class_balance=False,
                 loss_occ=None,
                 ):
        super(BEVOCCHead2D_V2, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz

        # voxel-level prediction
        self.occ_convs = nn.ModuleList()
        self.final_conv = ConvModule(
            in_dim,
            self.out_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d')
        )
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, num_classes * Dz),
            )

        self.use_mask = use_mask
        self.num_classes = num_classes

        self.class_balance = class_balance
        if self.class_balance:
            class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_classes] + 0.001))
            self.cls_weights = class_weights
        self.loss_occ = build_loss(loss_occ)

    def forward(self, img_feats):
        """
        Args:
            img_feats: (B, C, Dy=200, Dx=200)
            img_feats: [(B, C, 100, 100), (B, C, 50, 50), (B, C, 25, 25)]   if ms
        Returns:

        """
        # (B, C, Dy, Dx) --> (B, C, Dy, Dx) --> (B, Dx, Dy, C)
        occ_pred = self.final_conv(img_feats).permute(0, 3, 2, 1)
        bs, Dx, Dy = occ_pred.shape[:3]
        if self.use_predicter:
            # (B, Dx, Dy, C) --> (B, Dx, Dy, 2*C) --> (B, Dx, Dy, Dz*n_cls)
            occ_pred = self.predicter(occ_pred)
            occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_classes)

        return occ_pred

    def loss(self, occ_pred, voxel_semantics, mask_camera):
        """
        v2较于v1，在loss部分去除了对于usemask等的判断，即没有使用mask
        增加了loss_voxel_sem_scal，loss_voxel_geo_scal，loss_voxel_lovasz

        Args:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:

        """
        loss = dict()
        voxel_semantics = voxel_semantics.long()    # (B, Dx, Dy, Dz)
        preds = occ_pred.permute(0, 4, 1, 2, 3).contiguous()    # (B, n_cls, Dx, Dy, Dz)
        loss_occ = self.loss_occ(
            preds,
            voxel_semantics,
            weight=self.cls_weights.to(preds),
        ) * 100.0
        loss['loss_occ'] = loss_occ
        loss['loss_voxel_sem_scal'] = sem_scal_loss(preds, voxel_semantics)
        loss['loss_voxel_geo_scal'] = geo_scal_loss(preds, voxel_semantics, non_empty_idx=17)
        loss['loss_voxel_lovasz'] = lovasz_softmax(torch.softmax(preds, dim=1), voxel_semantics)

        return loss

    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)     # (B, Dx, Dy, Dz)
        return list(occ_res)

    def get_occ_gpu(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1).int()      # (B, Dx, Dy, Dz)
        return list(occ_res)


@HEADS.register_module()
class BEVOCCHead2D_HPE(BEVOCCHead2D):
    def __init__(self, hpe_channels=32, **kwargs):
        super().__init__(**kwargs)
        self.hpe_channels = hpe_channels
        in_dim = self.in_dim
        self.hpe_proj = nn.Sequential(
            nn.Conv2d(in_dim + hpe_channels, in_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, img_feats, height_prior=None):
        if height_prior is not None:
            img_feats = self.hpe_proj(torch.cat([img_feats, height_prior], dim=1))
        return super().forward(img_feats)
