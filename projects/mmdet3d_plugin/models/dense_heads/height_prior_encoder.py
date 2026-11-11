# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F


class HeightPriorEncoder(nn.Module):
    """Encode LiDAR voxel coordinates into a BEV height prior descriptor."""

    def __init__(self, Dz=16, out_channels=32):
        super().__init__()
        self.Dz = Dz
        self.out_channels = out_channels

        # Learnable compression from vertical occupancy bins to compact channels.
        self.z_compress = nn.Sequential(
            nn.Conv2d(Dz, out_channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Local spatial aggregation for better context on BEV plane.
        self.spatial_ctx = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def build_height_occupancy(coors, batch_size, sparse_shape, target_shape):
        """
        Build binary occupancy from voxel coordinates and downsample to target shape.

        Args:
            coors (Tensor): (N, 4) with columns (batch_idx, z, y, x), long type.
            batch_size (int): Batch size.
            sparse_shape (tuple): (Dz_s, Hy_s, Wx_s) of sparse voxel space.
            target_shape (tuple): (Dz, H, W) target BEV shape.

        Returns:
            Tensor: (B, Dz, H, W), float32 binary occupancy map.
        """
        Dz_s, Hy_s, Wx_s = sparse_shape
        occ = torch.zeros(
            (batch_size, Dz_s, Hy_s, Wx_s),
            dtype=torch.float32,
            device=coors.device,
        )

        if coors.numel() > 0:
            coors = coors.long()
            occ[coors[:, 0], coors[:, 1], coors[:, 2], coors[:, 3]] = 1.0

        Dz, H, W = target_shape
        occ = F.adaptive_max_pool3d(occ, output_size=(Dz, H, W))
        occ = (occ > 0).to(dtype=torch.float32)
        return occ

    def forward(self, coors, batch_size, sparse_shape, target_shape):
        occ = self.build_height_occupancy(
            coors=coors,
            batch_size=batch_size,
            sparse_shape=sparse_shape,
            target_shape=target_shape,
        )
        height_prior = self.z_compress(occ)
        height_prior = self.spatial_ctx(height_prior)
        return height_prior, occ
