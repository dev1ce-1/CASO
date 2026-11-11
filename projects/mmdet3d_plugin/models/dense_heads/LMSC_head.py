import numpy as np
import torch
from torch import nn

from mmdet3d.models.builder import HEADS, build_loss
from mmcv.runner import BaseModule
from mmcv.cnn import ConvModule

@HEADS.register_module()
class LMSCHead3D(BaseModule):
    '''
    3D Segmentation heads to retrieve semantic segmentation at each scale.
    Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
    '''

    def __init__(self, in_dim=256, dim=256, Dz=16, num_classes=18, use_predicter=True, class_balance=False,
                 use_mask=True,
                 loss_occ=None, dilations_conv_list=[1, 2, 3]):
        super().__init__()

        # First convolution
        self.conv0 = nn.Conv3d(in_dim, dim, kernel_size=3, padding=1, stride=1)

        # ASPP Block
        self.conv_list = dilations_conv_list
        self.conv1 = nn.ModuleList(
            [nn.Conv3d(dim, dim, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
        self.bn1 = nn.ModuleList([nn.BatchNorm3d(dim) for dil in dilations_conv_list])
        self.conv2 = nn.ModuleList(
            [nn.Conv3d(dim, dim, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
        self.bn2 = nn.ModuleList([nn.BatchNorm3d(dim) for dil in dilations_conv_list])
        self.relu = nn.ReLU(inplace=True)

        # Convolution for output
        self.conv_classes = nn.Conv3d(dim, num_classes * Dz, kernel_size=3, padding=1, stride=1)

        self.use_mask = use_mask
        self.loss_occ = build_loss(loss_occ)
        self.Dz = Dz
        self.num_classes = num_classes
        self.class_balance = class_balance

    def forward(self, x_in):
        # (B, C, Dy, Dx)
        bs, _, Dy, Dx = x_in.shape

        # Dimension exapension
        # x_in = x_in[:, None, :, :, :]

        x_in = x_in[:, :, None, :, :]

        # Convolution to go from inplanes to planes features...
        x_in = self.relu(self.conv0(x_in))

        y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
        for i in range(1, len(self.conv_list)):
            y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
        x_in = self.relu(y + x_in)  # modified

        x_in = self.conv_classes(x_in)

        x_in.view(bs, Dx, Dy, self.Dz, self.num_classes)

        return x_in

    def loss(self, occ_pred, voxel_semantics, mask_camera):
        loss = dict()
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)  # (B, Dx, Dy, Dz)
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
                preds,  # (B*Dx*Dy*Dz, n_cls)
                voxel_semantics,  # (B*Dx*Dy*Dz, )
                mask_camera,  # (B*Dx*Dy*Dz, )
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
        occ_score = occ_pred.softmax(-1)  # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)  # (B, Dx, Dy, Dz)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)  # (B, Dx, Dy, Dz)
        return list(occ_res)


@HEADS.register_module()
class LMSCHead2D(BaseModule):
    '''
    3D Segmentation heads to retrieve semantic segmentation at each scale.
    Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
    '''

    def __init__(self, in_dim=256, dim=256, Dz=16, num_classes=18, use_predicter=True, class_balance=False,
                 use_mask=True,
                 loss_occ=None, dilations_conv_list=[1, 2, 3]):
        super().__init__()

        # First convolution
        self.conv0 = nn.Conv2d(in_dim, dim, kernel_size=3, padding=1, stride=1)

        # ASPP Block
        self.conv_list = dilations_conv_list
        self.conv1 = nn.ModuleList(
            [nn.Conv2d(dim, dim, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(dim) for dil in dilations_conv_list])
        self.conv2 = nn.ModuleList(
            [nn.Conv2d(dim, dim, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(dim) for dil in dilations_conv_list])
        self.relu = nn.ReLU(inplace=True)

        # Convolution for output
        self.conv_classes = nn.Conv2d(dim, num_classes * Dz, kernel_size=3, padding=1, stride=1)

        self.use_mask = use_mask
        self.loss_occ = build_loss(loss_occ)
        self.Dz = Dz
        self.num_classes = num_classes
        self.class_balance = class_balance

    def forward(self, x_in):
        # (B, C, Dy, Dx)
        bs, _, Dy, Dx = x_in.shape


        # Convolution to go from inplanes to planes features...
        x_in = self.relu(self.conv0(x_in))

        y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
        for i in range(1, len(self.conv_list)):
            y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
        x_in = self.relu(y + x_in)  # modified

        x_in = self.conv_classes(x_in)

        x_in.view(bs, Dx, Dy, self.Dz, self.num_classes)

        return x_in

    def loss(self, occ_pred, voxel_semantics, mask_camera):
        loss = dict()
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)  # (B, Dx, Dy, Dz)
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
                preds,  # (B*Dx*Dy*Dz, n_cls)
                voxel_semantics,  # (B*Dx*Dy*Dz, )
                mask_camera,  # (B*Dx*Dy*Dz, )
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
        occ_score = occ_pred.softmax(-1)  # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)  # (B, Dx, Dy, Dz)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)  # (B, Dx, Dy, Dz)
        return list(occ_res)


