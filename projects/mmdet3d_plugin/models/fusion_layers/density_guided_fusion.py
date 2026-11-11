# Copyright (c) 2024. All rights reserved.
# Author: Research Team
# Module: DGCF - Density-Guided Cross-modal Fusion
#
# 创新点说明：
# 本模块提出了一种新型的密度引导跨模态融合（DGCF）机制，用于摄像头-LiDAR
# BEV特征融合，核心创新包括：
#
# 1. 自适应密度估计：从LiDAR BEV特征中隐式估计点云密度分布，无需额外标注。
# 2. 非对称跨模态增强：
#    - LiDAR→Camera路径：以密度图为空间引导，对摄像头特征做几何对齐
#    - Camera→LiDAR路径：以通道注意力方式，将语义信息填充进LiDAR稀疏区域
# 3. 密度自适应门控：最终融合权重由估计密度动态调控，高密度区域信任LiDAR
#    几何精度，低密度区域信任摄像头语义丰富性。
#
# 参考文献/灵感来源：
# - SE-Net (Squeeze-and-Excitation, CVPR 2018) —— 通道注意力思路
# - CBAM (Convolutional Block Attention, ECCV 2018) —— 空间+通道联合注意力
# - BEVFusion (MIT, NeurIPS 2022) —— 多模态BEV融合框架
# - OccFusion (ECCV 2024) —— 占位预测中的多模态融合

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet3d.models.builder import FUSION_LAYERS


class DensityEstimator(nn.Module):
    """
    从LiDAR BEV特征中估计点云密度图。

    网络通过端到端训练隐式学习密度估计，密度图反映了各BEV格子中
    LiDAR点云的覆盖程度，用于后续门控融合。
    """

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        mid_channels = max(in_channels // reduction, 32)
        self.estimator = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, pts_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pts_feat: (B, C, H, W) LiDAR BEV特征
        Returns:
            density: (B, 1, H, W) 密度图，值域[0,1]，越大表示LiDAR点越密集
        """
        return self.estimator(pts_feat)


class CameraToLidarSE(nn.Module):
    """
    Camera→LiDAR 通道式语义迁移模块（基于SE-Net思路）。

    核心逻辑：利用摄像头BEV特征的全局语义上下文，通过通道注意力对
    LiDAR BEV特征做语义校准，使得LiDAR稀疏区域的语义判别力得到增强。
    """

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        mid_channels = max(in_channels // reduction, 32)
        # 对摄像头特征做全局平均池化，提取全局语义描述子
        self.cam_descriptor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                          # (B,C,1,1)
            nn.Conv2d(in_channels, mid_channels, 1),          # (B,C//4,1,1)
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1),          # (B,C,1,1)
            nn.Sigmoid()
        )
        # 带位置感知的局部补充：在全局通道权重基础上增加局部上下文
        self.local_supplement = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, img_feat: torch.Tensor, pts_feat: torch.Tensor,
                density: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_feat:  (B, C, H, W) 摄像头BEV特征
            pts_feat:  (B, C, H, W) LiDAR BEV特征
            density:   (B, 1, H, W) LiDAR密度图
        Returns:
            pts_enhanced: (B, C, H, W) 语义增强后的LiDAR特征
        """
        # 1. 提取摄像头全局通道注意力
        cam_channel_att = self.cam_descriptor(img_feat)       # (B, C, 1, 1)

        # 2. 在LiDAR稀疏区域（1-density大），摄像头语义权重更大
        # 构造稀疏区域自适应权重：sparse_weight = 1 - density
        sparse_weight = 1.0 - density                         # (B, 1, H, W)

        # 3. 用稀疏权重调制通道注意力强度
        cam_att_spatially_modulated = 1.0 + sparse_weight * (cam_channel_att - 0.5) * 2.0

        # 4. 同时利用局部摄像头特征进行补充
        local_ctx = self.local_supplement(torch.cat([pts_feat, img_feat], dim=1))

        # 5. 残差形式的特征增强
        pts_enhanced = pts_feat * cam_att_spatially_modulated + sparse_weight * local_ctx

        return pts_enhanced


class LidarToCameraGeom(nn.Module):
    """
    LiDAR→Camera 密度引导空间几何对齐模块。

    核心逻辑：利用LiDAR密度图作为空间引导，生成空间注意力图，
    使摄像头BEV特征在有LiDAR几何约束的区域得到精细化校准。
    这类似于CBAM空间注意力，但以密度图为外部几何先验。
    """

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        mid_channels = max(in_channels // reduction, 32)
        # 输入：img_feat + density → 输出：空间注意力图
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels + 1, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        # 几何引导的特征细化
        self.geom_refine = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, img_feat: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_feat: (B, C, H, W) 摄像头BEV特征
            density:  (B, 1, H, W) LiDAR密度图
        Returns:
            img_enhanced: (B, C, H, W) 几何对齐后的摄像头特征
        """
        feat_with_density = torch.cat([img_feat, density], dim=1)  # (B, C+1, H, W)

        # 空间注意力：LiDAR密度高的区域获得更强的几何约束
        spatial_att = self.spatial_attn(feat_with_density)          # (B, 1, H, W)

        # 几何细化特征
        geom_feat = self.geom_refine(feat_with_density)             # (B, C, H, W)

        # 残差形式：density高的区域，几何细化贡献更大
        img_enhanced = img_feat + density * spatial_att * geom_feat

        return img_enhanced


class DensityAdaptiveGate(nn.Module):
    """
    密度自适应门控融合模块。

    在增强后的摄像头特征和LiDAR特征之间，以密度图为额外输入，
    学习每个位置、每个通道的最优融合比例。
    - 密度高处：LiDAR几何精度高，门控偏向LiDAR
    - 密度低处：LiDAR稀疏，摄像头语义丰富，门控偏向Camera
    门控为可学习的细化，而非硬规则。
    """

    def __init__(self, in_channels: int):
        super().__init__()
        # 门控网络：输入为双模态特征拼接 + 密度图
        self.gate_net = nn.Sequential(
            nn.Conv2d(in_channels * 2 + 1, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, img_enhanced: torch.Tensor, pts_enhanced: torch.Tensor,
                density: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_enhanced: (B, C, H, W) 几何对齐后的摄像头特征
            pts_enhanced: (B, C, H, W) 语义增强后的LiDAR特征
            density:      (B, 1, H, W) LiDAR密度图
        Returns:
            fused: (B, C, H, W) 融合特征
        """
        # 学习融合门控（值域[0,1]，越接近1表示偏向LiDAR）
        gate = self.gate_net(
            torch.cat([img_enhanced, pts_enhanced, density], dim=1)
        )  # (B, C, H, W)

        # 密度先验 + 学习门控的联合加权
        # 最终LiDAR权重：密度越高，LiDAR权重越大；gate是在密度基础上的精细调节
        lidar_w = 0.5 * (density + gate)        # (B, C, H, W)，值域≈[0,1]
        camera_w = 1.0 - lidar_w

        fused = lidar_w * pts_enhanced + camera_w * img_enhanced
        return fused


@FUSION_LAYERS.register_module()
class DGCFFusion(nn.Module):
    """
    密度引导跨模态融合模块（DGCF: Density-Guided Cross-modal Fusion）

    【核心创新】
    本模块针对摄像头-LiDAR BEV特征融合设计，解决了现有方法的两个关键问题：

    1. 对称双向注意力的局限性：
       现有方法（如CLOcc-v2）对两种模态进行对称的双向可变形注意力融合，
       未考虑两种传感器的互补性差异：LiDAR在几何精度上优越，摄像头在
       语义丰富性上优越。本方法通过非对称设计显式建模这种互补关系。

    2. 梯度阻断问题：
       CLOcc-v2在融合时对特征使用detach()，阻断了梯度从融合路径回传到
       特征提取器，损害了端到端训练效果。本方法移除detach()，恢复完整的
       梯度流。

    【模块结构】
    - DensityEstimator:    从LiDAR BEV特征估计点云密度分布
    - CameraToLidarSE:     Camera→LiDAR，通道式语义迁移（SE-Net风格）
    - LidarToCameraGeom:   LiDAR→Camera，密度引导空间几何对齐（CBAM风格）
    - DensityAdaptiveGate: 密度自适应门控，动态调控最终融合权重
    - OutputProjection:    输出投影（残差连接保证训练稳定）

    【计算效率】
    所有操作均为卷积，无需Attention的O(N²)开销，适合在RTX 3090上高效训练。
    额外参数量约1-2M，相对整体模型可忽略不计。

    Args:
        in_channels (int): 输入特征通道数（摄像头和LiDAR均为此维度）。默认256。
        reduction (int): 中间通道的缩减比例。默认4。
    """

    def __init__(self, in_channels: int = 256, reduction: int = 4):
        super().__init__()
        self.in_channels = in_channels

        # 子模块
        self.density_estimator = DensityEstimator(in_channels, reduction)
        self.cam_to_lidar = CameraToLidarSE(in_channels, reduction)
        self.lidar_to_cam = LidarToCameraGeom(in_channels, reduction)
        self.fusion_gate = DensityAdaptiveGate(in_channels)

        # 输出投影（带残差跳连，增强训练稳定性）
        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.out_act = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.out_proj[-1].weight, 0)

    def forward(self, img_feat: torch.Tensor,
                pts_feat: torch.Tensor):
        """
        Args:
            img_feat: (B, C, H, W) 摄像头BEV特征（由LSS视角变换获得）
            pts_feat: (B, C, H, W) LiDAR BEV特征（由SparseEncoder获得）
        Returns:
            output:  (B, C, H, W) 融合后的BEV特征
            density: (B, 1, H, W) 估计的LiDAR密度图（可用于可视化或辅助损失）
        """
        # fp16下SparseEncoder输出可能含inf/NaN，在进入卷积之前截断保护
        pts_feat = torch.nan_to_num(pts_feat, nan=0.0, posinf=65504.0, neginf=-65504.0)

        identity = img_feat  # 保留残差路径

        # Step 1: 从LiDAR特征估计密度图
        density = self.density_estimator(pts_feat)          # (B, 1, H, W)

        # Step 2: Camera→LiDAR 通道语义迁移
        # 摄像头的全局语义信息校准LiDAR通道权重，重点在LiDAR稀疏区域
        pts_enhanced = self.cam_to_lidar(img_feat, pts_feat, density)  # (B, C, H, W)

        # Step 3: LiDAR→Camera 密度引导几何对齐
        # LiDAR密度图引导摄像头特征的空间注意力，使其与几何边界对齐
        img_enhanced = self.lidar_to_cam(img_feat, density)             # (B, C, H, W)

        # Step 4: 密度自适应门控融合
        # 高密度区域偏向LiDAR（几何精确），低密度区域偏向Camera（语义丰富）
        fused = self.fusion_gate(img_enhanced, pts_enhanced, density)   # (B, C, H, W)

        # Step 5: 输出投影 + 残差连接
        out = self.out_proj(fused)
        out = self.out_act(out + identity)                              # 残差跳连

        return out, density
