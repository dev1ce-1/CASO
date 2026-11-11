import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES

"""
occfusion论文中，对于使用occ3d数据集的情况，将lidar/radar均变换到了ego坐标系而非lidar坐标系
这里follow该论文的方法
"""
@PIPELINES.register_module
class PointTransform(object):
    def __init__(self, lidar_trans=False, radar_trans=False,pc_range=None):
        self.lidar_trans = lidar_trans
        self.radar_trans = radar_trans
        self.pc_range = pc_range

    def __call__(self, results):
        if self.lidar_trans:
            # 解决results['lidar_points']的读取问题
            lidar2ego_rotation = torch.from_numpy(results['lidar2ego'][:3, :3]).to(torch.float32)
            lidar2ego_translation = torch.from_numpy(results['lidar2ego'][:3, -1]).to(torch.float32)
            points = results['points']
            points.tensor[:, :3] = points.tensor[:, :3] @ lidar2ego_rotation.T
            points.tensor[:, :3] += lidar2ego_translation
            # points = torch.from_numpy(points)
            idx = torch.where((points.tensor[:, 0] > self.pc_range[0])
                              & (points.tensor[:, 1] > self.pc_range[1])
                              & (points.tensor[:, 2] > self.pc_range[2])
                              & (points.tensor[:, 0] < self.pc_range[3])
                              & (points.tensor[:, 1] < self.pc_range[4])
                              & (points.tensor[:, 2] < self.pc_range[5]))
            points.tensor = points.tensor[idx]
            results['points'] = points

            return results