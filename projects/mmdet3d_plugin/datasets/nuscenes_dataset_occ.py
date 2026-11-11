# Copyright (c) OpenMMLab. All rights reserved.
import os
import mmcv
import torch
import cv2
import numpy as np
from tqdm import tqdm

from mmdet3d.datasets import DATASETS
from .nuscenes_dataset_bevdet import NuScenesDatasetBEVDet as NuScenesDataset
from ..core.evaluation.occ_metrics import Metric_mIoU, Metric_FScore
from .ego_pose_dataset import EgoPoseDataset
from ..core.evaluation.ray_metrics import main as calc_rayiou
from torch.utils.data import DataLoader
from ..core.evaluation.ray_metrics import main_raypq
import torch
import glob
import math

from pyquaternion import Quaternion
def nuscenes_get_rt_matrix(
        src_sample,
        dest_sample,
        src_mod,
        dest_mod):
    """
    CAM_FRONT_XYD indicates going from 2d image coords + depth
        Note that image coords need to multiplied with said depths first to bring it into 2d hom coords.
    CAM_FRONT indicates going from camera coordinates xyz

    Method is: whatever the input is, transform to global first.
    """
    possible_mods = ['CAM_FRONT_XYD',
                     'CAM_FRONT_RIGHT_XYD',
                     'CAM_FRONT_LEFT_XYD',
                     'CAM_BACK_XYD',
                     'CAM_BACK_LEFT_XYD',
                     'CAM_BACK_RIGHT_XYD',
                     'CAM_FRONT',
                     'CAM_FRONT_RIGHT',
                     'CAM_FRONT_LEFT',
                     'CAM_BACK',
                     'CAM_BACK_LEFT',
                     'CAM_BACK_RIGHT',
                     'lidar',
                     'ego',
                     'global']

    assert src_mod in possible_mods and dest_mod in possible_mods

    src_lidar_to_ego = np.eye(4, 4)
    src_lidar_to_ego[:3, :3] = Quaternion(src_sample['lidar2ego_rotation']).rotation_matrix
    src_lidar_to_ego[:3, 3] = np.array(src_sample['lidar2ego_translation'])

    src_ego_to_global = np.eye(4, 4)
    src_ego_to_global[:3, :3] = Quaternion(src_sample['ego2global_rotation']).rotation_matrix
    src_ego_to_global[:3, 3] = np.array(src_sample['ego2global_translation'])

    dest_lidar_to_ego = np.eye(4, 4)
    dest_lidar_to_ego[:3, :3] = Quaternion(dest_sample['lidar2ego_rotation']).rotation_matrix
    dest_lidar_to_ego[:3, 3] = np.array(dest_sample['lidar2ego_translation'])

    dest_ego_to_global = np.eye(4, 4)
    dest_ego_to_global[:3, :3] = Quaternion(dest_sample['ego2global_rotation']).rotation_matrix
    dest_ego_to_global[:3, 3] = np.array(dest_sample['ego2global_translation'])

    src_mod_to_global = None
    dest_global_to_mod = None

    if src_mod == "global":
        src_mod_to_global = np.eye(4, 4)
    elif src_mod == "ego":
        src_mod_to_global = src_ego_to_global
    elif src_mod == "lidar":
        src_mod_to_global = src_ego_to_global @ src_lidar_to_ego
    elif "CAM" in src_mod:
        src_sample_cam = src_sample['cams'][src_mod.replace("_XYD", "")]

        src_cam_to_lidar = np.eye(4, 4)
        src_cam_to_lidar[:3, :3] = src_sample_cam['sensor2lidar_rotation']
        src_cam_to_lidar[:3, 3] = src_sample_cam['sensor2lidar_translation']

        src_cam_intrinsics = np.eye(4, 4)
        src_cam_intrinsics[:3, :3] = src_sample_cam['cam_intrinsic']

        if "XYD" not in src_mod:
            src_mod_to_global = (src_ego_to_global @ src_lidar_to_ego @
                                 src_cam_to_lidar)
        else:
            src_mod_to_global = (src_ego_to_global @ src_lidar_to_ego @
                                 src_cam_to_lidar @ np.linalg.inv(src_cam_intrinsics))

    if dest_mod == "global":
        dest_global_to_mod = np.eye(4, 4)
    elif dest_mod == "ego":
        dest_global_to_mod = np.linalg.inv(dest_ego_to_global)
    elif dest_mod == "lidar":
        dest_global_to_mod = np.linalg.inv(dest_ego_to_global @ dest_lidar_to_ego)
    elif "CAM" in dest_mod:
        dest_sample_cam = dest_sample['cams'][dest_mod.replace("_XYD", "")]

        dest_cam_to_lidar = np.eye(4, 4)
        dest_cam_to_lidar[:3, :3] = dest_sample_cam['sensor2lidar_rotation']
        dest_cam_to_lidar[:3, 3] = dest_sample_cam['sensor2lidar_translation']

        dest_cam_intrinsics = np.eye(4, 4)
        dest_cam_intrinsics[:3, :3] = dest_sample_cam['cam_intrinsic']

        if "XYD" not in dest_mod:
            dest_global_to_mod = np.linalg.inv(dest_ego_to_global @ dest_lidar_to_ego @
                                               dest_cam_to_lidar)
        else:
            dest_global_to_mod = np.linalg.inv(dest_ego_to_global @ dest_lidar_to_ego @
                                               dest_cam_to_lidar @ np.linalg.inv(dest_cam_intrinsics))

    return dest_global_to_mod @ src_mod_to_global


colors_map = np.array(
    [
        [0,   0,   0, 255],  # 0 undefined
        [255, 158, 0, 255],  # 1 car  orange
        [0, 0, 230, 255],    # 2 pedestrian  Blue
        [47, 79, 79, 255],   # 3 sign  Darkslategrey
        [220, 20, 60, 255],  # 4 CYCLIST  Crimson
        [255, 69, 0, 255],   # 5 traiffic_light  Orangered
        [255, 140, 0, 255],  # 6 pole  Darkorange
        [233, 150, 70, 255], # 7 construction_cone  Darksalmon
        [255, 61, 99, 255],  # 8 bycycle  Red
        [112, 128, 144, 255],# 9 motorcycle  Slategrey
        [222, 184, 135, 255],# 10 building Burlywood
        [0, 175, 0, 255],    # 11 vegetation  Green
        [165, 42, 42, 255],  # 12 trunk  nuTonomy green
        [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
        [75, 0, 75, 255], # 14 walkable, sidewalk
        [255, 0, 0, 255], # 15 unobsrvd
        [0, 0, 0, 0],  # 16 undefined
        [0, 0, 0, 0],  # 16 undefined
    ])


@DATASETS.register_module()
class NuScenesDatasetOccpancy(NuScenesDataset):
    def __init__(self,
                use_sequence_group_flag=False,
                sequences_split_num=1,
                 **kwargs
                 ):
        self.use_sequence_group_flag = use_sequence_group_flag
        self.sequences_split_num = sequences_split_num
        super().__init__(**kwargs)
        if self.use_sequence_group_flag:
            self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.
        if self.test_mode:
            assert self.sequences_split_num == 1

    # 可以理解为这个函数的输出就是初始的results字典
    # 整个预处理的流程参考custom_3d中的prepare_train_data
    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        input_dict = super(NuScenesDatasetOccpancy, self).get_data_info(index)
        # standard protocol modified from SECOND.Pytorch
        # input_dict['occ_gt_path'] = os.path.join(self.data_root, self.data_infos[index]['occ_path'])
        input_dict['occ_gt_path'] = self.data_infos[index]['occ_path']
        # 用于fuse_history
        if self.use_sequence_group_flag:
            input_dict['sample_index'] = index
            input_dict['sequence_group_idx'] = self.flag[index]
            input_dict['start_of_sequence'] = index == 0 or self.flag[index - 1] != self.flag[index]
            # Get a transformation matrix from current keyframe lidar to previous keyframe lidar
            # if they belong to same sequence.
            input_dict['nuscenes_get_rt_matrix'] = dict(
                lidar2ego_rotation=self.data_infos[index]['lidar2ego_rotation'],
                lidar2ego_translation=self.data_infos[index]['lidar2ego_translation'],
                ego2global_rotation=self.data_infos[index]['ego2global_rotation'],
                ego2global_translation=self.data_infos[index]['ego2global_translation'],
            )
            if not input_dict['start_of_sequence']:
                input_dict['curr_to_prev_lidar_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                    self.data_infos[index], self.data_infos[index - 1],
                    "lidar", "lidar"))
                input_dict['prev_lidar_to_global_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                    self.data_infos[index - 1], self.data_infos[index],
                    "lidar", "global"))  # TODO: Note that global is same for all.
                input_dict['curr_to_prev_ego_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                    self.data_infos[index], self.data_infos[index - 1],
                    "ego", "ego"))
            else:
                input_dict['curr_to_prev_lidar_rt'] = torch.eye(4).float()
                input_dict['prev_lidar_to_global_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                    self.data_infos[index], self.data_infos[index], "lidar", "global")
                )
                input_dict['curr_to_prev_ego_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                    self.data_infos[index], self.data_infos[index],
                    "ego", "ego"))
            input_dict['global_to_curr_lidar_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                self.data_infos[index], self.data_infos[index],
                "global", "lidar"))
        return input_dict

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        metric = eval_kwargs['metric'][0]
        print("metric = ", metric)
        if metric == 'ray-iou':
            occ_gts = []
            occ_preds = []
            lidar_origins = []
            inst_gts = []
            inst_preds = []
            
            print('\nStarting Evaluation...')

            data_loader = DataLoader(
                EgoPoseDataset(self.data_infos),
                batch_size=1,
                shuffle=False,
                num_workers=8
            )

            sample_tokens = [info['token'] for info in self.data_infos]

            for i, batch in enumerate(data_loader):
                # if i > 5:
                #     break
                token = batch[0][0]
                output_origin = batch[1]

                data_id = sample_tokens.index(token)
                info = self.data_infos[data_id]
                # occ_gt = np.load(os.path.join(self.data_root, info['occ_path'], 'labels.npz'))
                # occ_gt = np.load(os.path.join(info['occ_path'], 'labels.npz'))
                occ_gt = np.load(os.path.join(info['occ_path'].replace('data/nuscenes/gts/', 'data/nuscenes/occ3d_panoptic/'), 'labels.npz'))
                gt_semantics = occ_gt['semantics']      # (Dx, Dy, Dz)
                mask_lidar = occ_gt['mask_lidar'].astype(bool)      # (Dx, Dy, Dz)
                mask_camera = occ_gt['mask_camera'].astype(bool)    # (Dx, Dy, Dz)
                occ_pred = occ_results[data_id]['pred_occ'].cpu().numpy()     # (Dx, Dy, Dz)
                # occ_pred = occ_results[data_id]['pred_occ']     # (Dx, Dy, Dz)

                lidar_origins.append(output_origin)
                occ_gts.append(gt_semantics)
                occ_preds.append(occ_pred)

                if 'pano_inst' in occ_results[data_id].keys():
                    pano_inst = occ_results[data_id]['pano_inst'].cpu()
                    # pano_inst = torch.from_numpy(occ_results[data_id]['pano_inst'])
                    pano_inst = pano_inst.squeeze(0).numpy()
                    gt_instances = occ_gt['instances']
                    inst_gts.append(gt_instances)
                    inst_preds.append(pano_inst)
                    
            eval_results = calc_rayiou(occ_preds, occ_gts, lidar_origins)
            if len(inst_preds) > 0:
                eval_results.update(main_raypq(occ_preds, occ_gts, inst_preds, inst_gts, lidar_origins))
            # eval_results = main_raypq(occ_preds, occ_gts, inst_preds, inst_gts, lidar_origins)
        else:
            self.occ_eval_metrics = Metric_mIoU(
                num_classes=18,
                use_lidar_mask=False,
                use_image_mask=True)

            print('\nStarting Evaluation...')
            for index, occ_pred in enumerate(tqdm(occ_results)):
                # occ_pred: (Dx, Dy, Dz)
                info = self.data_infos[index]
                # occ_gt = np.load(os.path.join(self.data_root, info['occ_path'], 'labels.npz'))
                occ_gt = np.load(os.path.join(info['occ_path'], 'labels.npz'))
                gt_semantics = occ_gt['semantics']      # (Dx, Dy, Dz)
                mask_lidar = occ_gt['mask_lidar'].astype(bool)      # (Dx, Dy, Dz)
                mask_camera = occ_gt['mask_camera'].astype(bool)    # (Dx, Dy, Dz)
                # occ_pred = occ_pred
                self.occ_eval_metrics.add_batch(
                    occ_pred['pred_occ'] if (isinstance(occ_pred, dict) and 'pred_occ' in occ_pred) else occ_pred,   # (Dx, Dy, Dz)
                    gt_semantics,   # (Dx, Dy, Dz)
                    mask_lidar,     # (Dx, Dy, Dz)
                    mask_camera     # (Dx, Dy, Dz)
                )

                # if index % 100 == 0 and show_dir is not None:
                #     gt_vis = self.vis_occ(gt_semantics)
                #     pred_vis = self.vis_occ(occ_pred)
                #     mmcv.imwrite(np.concatenate([gt_vis, pred_vis], axis=1),
                #                  os.path.join(show_dir + "%d.jpg"%index))
                
                if show_dir is not None:
                    mmcv.mkdir_or_exist(show_dir)
                    # scene_name = info['scene_name']
                    scene_name = [tem for tem in info['occ_path'].split('/') if 'scene-' in tem][0]
                    sample_token = info['token']
                    mmcv.mkdir_or_exist(os.path.join(show_dir, scene_name, sample_token))
                    save_path = os.path.join(show_dir, scene_name, sample_token, 'pred.npz')
                    np.savez_compressed(save_path, pred=occ_pred['pred_occ'] if (isinstance(occ_pred, dict) and 'pred_occ' in occ_pred) else occ_pred.cpu(), gt=occ_gt, sample_token=sample_token)

            eval_results = self.occ_eval_metrics.count_miou()

        return eval_results


    def vis_occ(self, semantics):
        # simple visualization of result in BEV
        semantics_valid = np.logical_not(semantics == 17)
        d = np.arange(16).reshape(1, 1, 16)
        d = np.repeat(d, 200, axis=0)
        d = np.repeat(d, 200, axis=1).astype(np.float32)
        d = d * semantics_valid
        selected = np.argmax(d, axis=2)

        selected_torch = torch.from_numpy(selected)
        semantics_torch = torch.from_numpy(semantics)

        occ_bev_torch = torch.gather(semantics_torch, dim=2,
                                     index=selected_torch.unsqueeze(-1))
        occ_bev = occ_bev_torch.numpy()

        occ_bev = occ_bev.flatten().astype(np.int32)
        occ_bev_vis = colors_map[occ_bev].astype(np.uint8)
        occ_bev_vis = occ_bev_vis.reshape(200, 200, 4)[::-1, ::-1, :3]
        occ_bev_vis = cv2.resize(occ_bev_vis,(400,400))
        return occ_bev_vis

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """

        res = []
        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and len(self.data_infos[idx]['prev']) == 0:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.sequences_split_num != 1:
            if self.sequences_split_num == 'all':
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0,
                                   bin_counts[curr_flag],
                                   math.ceil(bin_counts[curr_flag] / self.sequences_split_num)))
                        + [bin_counts[curr_flag]])
                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.sequences_split_num
                self.flag = np.array(new_flags, dtype=np.int64)
