"""RiskOcc Stage 4 streaming safety evaluator (v1).

Streams the FlashOCC val split through the model once, simultaneously computing

  * the official Occ3D-nuScenes mIoU (and per-class IoU), matching the formula
    used by `projects.mmdet3d_plugin.core.evaluation.occ_metrics.Metric_mIoU`
    and hence by `tools/test.py`. This is the *hard verification gate*.
  * the seven core safety metrics defined by the Stage 4 protocol, evaluated at
    the default operating point (argmax over the full 18-way softmax).

The script never modifies any source file. It mirrors `tools/probe_logits.py`
for the model/data plumbing and `tools/test.py` for the dataset/eval semantics.

Usage::

  python tools/streaming_safety_evaluator.py \
      --config projects/configs/flashocc/flashocc-r50.py \
      --checkpoint ckpts/flashocc-r50-256x704.pth \
      --protocol configs/eval/safety_protocol.yaml \
      --baseline-miou 31.95 \
      --report reports/04_safety_protocol_report.md
"""

import argparse
import importlib
import json
import os
import sys
import time
import types
import warnings
from collections import OrderedDict


def _install_ray_metrics_stub():
    """Stub out projects.mmdet3d_plugin.core.evaluation.ray_metrics before any
    plugin module imports it. ray_metrics.py JIT-compiles a CUDA extension at
    import time and is only used by the 'ray-iou' evaluation branch, which
    this script never exercises. Skipping the build keeps the evaluator
    runnable on machines whose nvcc lacks support for the host GPU.
    """
    mod_name = 'projects.mmdet3d_plugin.core.evaluation.ray_metrics'
    if mod_name in sys.modules:
        return
    stub = types.ModuleType(mod_name)

    def _ray_metrics_unavailable(*_args, **_kwargs):
        raise RuntimeError(
            'ray_metrics has been stubbed out by streaming_safety_evaluator; '
            'this code path should never run inside the streaming evaluator')

    stub.main = _ray_metrics_unavailable
    stub.main_raypq = _ray_metrics_unavailable
    sys.modules[mod_name] = stub


_install_ray_metrics_stub()

import mmcv  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from mmcv import Config, DictAction  # noqa: E402
from mmcv.parallel import MMDataParallel  # noqa: E402
from mmcv.runner import load_checkpoint, wrap_fp16_model  # noqa: E402

import mmdet  # noqa: E402
from mmdet.apis import set_random_seed  # noqa: E402
from mmdet.datasets import replace_ImageToTensor  # noqa: E402
from mmdet3d.datasets import build_dataloader, build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402

if mmdet.__version__ > '2.23.0':
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg


OCC3D_CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'free'
]


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='RiskOcc Stage 4 streaming safety evaluator (v1)')
    p.add_argument('--config', required=True,
                   help='model/test config file (same as tools/test.py)')
    p.add_argument('--checkpoint', required=True,
                   help='model checkpoint (same as tools/test.py)')
    p.add_argument('--protocol', default='configs/eval/safety_protocol.yaml',
                   help='safety-protocol YAML file')
    p.add_argument('--baseline-miou', type=float, default=None,
                   help='baseline mIoU (percent, e.g. 31.95) reproduced by '
                        'tools/test.py. Used for the hard verification gate.')
    p.add_argument('--miou-tolerance', type=float, default=0.5,
                   help='allowed absolute gap (in percentage points) between '
                        'streaming mIoU and --baseline-miou')
    p.add_argument('--report', default='reports/04_safety_protocol_report.md')
    p.add_argument('--metrics-json',
                   default='reports/04_safety_protocol_metrics.json',
                   help='machine-readable metrics dump path')
    p.add_argument('--max-frames', type=int, default=None,
                   help='cap on number of frames (default = full val)')
    p.add_argument('--gpu-id', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--no-aavt', action='store_true',
                   help='do not align after view transformer, mirroring test.py')
    p.add_argument('--cfg-options', nargs='+', action=DictAction,
                   help='override config options, same format as tools/test.py')
    args = p.parse_args()
    os.environ.setdefault('LOCAL_RANK', '0')
    return args


# ----------------------------------------------------------------------------
# Build (mirror of tools/probe_logits.build_probe_objects)
# ----------------------------------------------------------------------------

def import_plugin(cfg, config_path):
    if not getattr(cfg, 'plugin', False):
        return
    if hasattr(cfg, 'plugin_dir'):
        module_parts = os.path.dirname(cfg.plugin_dir).split('/')
    else:
        module_parts = os.path.dirname(config_path).split('/')
    module_path = module_parts[0]
    for part in module_parts[1:]:
        module_path = module_path + '.' + part
    importlib.import_module(module_path)


def build_eval_objects(args):
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    import_plugin(cfg, args.config)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    cfg.gpu_ids = [args.gpu_id]
    distributed = False

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }
    test_loader_cfg['samples_per_gpu'] = 1
    test_loader_cfg['shuffle'] = False

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    if not args.no_aavt:
        if '4D' in cfg.model.type:
            cfg.model.align_after_view_transfromation = True
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16', None) is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        model.PALETTE = dataset.PALETTE

    model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    model.eval()
    return cfg, dataset, data_loader, model


# ----------------------------------------------------------------------------
# Hook setup (capture occ_head logits)
# ----------------------------------------------------------------------------

def register_occ_head_hook(model):
    captured = {}
    root = model.module if hasattr(model, 'module') else model
    target = None
    for name, mod in root.named_modules():
        if name.endswith('occ_head') or name == 'occ_head':
            target = (name, mod)
    if target is None:
        raise RuntimeError(
            'no module named "occ_head" found on the model; cannot capture '
            'logits for safety evaluation')

    def hook(_module, _inputs, output):
        captured['logits'] = output

    handle = target[1].register_forward_hook(hook)
    return captured, handle, target[0]


# ----------------------------------------------------------------------------
# GT lookup (mirror of tools/probe_logits.gt_from_dataset)
# ----------------------------------------------------------------------------

def candidate_gt_paths(dataset, index):
    info = dataset.data_infos[index]
    occ_path = info.get('occ_path', None)
    if occ_path is None:
        return []

    data_root = getattr(dataset, 'data_root', '')
    base_paths = [occ_path]
    if data_root and not os.path.isabs(occ_path):
        base_paths.append(os.path.join(data_root, occ_path))

    paths = []
    for base in base_paths:
        paths.append(os.path.join(base, 'labels.npz'))
        if 'data/nuscenes/gts/' in base:
            paths.append(os.path.join(
                base.replace('data/nuscenes/gts/',
                             'data/nuscenes/occ3d_panoptic/'),
                'labels.npz'))
    return paths


def load_gt(dataset, index):
    for path in candidate_gt_paths(dataset, index):
        if os.path.exists(path):
            data = np.load(path)
            semantics = data['semantics']
            mask_camera = data['mask_camera'].astype(bool) \
                if 'mask_camera' in data.files else None
            return semantics, mask_camera, path
    raise FileNotFoundError(
        f'no labels.npz found for sample index {index}')


# ----------------------------------------------------------------------------
# Logits normalization (mirror of tools/probe_logits.normalize_logits)
# ----------------------------------------------------------------------------

def normalize_logits(tensor, num_classes, dz):
    """Return logits in [B, X, Y, Z, C] layout."""
    if tensor.dim() == 5:
        # find class dim (last one matching num_classes preferred)
        dims = [idx for idx, size in enumerate(tensor.shape)
                if size == num_classes]
        if not dims:
            raise RuntimeError(
                f'no class dim equals num_classes={num_classes}; shape='
                f'{tuple(tensor.shape)}')
        class_dim = dims[-1]
        order = [idx for idx in range(tensor.dim()) if idx != class_dim]
        order.append(class_dim)
        return tensor.permute(*order).contiguous()
    if tensor.dim() == 4 and dz is not None \
            and tensor.shape[-1] == num_classes * dz:
        b, x, y, _ = tensor.shape
        return tensor.contiguous().view(b, x, y, dz, num_classes)
    raise RuntimeError(
        f'unsupported logits shape {tuple(tensor.shape)} for '
        f'num_classes={num_classes}, Dz={dz}')


# ----------------------------------------------------------------------------
# Protocol resolution
# ----------------------------------------------------------------------------

def resolve_protocol(protocol_path):
    with open(protocol_path, 'r') as f:
        proto = yaml.safe_load(f)
    class_names = proto.get('class_names') or OCC3D_CLASS_NAMES
    name_to_id = {n: i for i, n in enumerate(class_names)}

    def to_ids(names, key):
        ids = []
        for n in names:
            if n not in name_to_id:
                raise RuntimeError(
                    f'class "{n}" in {key} not present in class_names')
            ids.append(name_to_id[n])
        return sorted(set(ids))

    proto['_class_names_used'] = class_names
    proto['_dynamic_ids'] = to_ids(
        proto['dynamic_classes'], 'dynamic_classes')
    proto['_vulnerable_ids'] = to_ids(
        proto['vulnerable_classes'], 'vulnerable_classes')
    occupied_set = set(proto['occupied_ids'])
    proto['_static_ids'] = sorted(
        c for c in occupied_set if c not in set(proto['_dynamic_ids']))
    return proto


# ----------------------------------------------------------------------------
# Streaming accumulator
# ----------------------------------------------------------------------------

class StreamingAccumulator:
    """Owns all running counts so per-frame tensors can be freed."""

    def __init__(self, proto, grid_shape, device):
        self.proto = proto
        self.device = device
        self.num_classes = int(proto['num_classes'])
        self.free_id = int(proto['free_id'])
        self.occupied_ids = list(proto['occupied_ids'])
        self.dynamic_ids = list(proto['_dynamic_ids'])
        self.vulnerable_ids = list(proto['_vulnerable_ids'])
        self.static_ids = list(proto['_static_ids'])
        self.d_near = float(proto['d_near'])
        self.d_block = float(proto['d_block'])
        self.z_plan = list(proto['z_plan'])
        self.pc_range = list(proto['pc_range'])
        self.voxel_size = float(proto['voxel_size'])
        self.distance_bins = OrderedDict(
            [(name, tuple(proto['distance_bins'][name]))
             for name in ['near', 'mid', 'far']])
        self.grid_shape = tuple(int(x) for x in grid_shape)

        # 18x18 confusion matrix (rows=gt, cols=pred), accumulated on CPU.
        self.hist = np.zeros((self.num_classes, self.num_classes),
                             dtype=np.float64)

        # Per-frame counts for sanity checks.
        self.frames = 0
        self.valid_voxels = 0

        # Safety metric counters.
        self.cnt = {
            'HighRiskFN_num': 0, 'HighRiskFN_den': 0,
            'NearOccRecall_num': 0, 'NearOccRecall_den': 0,
            'DynamicFN_num': 0, 'DynamicFN_den': 0,
            'NearFP_num': 0, 'NearFP_den': 0,
            'OccupiedVolumeRatio_num': 0, 'OccupiedVolumeRatio_den': 0,
            'FreeSpaceBlocking_num': 0, 'FreeSpaceBlocking_den': 0,
            'SCIoU_inter': 0, 'SCIoU_union': 0,
        }

        # distance(near/mid/far) x group(dynamic/vulnerable/static) FN matrix.
        self.bin_names = ['near', 'mid', 'far']
        self.group_names = ['dynamic', 'vulnerable', 'static']
        self.fn_num = np.zeros((3, 3), dtype=np.int64)
        self.fn_den = np.zeros((3, 3), dtype=np.int64)

        # Coarse histograms for follow-up tooling. These do NOT store
        # per-voxel arrays. Bin edges fixed by protocol.
        nbins = int(proto.get('prob_histogram_bins') or 0)
        self.hist_bins = nbins
        if nbins > 0:
            self.p_occ_hist = np.zeros(nbins, dtype=np.int64)
            self.s_v_pos_hist = np.zeros(nbins, dtype=np.int64)
            self.s_v_neg_hist = np.zeros(nbins, dtype=np.int64)
            self.s_v_max = 30.0  # logits saturated outside this range
        else:
            self.p_occ_hist = None
            self.s_v_pos_hist = None
            self.s_v_neg_hist = None

        self._precompute_static(device)

    def _precompute_static(self, device):
        x_count, y_count, z_count = self.grid_shape
        vs = self.voxel_size
        pr = self.pc_range
        xs = (torch.arange(x_count, dtype=torch.float32, device=device) + 0.5) \
            * vs + pr[0]
        ys = (torch.arange(y_count, dtype=torch.float32, device=device) + 0.5) \
            * vs + pr[1]
        zs = (torch.arange(z_count, dtype=torch.float32, device=device) + 0.5) \
            * vs + pr[2]
        x_grid, y_grid = torch.meshgrid(xs, ys, indexing='ij')
        bev_dist = torch.sqrt(x_grid.square() + y_grid.square())  # [X, Y]
        self.bev_dist = bev_dist
        self.bev_dist_3d = bev_dist[:, :, None].expand(x_count, y_count,
                                                      z_count)
        # Distance-bin tensor: 0=near, 1=mid, 2=far. Shape [X, Y, Z].
        bin_id = torch.full((x_count, y_count, z_count), 2, dtype=torch.int64,
                            device=device)
        near_lo, near_hi = self.distance_bins['near']
        mid_lo, mid_hi = self.distance_bins['mid']
        bin_id[(self.bev_dist_3d >= near_lo) & (self.bev_dist_3d < near_hi)] = 0
        bin_id[(self.bev_dist_3d >= mid_lo) & (self.bev_dist_3d < mid_hi)] = 1
        self.bin_id_3d = bin_id

        # Near voxel mask (single threshold for NearOccRecall / NearFP).
        self.near_mask = self.bev_dist_3d <= self.d_near
        # Near column mask for FreeSpaceBlocking.
        self.near_col_mask = bev_dist <= self.d_block

        # Z slab for blocking (z_plan inclusive at lo, inclusive at hi).
        z_lo, z_hi = self.z_plan
        z_in_slab = (zs >= z_lo) & (zs <= z_hi)
        self.z_in_slab = z_in_slab  # [Z]
        if z_in_slab.sum().item() == 0:
            warnings.warn(
                f'z_plan={self.z_plan} selects no z voxels; '
                f'FreeSpaceBlocking will be zero by construction')

        # Group membership lookup: for each class id 0..C-1, a code in
        # {0=dynamic, 1=vulnerable, 2=static, 3=none/free}.
        # The protocol's metric definitions treat vulnerable as a subset of
        # dynamic, but for the FN matrix we follow the document's "G(y_v) =
        # dynamic/vulnerable/static" partition. We resolve overlap by giving
        # vulnerable priority over dynamic so VRUs land in the vulnerable row.
        group = np.full(self.num_classes, 3, dtype=np.int64)
        for cid in self.dynamic_ids:
            group[cid] = 0
        for cid in self.vulnerable_ids:
            group[cid] = 1
        for cid in self.static_ids:
            group[cid] = 2
        self.group_lookup = torch.from_numpy(group).to(device)

        # Dynamic and vulnerable id sets as boolean lookup tables.
        dyn = np.zeros(self.num_classes, dtype=bool)
        dyn[self.dynamic_ids] = True
        vul = np.zeros(self.num_classes, dtype=bool)
        vul[self.vulnerable_ids] = True
        # The document spec for DynamicFN says "y_v in dynamic_classes" where
        # dynamic_classes already includes the vulnerable VRUs, so we use the
        # union.
        self.dynamic_union_lookup = torch.from_numpy(
            np.logical_or(dyn, vul)).to(device)

    @staticmethod
    def _bincount2d(a, b, n):
        """Return a length-n^2 bincount of (n*a + b) where a, b are int64."""
        return torch.bincount(n * a + b, minlength=n * n)

    def add_frame(self, logits_bxyzc, gt_np, mask_camera_np):
        """Accumulate one frame of [1, X, Y, Z, C] logits."""
        device = logits_bxyzc.device
        logits = logits_bxyzc[0]  # [X, Y, Z, C]
        x_count, y_count, z_count, n_cls = logits.shape
        assert n_cls == self.num_classes, \
            f'logits class dim {n_cls} != num_classes {self.num_classes}'
        assert (x_count, y_count, z_count) == self.grid_shape, \
            f'logits grid {(x_count, y_count, z_count)} != ' \
            f'protocol grid {self.grid_shape}'

        gt = torch.from_numpy(gt_np.astype(np.int64)).to(device=device)
        if mask_camera_np is None:
            # No mask: fall back to "everywhere gt is a valid label".
            mask = (gt >= 0) & (gt < self.num_classes)
        else:
            mask = torch.from_numpy(mask_camera_np.astype(bool)).to(
                device=device)

        # Filter out non-label voxels (e.g. 255) regardless of mask source.
        label_valid = (gt >= 0) & (gt < self.num_classes)
        valid = mask & label_valid

        valid_count = int(valid.sum().item())
        self.valid_voxels += valid_count

        # ------------------------------------------------------------------
        # Predictions (default operating point = argmax over 18 classes).
        # ------------------------------------------------------------------
        pred = logits.argmax(dim=-1)            # [X, Y, Z], int64
        pred_occ = pred != self.free_id          # bool
        pred_free = ~pred_occ
        gt_occ = (gt != self.free_id) & label_valid
        gt_free = (gt == self.free_id) & label_valid

        # ------------------------------------------------------------------
        # mIoU histogram (18x18). Match Metric_mIoU.hist_info exactly:
        # only voxels inside mask_camera and with valid GT contribute.
        # ------------------------------------------------------------------
        if valid_count > 0:
            gt_v = gt[valid]
            pred_v = pred[valid]
            flat = self._bincount2d(gt_v, pred_v, self.num_classes)
            self.hist += flat.detach().cpu().numpy().reshape(
                self.num_classes, self.num_classes)
            self.cnt['SCIoU_inter'] += int(
                (valid & gt_occ & pred_occ).sum().item())
            self.cnt['SCIoU_union'] += int(
                (valid & (gt_occ | pred_occ)).sum().item())

        # ------------------------------------------------------------------
        # Safety metric counters.
        # ------------------------------------------------------------------
        near_mask = self.near_mask  # [X, Y, Z], bool, on device

        # HighRiskFN: GT-occupied AND (near OR dynamic_union) -> high-risk
        #   * follow document: H_v = I_v AND O_v AND (Near_v
        #                       OR y_v in dynamic
        #                       OR y_v in vulnerable
        #                       OR BoundaryGT_v)
        #     We do not have boundary GT (no oracle), so BoundaryGT_v=False.
        dyn_union_mask = self.dynamic_union_lookup[gt.clamp(min=0,
                                                             max=self.num_classes - 1)]
        # Clamp is safe because we only use this mask where label_valid is True.
        is_dyn_or_vul = dyn_union_mask & label_valid
        high_risk_gt = valid & gt_occ & (near_mask | is_dyn_or_vul)
        self.cnt['HighRiskFN_num'] += int(
            (high_risk_gt & pred_free).sum().item())
        self.cnt['HighRiskFN_den'] += int(high_risk_gt.sum().item())

        # NearOccRecall
        near_occ = valid & gt_occ & near_mask
        self.cnt['NearOccRecall_num'] += int(
            (near_occ & pred_occ).sum().item())
        self.cnt['NearOccRecall_den'] += int(near_occ.sum().item())

        # DynamicFN
        dyn_gt = valid & gt_occ & is_dyn_or_vul
        self.cnt['DynamicFN_num'] += int((dyn_gt & pred_free).sum().item())
        self.cnt['DynamicFN_den'] += int(dyn_gt.sum().item())

        # NearFP
        near_free = valid & gt_free & near_mask
        self.cnt['NearFP_num'] += int((near_free & pred_occ).sum().item())
        self.cnt['NearFP_den'] += int(near_free.sum().item())

        # OccupiedVolumeRatio
        self.cnt['OccupiedVolumeRatio_num'] += int(
            (valid & pred_occ).sum().item())
        self.cnt['OccupiedVolumeRatio_den'] += int(valid.sum().item())

        # ------------------------------------------------------------------
        # Free-Space Blocking (column-level, BEV).
        # ------------------------------------------------------------------
        z_in_slab = self.z_in_slab.view(1, 1, z_count)  # [1,1,Z]
        valid_slab = valid & z_in_slab                 # [X,Y,Z]
        slab_present_col = valid_slab.any(dim=2)       # [X,Y]
        # gt_free_slab voxel: voxel is in slab, valid, and gt is free.
        gt_free_slab = gt_free & z_in_slab & mask & label_valid
        gt_occ_slab = gt_occ & z_in_slab & mask & label_valid
        # GT-free column = every in-slab valid voxel is free, AND at least one
        # valid voxel exists (so we ignore columns that have no observed
        # voxels in the slab).
        gt_free_col = (~gt_occ_slab.any(dim=2)) & slab_present_col
        pred_blocked_col = (valid_slab & pred_occ.view(x_count, y_count,
                                                       z_count)).any(dim=2)
        near_col = self.near_col_mask
        col_mask = gt_free_col & near_col
        self.cnt['FreeSpaceBlocking_num'] += int(
            (col_mask & pred_blocked_col).sum().item())
        self.cnt['FreeSpaceBlocking_den'] += int(col_mask.sum().item())

        # ------------------------------------------------------------------
        # distance x class-group FN matrix.
        # ------------------------------------------------------------------
        # group: 0=dynamic-non-VRU, 1=vulnerable, 2=static, 3=none/free
        group_3d = self.group_lookup[gt.clamp(min=0,
                                              max=self.num_classes - 1)]
        bin_3d = self.bin_id_3d
        gt_valid_mask = valid & gt_occ
        for b_idx in range(3):
            for g_idx in range(3):
                cell_mask = gt_valid_mask & (bin_3d == b_idx) \
                            & (group_3d == g_idx)
                den = int(cell_mask.sum().item())
                if den == 0:
                    continue
                num = int((cell_mask & pred_free).sum().item())
                self.fn_num[b_idx, g_idx] += num
                self.fn_den[b_idx, g_idx] += den

        # ------------------------------------------------------------------
        # Optional coarse histograms over (p_occ, s_v).
        # ------------------------------------------------------------------
        if self.hist_bins > 0 and valid_count > 0:
            p = torch.softmax(logits, dim=-1)
            p_occ = (1.0 - p[..., self.free_id])[valid]
            occ_logits = torch.cat(
                [logits[..., :self.free_id],
                 logits[..., self.free_id + 1:]], dim=-1)
            s_v = (torch.logsumexp(occ_logits, dim=-1)
                   - logits[..., self.free_id])[valid]
            p_idx = torch.clamp((p_occ * self.hist_bins).long(),
                                 0, self.hist_bins - 1)
            self.p_occ_hist += torch.bincount(
                p_idx, minlength=self.hist_bins
            ).detach().cpu().numpy()[: self.hist_bins]
            s_norm = torch.clamp(s_v / self.s_v_max,
                                 -1.0 + 1e-6, 1.0 - 1e-6)
            s_idx = ((s_norm.abs() * self.hist_bins).long()).clamp(
                0, self.hist_bins - 1)
            s_pos_mask = s_v >= 0
            pos_hist = torch.bincount(s_idx[s_pos_mask],
                                       minlength=self.hist_bins
                                       ).detach().cpu().numpy()[
                       : self.hist_bins]
            neg_hist = torch.bincount(s_idx[~s_pos_mask],
                                       minlength=self.hist_bins
                                       ).detach().cpu().numpy()[
                       : self.hist_bins]
            self.s_v_pos_hist += pos_hist
            self.s_v_neg_hist += neg_hist
            del p, p_occ, occ_logits, s_v

        self.frames += 1

        # Cleanup.
        del gt, mask, valid, label_valid, pred, pred_occ, pred_free, gt_occ
        del gt_free, dyn_union_mask, is_dyn_or_vul, high_risk_gt, near_occ
        del dyn_gt, near_free, z_in_slab, valid_slab, slab_present_col
        del gt_free_slab, gt_occ_slab, gt_free_col, pred_blocked_col
        del col_mask, group_3d, gt_valid_mask

    # ----- final metric computation ------------------------------------

    def per_class_iou(self):
        """Return per-class IoU as a length-num_classes float array."""
        with np.errstate(divide='ignore', invalid='ignore'):
            diag = np.diag(self.hist)
            iou = diag / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - diag)
        return iou

    def mIoU_official(self):
        """mean IoU over occupied classes 0..num_classes-2, x 100, rounded
        to 2 decimal places (matches Metric_mIoU.count_miou)."""
        iou = self.per_class_iou()
        sub = iou[: self.num_classes - 1]
        return float(round(np.nanmean(sub) * 100, 2))

    def sc_iou(self):
        """Binary occupied/free scene-completion IoU under the valid region."""
        union = self.cnt['SCIoU_union']
        if union == 0:
            return float('nan')
        return float(self.cnt['SCIoU_inter']) / float(union)

    def safety_metrics(self):
        def ratio(num_key, den_key):
            den = self.cnt[den_key]
            if den == 0:
                return float('nan')
            return float(self.cnt[num_key]) / float(den)

        m = OrderedDict()
        m['HighRiskFN'] = ratio('HighRiskFN_num', 'HighRiskFN_den')
        m['NearOccRecall'] = ratio('NearOccRecall_num', 'NearOccRecall_den')
        m['DynamicFN'] = ratio('DynamicFN_num', 'DynamicFN_den')
        m['NearFP'] = ratio('NearFP_num', 'NearFP_den')
        m['FreeSpaceBlocking'] = ratio('FreeSpaceBlocking_num',
                                       'FreeSpaceBlocking_den')
        m['OccupiedVolumeRatio'] = ratio('OccupiedVolumeRatio_num',
                                         'OccupiedVolumeRatio_den')
        return m

    def fn_matrix(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            rate = np.where(self.fn_den > 0,
                            self.fn_num / np.maximum(self.fn_den, 1),
                            np.nan)
        rate = np.where(self.fn_den > 0, rate, np.nan)
        return rate


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

def fmt(value, digits=4):
    if value is None:
        return 'nan'
    if isinstance(value, float) and np.isnan(value):
        return 'nan'
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def write_report(report_path, metrics_json_path, args, proto, accum,
                 baseline_miou, miou_tolerance, miou_pass, safety_pass,
                 selected_module, frames, wall_seconds, peak_mem_mb,
                 grid_shape, pc_range):
    mmcv.mkdir_or_exist(os.path.dirname(report_path) or '.')
    mmcv.mkdir_or_exist(os.path.dirname(metrics_json_path) or '.')

    miou = accum.mIoU_official()
    sc_iou = accum.sc_iou()
    iou = accum.per_class_iou()
    class_names = proto['_class_names_used']
    safety = accum.safety_metrics()
    fn_mat = accum.fn_matrix()
    dyn_iou_vals = [iou[c] for c in accum.dynamic_ids
                    if c < len(iou) and not np.isnan(iou[c])]
    dynamic_class_iou = float(np.mean(dyn_iou_vals)) if dyn_iou_vals else \
        float('nan')

    json_payload = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'protocol': args.protocol,
        'frames_processed': frames,
        'wall_seconds': wall_seconds,
        'peak_memory_mb': peak_mem_mb,
        'grid_shape': list(grid_shape),
        'pc_range': list(pc_range),
        'valid_region': proto.get('valid_region'),
        'default_operating_point': proto.get('default_operating_point'),
        'baseline_miou_percent': baseline_miou,
        'miou_tolerance_pp': miou_tolerance,
        'streaming_miou_percent': miou,
        'SC_IoU': sc_iou,
        'SC_IoU_inter': int(accum.cnt['SCIoU_inter']),
        'SC_IoU_union': int(accum.cnt['SCIoU_union']),
        'miou_gap_pp': (miou - baseline_miou) if baseline_miou is not None
        else None,
        'miou_pass': bool(miou_pass) if baseline_miou is not None else None,
        'safety_gate_pass': bool(safety_pass),
        'per_class_iou': {class_names[c]: float(iou[c])
                          for c in range(len(class_names))
                          if c < len(iou)},
        'dynamic_class_iou_mean': dynamic_class_iou,
        'safety_metrics': {k: (None if (isinstance(v, float) and np.isnan(v))
                               else float(v)) for k, v in safety.items()},
        'fn_matrix': {
            f'{b}__{g}': (None if np.isnan(fn_mat[bi, gi])
                          else float(fn_mat[bi, gi]))
            for bi, b in enumerate(accum.bin_names)
            for gi, g in enumerate(accum.group_names)
        },
        'fn_matrix_num': {
            f'{b}__{g}': int(accum.fn_num[bi, gi])
            for bi, b in enumerate(accum.bin_names)
            for gi, g in enumerate(accum.group_names)
        },
        'fn_matrix_den': {
            f'{b}__{g}': int(accum.fn_den[bi, gi])
            for bi, b in enumerate(accum.bin_names)
            for gi, g in enumerate(accum.group_names)
        },
    }
    if accum.hist_bins > 0:
        json_payload['p_occ_histogram'] = accum.p_occ_hist.tolist()
        json_payload['s_v_pos_histogram'] = accum.s_v_pos_hist.tolist()
        json_payload['s_v_neg_histogram'] = accum.s_v_neg_hist.tolist()
        json_payload['histogram_bins'] = accum.hist_bins
        json_payload['s_v_max'] = accum.s_v_max

    with open(metrics_json_path, 'w') as f:
        json.dump(json_payload, f, indent=2, sort_keys=True)

    status_miou = (
        'PASS' if (baseline_miou is not None and miou_pass)
        else ('FAIL' if baseline_miou is not None else 'N/A'))
    status_safety = 'PASS' if safety_pass else 'FAIL'

    lines = []
    lines.append('# RiskOcc Stage 4 Safety Protocol Report (v1)')
    lines.append('')
    lines.append(f'- config: `{args.config}`')
    lines.append(f'- checkpoint: `{args.checkpoint}`')
    lines.append(f'- protocol: `{args.protocol}`')
    lines.append(f'- selected logits module: `{selected_module}`')
    lines.append(f'- grid_shape: `{tuple(grid_shape)}`  '
                 f'pc_range: `{pc_range}`  voxel_size: '
                 f'`{proto["voxel_size"]}`')
    lines.append(f'- valid_region: `{proto.get("valid_region")}`  '
                 f'default_operating_point: '
                 f'`{proto.get("default_operating_point")}`')
    lines.append(f'- frames processed: `{frames}`  '
                 f'wall time: `{wall_seconds:.1f}` s  '
                 f'peak GPU mem: `{peak_mem_mb:.1f}` MB')
    lines.append('')
    lines.append('## Pipeline verification: streaming mIoU vs tools/test.py')
    lines.append('')
    lines.append('| key | value |')
    lines.append('| --- | --- |')
    lines.append(f'| baseline mIoU (tools/test.py) | '
                 f'`{fmt(baseline_miou, 2)}` |')
    lines.append(f'| streaming mIoU | `{fmt(miou, 2)}` |')
    lines.append(f'| SC-IoU (binary occupied/free) | `{fmt(sc_iou)}` |')
    if baseline_miou is not None:
        lines.append(f'| gap (streaming - baseline, pp) | '
                     f'`{fmt(miou - baseline_miou, 3)}` |')
        lines.append(f'| tolerance (pp) | `{fmt(miou_tolerance, 2)}` |')
    lines.append(f'| **pipeline gate** | **{status_miou}** |')
    lines.append('')
    lines.append('## Per-class IoU')
    lines.append('')
    lines.append('| class id | name | IoU |')
    lines.append('| --- | --- | --- |')
    for c in range(min(len(class_names), len(iou))):
        lines.append(f'| {c} | `{class_names[c]}` | `{fmt(float(iou[c]))}` |')
    lines.append('')
    lines.append(f'- dynamic-class IoU (mean over {accum.dynamic_ids}): '
                 f'`{fmt(dynamic_class_iou)}`')
    lines.append('')
    lines.append('## Core safety metrics at default operating point '
                 '(argmax)')
    lines.append('')
    lines.append('| metric | value | num | den |')
    lines.append('| --- | --- | --- | --- |')
    pairs = [
        ('HighRiskFN', 'HighRiskFN_num', 'HighRiskFN_den'),
        ('NearOccRecall', 'NearOccRecall_num', 'NearOccRecall_den'),
        ('DynamicFN', 'DynamicFN_num', 'DynamicFN_den'),
        ('NearFP', 'NearFP_num', 'NearFP_den'),
        ('FreeSpaceBlocking', 'FreeSpaceBlocking_num',
         'FreeSpaceBlocking_den'),
        ('OccupiedVolumeRatio', 'OccupiedVolumeRatio_num',
         'OccupiedVolumeRatio_den'),
    ]
    for name, num_key, den_key in pairs:
        lines.append(
            f'| {name} | `{fmt(safety[name])}` | '
            f'`{accum.cnt[num_key]}` | `{accum.cnt[den_key]}` |')
    lines.append('')
    lines.append(f'- **safety gate**: **{status_safety}** (all seven metrics '
                 'produced finite numbers on full val)')
    lines.append('')
    lines.append('## distance x class-group FN matrix')
    lines.append('')
    lines.append('Rows = BEV distance bins, columns = GT class group. '
                 'Vulnerable VRUs (pedestrian/bicycle/motorcycle) are pulled '
                 'out of the dynamic column to highlight VRU risk.')
    lines.append('')
    lines.append('| bin | dynamic-non-VRU | vulnerable | static |')
    lines.append('| --- | --- | --- | --- |')
    for bi, b in enumerate(accum.bin_names):
        cells = []
        for gi in range(3):
            num = accum.fn_num[bi, gi]
            den = accum.fn_den[bi, gi]
            if den == 0:
                cells.append('`nan (den=0)`')
            else:
                cells.append(f'`{fmt(float(num) / float(den))}` '
                             f'({num}/{den})')
        lines.append(f'| `{b}` | {cells[0]} | {cells[1]} | {cells[2]} |')
    lines.append('')
    lines.append('## Class group membership')
    lines.append('')
    lines.append(f'- dynamic_classes: '
                 f'`{[class_names[c] for c in accum.dynamic_ids]}`')
    lines.append(f'- vulnerable_classes (subset, also counted in dynamic): '
                 f'`{[class_names[c] for c in accum.vulnerable_ids]}`')
    lines.append(f'- static_ids: '
                 f'`{[class_names[c] for c in accum.static_ids]}`')
    lines.append('')
    lines.append(f'machine-readable metrics dumped to '
                 f'`{metrics_json_path}`.')
    lines.append('')

    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    args = parse_args()
    proto = resolve_protocol(args.protocol)

    cfg, dataset, data_loader, model = build_eval_objects(args)
    captured, hook_handle, selected_module = register_occ_head_hook(model)

    # Infer num_classes / Dz from cfg, fall back to protocol values.
    num_classes_cfg = int(getattr(cfg.model.occ_head, 'num_classes',
                                   proto['num_classes']))
    dz_cfg = int(getattr(cfg.model.occ_head, 'Dz', 16))
    if num_classes_cfg != proto['num_classes']:
        raise RuntimeError(
            f'config num_classes={num_classes_cfg} disagrees with protocol '
            f'num_classes={proto["num_classes"]}')

    accum = None
    grid_shape = None
    pc_range = list(proto['pc_range'])
    frames = 0

    device_str = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(args.gpu_id)
    start_t = time.time()

    print(f'[evaluator] starting; valid_region='
          f'{proto.get("valid_region")}, default_operating_point='
          f'{proto.get("default_operating_point")}, '
          f'baseline_miou={args.baseline_miou}')

    try:
        with torch.no_grad():
            for batch_idx, data in enumerate(data_loader):
                if args.max_frames is not None and frames >= args.max_frames:
                    break
                captured.clear()
                _ = model(return_loss=False, rescale=True, **data)
                if 'logits' not in captured:
                    raise RuntimeError(
                        'forward hook did not capture occ_head output; '
                        'check the head structure')

                logits = normalize_logits(captured['logits'],
                                          num_classes_cfg, dz_cfg)
                if accum is None:
                    grid_shape = tuple(int(x) for x in logits.shape[1:4])
                    accum = StreamingAccumulator(proto, grid_shape,
                                                 logits.device)
                    print(f'[evaluator] inferred grid {grid_shape}; '
                          f'protocol pc_range={pc_range}, voxel_size='
                          f'{proto["voxel_size"]}')

                gt_np, mask_camera_np, _ = load_gt(dataset, frames)

                if gt_np.shape != grid_shape:
                    raise RuntimeError(
                        f'GT shape {gt_np.shape} != grid {grid_shape}')

                accum.add_frame(logits, gt_np, mask_camera_np)
                frames += 1

                if frames % 200 == 0:
                    print(f'[evaluator] frame {frames}/'
                          f'{len(data_loader)}; running mIoU='
                          f'{accum.mIoU_official():.2f}')

                # Free per-frame tensors.
                del logits
                captured.clear()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    finally:
        hook_handle.remove()

    wall_seconds = time.time() - start_t
    peak_mem_mb = 0.0
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated(args.gpu_id) / 1024.0 / \
            1024.0

    if accum is None or frames == 0:
        raise RuntimeError('no frames were processed')

    miou = accum.mIoU_official()
    miou_pass = False
    if args.baseline_miou is not None:
        miou_pass = abs(miou - args.baseline_miou) <= args.miou_tolerance

    safety = accum.safety_metrics()
    safety_pass = all(not (isinstance(v, float) and np.isnan(v))
                       for v in safety.values())

    write_report(args.report, args.metrics_json, args, proto, accum,
                 args.baseline_miou, args.miou_tolerance, miou_pass,
                 safety_pass, selected_module, frames, wall_seconds,
                 peak_mem_mb, grid_shape, pc_range)

    print(f'[evaluator] frames={frames} wall={wall_seconds:.1f}s '
          f'peak_mem={peak_mem_mb:.1f}MB')
    print(f'[evaluator] streaming mIoU = {miou:.2f}')
    print(f'[evaluator] SC-IoU = {fmt(accum.sc_iou())}')
    if args.baseline_miou is not None:
        gap = miou - args.baseline_miou
        print(f'[evaluator] baseline mIoU = {args.baseline_miou:.2f}; gap='
              f'{gap:+.3f} pp; tolerance={args.miou_tolerance:.2f} pp -> '
              f'{"PASS" if miou_pass else "FAIL"}')
    for k, v in safety.items():
        print(f'[evaluator] {k} = {fmt(v)}')
    print(f'[evaluator] safety gate -> '
          f'{"PASS" if safety_pass else "FAIL"}')
    print(f'[evaluator] report:  {args.report}')
    print(f'[evaluator] metrics: {args.metrics_json}')


if __name__ == '__main__':
    main()
