import argparse
import importlib
import os
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

import mmdet

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Probe FlashOCC per-voxel logits and grid alignment')
    parser.add_argument('--config', required=True, help='test config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint file')
    parser.add_argument('--gpu-id', type=int, default=0, help='single GPU id')
    parser.add_argument('--max-frames', type=int, default=50)
    parser.add_argument('--report', default='reports/01_logit_alignment_probe.md')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument(
        '--no-aavt',
        action='store_true',
        help='Do not align after view transformer, mirroring tools/test.py')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config options, same format as tools/test.py')
    args = parser.parse_args()
    os.environ.setdefault('LOCAL_RANK', '0')
    return args


def import_plugin(cfg, config_path):
    if not hasattr(cfg, 'plugin') or not cfg.plugin:
        return
    if hasattr(cfg, 'plugin_dir'):
        plugin_dir = cfg.plugin_dir
        module_parts = os.path.dirname(plugin_dir).split('/')
    else:
        module_parts = os.path.dirname(config_path).split('/')
    module_path = module_parts[0]
    for part in module_parts[1:]:
        module_path = module_path + '.' + part
    print(module_path)
    importlib.import_module(module_path)


def build_probe_objects(args):
    # This block mirrors tools/test.py's config, dataset, dataloader, model,
    # checkpoint, MMDataParallel and eval setup, with batch size forced to 1.
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
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
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


def cfg_lookup(cfg, path, default=None):
    obj = cfg
    for part in path.split('.'):
        if isinstance(obj, dict) and part in obj:
            obj = obj[part]
        elif hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            return default
    return obj


def infer_num_classes(cfg):
    for path in [
            'model.occ_head.num_classes',
            'model.pts_bbox_head.num_classes',
            'occ_head.num_classes',
            'num_classes',
    ]:
        value = cfg_lookup(cfg, path)
        if value is not None:
            return int(value)
    return None


def infer_dz(cfg):
    for path in ['model.occ_head.Dz', 'occ_head.Dz', 'Dz']:
        value = cfg_lookup(cfg, path)
        if value is not None:
            return int(value)
    return None


def infer_class_names(cfg, num_classes):
    names = cfg.get('occ_class_names', None)
    if names is None:
        names = cfg.get('class_names', None)
    if names is not None and len(names) == num_classes:
        return list(names), 'config'
    if num_classes == len(OCC3D_CLASS_NAMES):
        return list(OCC3D_CLASS_NAMES), 'occ3d_default'
    if names is not None:
        return list(names), 'config_non_occ_len'
    return None, 'missing'


def infer_free_id(class_names, num_classes):
    if class_names is not None and len(class_names) == num_classes:
        lowered = [str(name).lower() for name in class_names]
        for key in ['free', 'empty', 'go']:
            if key in lowered:
                return lowered.index(key), 'class_names'
    if num_classes == 18:
        return 17, 'occ3d_default'
    return None, 'unknown'


def register_hooks(model):
    captured = OrderedDict()

    def mk(name):
        def hook(_module, _inputs, output):
            captured[name] = output
        return hook

    root = model.module if hasattr(model, 'module') else model
    handles = []
    for name, mod in root.named_modules():
        tag = (name + ' ' + mod.__class__.__name__).lower()
        if any(key in tag for key in ['occ', 'head', 'seg']):
            handles.append(mod.register_forward_hook(mk(name)))
    return captured, handles


def scan_captured(name, obj, rows, verbose=False):
    if torch.is_tensor(obj):
        rows.append(dict(name=name, shape=tuple(obj.shape), dtype=str(obj.dtype),
                         tensor=obj))
        if verbose:
            print('CAPTURED', name, tuple(obj.shape), obj.dtype)
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            scan_captured(f'{name}[{i}]', item, rows, verbose=verbose)
    elif isinstance(obj, dict):
        for key, item in obj.items():
            scan_captured(f'{name}.{key}', item, rows, verbose=verbose)


def select_logits(rows, num_classes, dz):
    candidates = []
    fused_candidates = []
    for row in rows:
        tensor = row['tensor']
        if not tensor.is_floating_point() or tensor.dim() < 4:
            continue
        shape = tuple(tensor.shape)
        dims_with_classes = [
            idx for idx, size in enumerate(shape) if size == num_classes
        ]
        if dims_with_classes:
            score = 0
            if tensor.dim() == 5:
                score += 100
            if shape[-1] == num_classes:
                score += 80
            if 'occ_head' in row['name']:
                score += 40
            if row['name'].endswith('occ_head'):
                score += 30
            score += min(tensor.numel() // max(num_classes, 1), 10_000_000)
            candidates.append((score, row, dims_with_classes))
        if dz is not None and shape[-1] == num_classes * dz and tensor.dim() == 4:
            score = 50
            if 'occ_head' in row['name']:
                score += 40
            fused_candidates.append((score, row, []))

    if candidates:
        _, row, dims = sorted(candidates, key=lambda item: item[0])[-1]
        return row, dims, False
    if fused_candidates:
        _, row, dims = sorted(fused_candidates, key=lambda item: item[0])[-1]
        return row, dims, True
    return None, [], False


def normalize_logits(tensor, num_classes, dz):
    layout_note = ''
    if tensor.dim() == 5:
        dims = [idx for idx, size in enumerate(tensor.shape)
                if size == num_classes]
        if not dims:
            raise RuntimeError(
                f'no class dimension equals num_classes={num_classes}')
        class_dim = dims[-1]
        order = [idx for idx in range(tensor.dim()) if idx != class_dim]
        order.append(class_dim)
        logits = tensor.permute(*order).contiguous()
        layout_note = (
            f'original_shape={tuple(tensor.shape)}, class_dim={class_dim}, '
            f'permute={order}, normalized=[B,X,Y,Z,C]')
        return logits, layout_note, class_dim

    if tensor.dim() == 4 and dz is not None and tensor.shape[-1] == num_classes * dz:
        b, x, y, _ = tensor.shape
        logits = tensor.contiguous().view(b, x, y, dz, num_classes)
        layout_note = (
            f'original_shape={tuple(tensor.shape)}, fused_last_dim=C*Dz, '
            f'Dz={dz}, normalized=[B,X,Y,Z,C]')
        return logits, layout_note, 3

    raise RuntimeError(
        f'unsupported logits shape {tuple(tensor.shape)} for '
        f'num_classes={num_classes}, Dz={dz}')


def infer_geometry(cfg, grid_shape):
    x_count, y_count, z_count = grid_shape
    grid_config = cfg.get('grid_config', None)
    if grid_config is not None and all(k in grid_config for k in ['x', 'y', 'z']):
        x_cfg, y_cfg, z_cfg = grid_config['x'], grid_config['y'], grid_config['z']
        pc_range = [
            float(x_cfg[0]), float(y_cfg[0]), float(z_cfg[0]),
            float(x_cfg[1]), float(y_cfg[1]), float(z_cfg[1])
        ]
    else:
        pc_range = list(cfg.get('point_cloud_range',
                                [-40, -40, -1, 40, 40, 5.4]))
    voxel_size = [
        (pc_range[3] - pc_range[0]) / float(x_count),
        (pc_range[4] - pc_range[1]) / float(y_count),
        (pc_range[5] - pc_range[2]) / float(z_count),
    ]
    return pc_range, voxel_size


def make_near_mask(grid_shape, pc_range, voxel_size, max_dist=20.0):
    x_count, y_count, _ = grid_shape
    xs = (torch.arange(x_count, dtype=torch.float32) + 0.5) * voxel_size[0]
    ys = (torch.arange(y_count, dtype=torch.float32) + 0.5) * voxel_size[1]
    xs = xs + pc_range[0]
    ys = ys + pc_range[1]
    x_grid, y_grid = torch.meshgrid(xs, ys, indexing='ij')
    dist = torch.sqrt(x_grid.square() + y_grid.square())
    near_xy = dist <= max_dist
    return near_xy[:, :, None].expand(grid_shape)


def unwrap_data(value):
    if hasattr(value, 'data'):
        value = value.data
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return unwrap_data(value[0])
    return value


def gt_from_data(data):
    for key in ['voxel_semantics', 'gt_occ', 'occ_gt']:
        if key not in data:
            continue
        value = unwrap_data(data[key])
        if torch.is_tensor(value):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3:
            return arr, None, f'data[{key}]'
    return None, None, None


def candidate_gt_paths(dataset, index):
    info = dataset.data_infos[index]
    occ_path = info.get('occ_path', None)
    if occ_path is None:
        return []

    data_root = getattr(dataset, 'data_root', '')
    paths = []
    base_paths = [occ_path]
    if data_root and not os.path.isabs(occ_path):
        base_paths.append(os.path.join(data_root, occ_path))

    for base in base_paths:
        paths.append(os.path.join(base, 'labels.npz'))
        if 'data/nuscenes/gts/' in base:
            paths.append(os.path.join(
                base.replace('data/nuscenes/gts/',
                             'data/nuscenes/occ3d_panoptic/'),
                'labels.npz'))
    return paths


def gt_from_dataset(dataset, index):
    tried = candidate_gt_paths(dataset, index)
    for path in tried:
        if os.path.exists(path):
            occ_gt = np.load(path)
            semantics = occ_gt['semantics']
            mask_camera = occ_gt['mask_camera'].astype(bool) \
                if 'mask_camera' in occ_gt else None
            return semantics, mask_camera, path
    raise FileNotFoundError(
        f'could not find labels.npz for dataset index {index}; tried {tried}')


def transform_gt(gt, mask, mode):
    if mode == 'normal':
        return gt, mask
    if mode == 'swapXY':
        gt_t = np.swapaxes(gt, 0, 1)
        mask_t = np.swapaxes(mask, 0, 1) if mask is not None else None
        return gt_t, mask_t
    if mode == 'flipX':
        gt_t = gt[::-1, :, :]
        mask_t = mask[::-1, :, :] if mask is not None else None
        return gt_t, mask_t
    if mode == 'flipY':
        gt_t = gt[:, ::-1, :]
        mask_t = mask[:, ::-1, :] if mask is not None else None
        return gt_t, mask_t
    raise ValueError(mode)


def safe_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float32)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return float('nan')
    if roc_auc_score is not None:
        return float(roc_auc_score(labels, scores))

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    pos = labels.astype(bool)
    n_pos = pos.sum()
    n_neg = labels.size - n_pos
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) /
                 (n_pos * n_neg))


def append_variant_scores(variant_scores, s_cpu, gt, mask_camera, near_mask,
                          free_id):
    for mode in ['normal', 'swapXY', 'flipX', 'flipY']:
        gt_t, mask_t = transform_gt(gt, mask_camera, mode)
        if gt_t.shape != s_cpu.shape:
            raise RuntimeError(
                f'{mode} GT shape {gt_t.shape} != logits grid {s_cpu.shape}')
        valid = (gt_t != 255) & near_mask
        if mask_t is not None:
            valid = valid & mask_t.astype(bool)
        labels = (gt_t != free_id) & valid
        variant_scores[mode]['labels'].append(labels[valid].astype(np.uint8))
        variant_scores[mode]['scores'].append(s_cpu[valid].astype(np.float32))


def compute_batch_stats(logits_bxyzc, gt, mask_camera, near_mask, free_id):
    device = logits_bxyzc.device
    logits_xyzc = logits_bxyzc[0]
    gt_t = torch.from_numpy(gt).to(device=device)
    near_t = near_mask.to(device=device)
    valid = (gt_t != 255) & near_t
    if mask_camera is not None:
        mask_t = torch.from_numpy(mask_camera.astype(bool)).to(device=device)
        valid = valid & mask_t

    gt_occ = (gt_t != free_id) & valid
    gt_free = (gt_t == free_id) & valid

    p = torch.softmax(logits_xyzc, dim=-1)
    p_occ = 1.0 - p[..., free_id]
    occ_logits = torch.cat(
        [logits_xyzc[..., :free_id], logits_xyzc[..., free_id + 1:]],
        dim=-1)
    s = torch.logsumexp(occ_logits, dim=-1) - logits_xyzc[..., free_id]
    pred_occ = p_occ >= 0.5

    inter = (pred_occ & gt_occ).sum().item()
    union = ((pred_occ | gt_occ) & valid).sum().item()

    occ_count = gt_occ.sum().item()
    free_count = gt_free.sum().item()
    occ_sum = s[gt_occ].sum().item() if occ_count > 0 else 0.0
    free_sum = s[gt_free].sum().item() if free_count > 0 else 0.0

    s_cpu = s.detach().float().cpu().numpy()
    del p, p_occ, occ_logits, s, pred_occ, gt_t, valid, gt_occ, gt_free
    return dict(
        inter=inter,
        union=union,
        occ_sum=occ_sum,
        occ_count=occ_count,
        free_sum=free_sum,
        free_count=free_count,
        s_cpu=s_cpu)


def fmt_float(value):
    if value is None or np.isnan(value):
        return 'nan'
    return f'{value:.6f}'


def write_report(path, args, captured_rows, selected_name, layout_note,
                 num_classes, free_id, class_names, class_name_source,
                 free_id_source, pc_range, grid_shape, voxel_size, metrics,
                 logits_pass, align_pass, gt_source_examples, mask_note):
    mmcv.mkdir_or_exist(os.path.dirname(path))
    status = 'PASS' if logits_pass and align_pass else 'FAIL / INVESTIGATE'
    lines = [
        '# RiskOcc Gate 1 Logit Alignment Probe',
        '',
        f'- config: `{args.config}`',
        f'- checkpoint: `{args.checkpoint}`',
        f'- max_frames: `{args.max_frames}`',
        f'- status: **{status}**',
        '',
        '## Captured tensors',
        '',
        '| name | shape | dtype |',
        '| --- | --- | --- |',
    ]
    for row in captured_rows:
        lines.append(
            f"| `{row['name']}` | `{tuple(row['shape'])}` | `{row['dtype']}` |")

    lines.extend([
        '',
        '## Selected logits',
        '',
        f'- name: `{selected_name}`',
        f'- layout: `{layout_note}`',
        f'- logits_gate: **{"PASS" if logits_pass else "FAIL"}**',
        '',
        '## Classes and geometry',
        '',
        f'- num_classes: `{num_classes}`',
        f'- free_id: `{free_id}` ({free_id_source})',
        f'- class_names_source: `{class_name_source}`',
        f'- class_names: `{class_names}`',
        f'- pc_range: `{pc_range}`',
        f'- grid_shape: `{grid_shape}`',
        f'- voxel_size: `{voxel_size}`',
        f'- valid_mask: `{mask_note}`',
        f'- gt_sources_sample: `{gt_source_examples[:5]}`',
        '',
        '## Alignment metrics',
        '',
        '| check | value |',
        '| --- | --- |',
        f"| mean s occupied near | `{fmt_float(metrics['mean_s_occ'])}` |",
        f"| mean s free near | `{fmt_float(metrics['mean_s_free'])}` |",
        f"| mean delta occupied-free | `{fmt_float(metrics['mean_s_delta'])}` |",
        f"| AUROC normal | `{fmt_float(metrics['auroc_normal'])}` |",
        f"| IoU p_occ>=0.5 | `{fmt_float(metrics['iou'])}` |",
        f"| AUROC swapXY | `{fmt_float(metrics['auroc_swapXY'])}` |",
        f"| AUROC flipX | `{fmt_float(metrics['auroc_flipX'])}` |",
        f"| AUROC flipY | `{fmt_float(metrics['auroc_flipY'])}` |",
        '',
        f'Alignment gate: **{"PASS" if align_pass else "FAIL / INVESTIGATE"}**',
        '',
    ])
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    args = parse_args()
    cfg, dataset, data_loader, model = build_probe_objects(args)

    num_classes = infer_num_classes(cfg)
    if num_classes is None:
        raise RuntimeError('FAIL: could not infer num_classes from cfg')
    dz = infer_dz(cfg)
    class_names, class_name_source = infer_class_names(cfg, num_classes)
    free_id, free_id_source = infer_free_id(class_names, num_classes)
    if free_id is None or not (0 <= free_id < num_classes):
        raise RuntimeError(
            f'FAIL: could not infer valid free_id for C={num_classes}')

    captured, handles = register_hooks(model)
    variant_scores = {
        mode: dict(labels=[], scores=[])
        for mode in ['normal', 'swapXY', 'flipX', 'flipY']
    }
    captured_rows_for_report = None
    selected_name = None
    layout_note = None
    pc_range = None
    voxel_size = None
    grid_shape = None
    near_mask = None
    gt_source_examples = []
    mask_note = 'mask_camera if present, else all valid non-255 voxels'
    totals = dict(inter=0, union=0, occ_sum=0.0, occ_count=0,
                  free_sum=0.0, free_count=0)

    frames = 0
    try:
        with torch.no_grad():
            for data in data_loader:
                if frames >= args.max_frames:
                    break
                captured.clear()
                _ = model(return_loss=False, rescale=True, **data)

                rows = []
                for name, output in captured.items():
                    scan_captured(name, output, rows, verbose=(frames == 0))

                selected_row, _dims, _fused = select_logits(
                    rows, num_classes, dz)
                assert selected_row is not None, (
                    'FAIL: no float logits captured. The head may only '
                    'return argmax labels; inspect occ_head.forward or the '
                    'loss/predict path for occ_pred logits.')

                logits, current_layout_note, class_dim = normalize_logits(
                    selected_row['tensor'], num_classes, dz)
                if logits.shape[-1] != num_classes:
                    raise RuntimeError(
                        f'normalized class dim {logits.shape[-1]} != '
                        f'num_classes {num_classes}')
                if not (0 <= free_id < logits.shape[-1]):
                    raise RuntimeError(
                        f'free_id {free_id} outside logits class dim '
                        f'{logits.shape[-1]}')

                if frames == 0:
                    captured_rows_for_report = [
                        {k: v for k, v in row.items() if k != 'tensor'}
                        for row in rows
                    ]
                    selected_name = selected_row['name']
                    layout_note = (
                        f'{current_layout_note}; selected_class_dim={class_dim}'
                    )
                    grid_shape = tuple(int(x) for x in logits.shape[1:4])
                    pc_range, voxel_size = infer_geometry(cfg, grid_shape)
                    near_mask = make_near_mask(grid_shape, pc_range, voxel_size)

                gt, mask_camera, gt_source = gt_from_data(data)
                if gt is None:
                    gt, mask_camera, gt_source = gt_from_dataset(dataset, frames)
                gt_source_examples.append(gt_source)
                if gt.shape != grid_shape:
                    raise RuntimeError(
                        f'GT shape {gt.shape} != logits grid {grid_shape}; '
                        'check logits layout/permute')

                batch_stats = compute_batch_stats(
                    logits, gt, mask_camera, near_mask, free_id)
                for key in ['inter', 'union', 'occ_sum', 'occ_count',
                            'free_sum', 'free_count']:
                    totals[key] += batch_stats[key]

                append_variant_scores(
                    variant_scores, batch_stats['s_cpu'], gt, mask_camera,
                    near_mask.numpy().astype(bool), free_id)

                logits_cpu = logits.detach().float().cpu()
                del logits_cpu, logits, batch_stats, rows
                captured.clear()
                torch.cuda.empty_cache()
                frames += 1
    finally:
        for handle in handles:
            handle.remove()

    if frames == 0:
        raise RuntimeError('FAIL: no frames were processed')

    metrics = {}
    metrics['mean_s_occ'] = (
        totals['occ_sum'] / totals['occ_count']
        if totals['occ_count'] else float('nan'))
    metrics['mean_s_free'] = (
        totals['free_sum'] / totals['free_count']
        if totals['free_count'] else float('nan'))
    metrics['mean_s_delta'] = metrics['mean_s_occ'] - metrics['mean_s_free']
    metrics['iou'] = (
        totals['inter'] / totals['union'] if totals['union'] else float('nan'))

    for mode in ['normal', 'swapXY', 'flipX', 'flipY']:
        labels = np.concatenate(variant_scores[mode]['labels'])
        scores = np.concatenate(variant_scores[mode]['scores'])
        metrics[f'auroc_{mode}'] = safe_auc(labels, scores)

    logits_pass = (
        selected_name is not None and grid_shape is not None and
        num_classes is not None and free_id is not None)
    transformed_aurocs = [
        metrics['auroc_swapXY'],
        metrics['auroc_flipX'],
        metrics['auroc_flipY'],
    ]
    align_pass = (
        metrics['mean_s_delta'] > 0 and
        metrics['auroc_normal'] > 0.5 and
        metrics['iou'] > 0.15 and
        all(metrics['auroc_normal'] >= value for value in transformed_aurocs
            if not np.isnan(value)))

    write_report(
        args.report, args, captured_rows_for_report or [], selected_name,
        layout_note, num_classes, free_id, class_names, class_name_source,
        free_id_source, pc_range, grid_shape, voxel_size, metrics, logits_pass,
        align_pass, gt_source_examples, mask_note)

    print(f'Processed frames: {frames}')
    print(f'Report written to: {args.report}')
    print(f'LOGITS_GATE: {"PASS" if logits_pass else "FAIL"}')
    print(f'ALIGNMENT_GATE: {"PASS" if align_pass else "FAIL / INVESTIGATE"}')
    print(
        'METRICS '
        f"mean_s_occ={fmt_float(metrics['mean_s_occ'])} "
        f"mean_s_free={fmt_float(metrics['mean_s_free'])} "
        f"auroc={fmt_float(metrics['auroc_normal'])} "
        f"iou={fmt_float(metrics['iou'])} "
        f"swapXY={fmt_float(metrics['auroc_swapXY'])} "
        f"flipX={fmt_float(metrics['auroc_flipX'])} "
        f"flipY={fmt_float(metrics['auroc_flipY'])}")


if __name__ == '__main__':
    main()
