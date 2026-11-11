#!/usr/bin/env python
"""Collect 200-bin regional occupied/free diagnostic histograms.

Uses the streaming occupancy evaluator, camera-valid mask and canonical scene
split. Thresholds are selected on tune scenes; reporting uses disjoint scenes.
The final manuscript's AUPRC uses 1,200-bin histograms collected by
analysis/collect_scene_statistics.py. See analysis/verify_release.py for
aggregation of those archived histograms."""
import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'tools'))
import streaming_safety_evaluator as ev          # noqa: E402
import scipy.ndimage as ndi                       # noqa: E402

NBINS = 200
STRUCT1 = ndi.generate_binary_structure(3, 1)     # 6-connectivity
STRUCT3 = ndi.generate_binary_structure(3, 3)     # 26-connectivity
BIN_NAMES = ['near', 'mid', 'far']
GROUP_NAMES = ['dynamic', 'vulnerable', 'static']  # group id 0,1,2 (see build_static)
REGIONS = ['far', 'near_dynamic', 'boundary']
SPLITS = ['tune', 'report']


def make_eval_args(config, checkpoint, protocol):
    return argparse.Namespace(
        config=config, checkpoint=checkpoint, protocol=protocol,
        baseline_miou=None, miou_tolerance=0.5,
        report='/tmp/_rpr_report.md', metrics_json='/tmp/_rpr_metrics.json',
        max_frames=None, gpu_id=0, seed=0, deterministic=True,
        no_aavt=False, cfg_options=None)


def build_static(proto, grid):
    """Return (bin_id[X,Y,Z] np int, group_lookup[num_classes] np int).

    Mirrors StreamingAccumulator._precompute_static: bev_dist = sqrt(xc^2+yc^2)
    on voxel centres; bins near[lo,hi)=0 mid=1 far=2. group: dynamic(non-VRU)=0,
    vulnerable=1 (overrides dynamic), static=2, free/none=3.
    """
    X, Y, Z = grid
    pr = [float(v) for v in proto['pc_range']]
    vs = float(proto['voxel_size'])
    xs = (np.arange(X) + 0.5) * vs + pr[0]
    ys = (np.arange(Y) + 0.5) * vs + pr[1]
    xg, yg = np.meshgrid(xs, ys, indexing='ij')
    bev = np.sqrt(xg ** 2 + yg ** 2)                        # [X,Y]
    bev3 = np.repeat(bev[:, :, None], Z, axis=2)            # [X,Y,Z]
    nlo, nhi = proto['distance_bins']['near']
    mlo, mhi = proto['distance_bins']['mid']
    bin_id = np.full((X, Y, Z), 2, dtype=np.int64)
    bin_id[(bev3 >= nlo) & (bev3 < nhi)] = 0
    bin_id[(bev3 >= mlo) & (bev3 < mhi)] = 1
    nc = int(proto['num_classes'])
    group = np.full(nc, 3, dtype=np.int64)
    for c in proto['_dynamic_ids']:
        group[c] = 0
    for c in proto['_vulnerable_ids']:
        group[c] = 1
    for c in proto['_static_ids']:
        group[c] = 2
    return bin_id, group


def region_masks(gt_occ, gt_free, bin_id, group_3d, rcfg):
    """Boolean region masks (numpy [X,Y,Z]); see protocol/regions.yaml."""
    far = (bin_id == 2)

    nd_dil = int(rcfg['near_dynamic']['dilation_voxels'])
    dyn_union_occ = gt_occ & ((group_3d == 0) | (group_3d == 1))
    near_dynamic = (bin_id == 0) & ndi.binary_dilation(
        dyn_union_occ, structure=STRUCT3, iterations=nd_dil)

    internal = gt_occ & ~ndi.binary_erosion(gt_occ, STRUCT1, border_value=0)
    external = gt_free & ndi.binary_dilation(gt_occ, STRUCT1, border_value=0)
    boundary = internal | external
    return {'far': far, 'near_dynamic': near_dynamic, 'boundary': boundary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--protocol', default='configs/eval/safety_protocol.yaml')
    ap.add_argument('--split-json', default='protocol/split_scene.json')
    ap.add_argument('--regions', default='protocol/regions.yaml')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-frames', type=int, default=None,
                    help='cap frames (for pipeline smoke only; fn_matrix needs full val)')
    args = ap.parse_args()

    proto = ev.resolve_protocol(args.protocol)
    free_id = int(proto['free_id'])
    rcfg = yaml.safe_load(open(args.regions))['regions']
    split = json.load(open(args.split_json))
    tune_set, report_set = set(split['tune_scenes']), set(split['report_scenes'])

    eargs = make_eval_args(args.config, args.checkpoint, args.protocol)
    cfg, dataset, data_loader, model = ev.build_eval_objects(eargs)
    captured, _handle, _name = ev.register_occ_head_hook(model)
    num_classes = int(getattr(cfg.model.occ_head, 'num_classes',
                              proto['num_classes']))
    dz = int(getattr(cfg.model.occ_head, 'Dz', 16))

    # per region: [distance-bin(0=near,1=mid,2=far), prob-bin] so E0-spatial
    # per-distance-bin thresholds are computable; whole-region PR = sum over axis0.
    pos_hist = {r: {s: np.zeros((3, NBINS), np.int64) for s in SPLITS} for r in REGIONS}
    neg_hist = {r: {s: np.zeros((3, NBINS), np.int64) for s in SPLITS} for r in REGIONS}
    # nav-blocking analysis (near bin only): near-field free = blocking budget
    # denominator; near-field high-risk occupied (dynamic_union) = recall target.
    nav_free = {s: np.zeros(NBINS, np.int64) for s in SPLITS}
    nav_hr = {s: np.zeros(NBINS, np.int64) for s in SPLITS}
    fn_num = np.zeros((3, 3), np.int64)
    fn_den = np.zeros((3, 3), np.int64)
    bin_id = group = None
    frames = 0

    with torch.no_grad():
        for data in data_loader:
            if args.max_frames is not None and frames >= args.max_frames:
                break
            sc = dataset.data_infos[frames].get('scene_token')
            fsplit = 'tune' if sc in tune_set else (
                'report' if sc in report_set else None)

            captured.clear()
            _ = model(return_loss=False, rescale=True, **data)
            logits = ev.normalize_logits(captured['logits'], num_classes, dz)[0]

            if bin_id is None:
                grid = tuple(int(v) for v in logits.shape[:3])
                bin_id, group = build_static(proto, grid)

            gt_np, mask_np, _ = ev.load_gt(dataset, frames)
            gt = gt_np.astype(np.int64)
            label_valid = (gt >= 0) & (gt < num_classes)
            mask = mask_np.astype(bool) if mask_np is not None else label_valid
            valid = mask & label_valid
            gt_occ = (gt != free_id) & valid
            gt_free = (gt == free_id) & valid

            prob_free = torch.softmax(logits, dim=-1)[..., free_id]
            p_occ = (1.0 - prob_free).detach().cpu().numpy()
            pred_free = (logits.argmax(dim=-1).detach().cpu().numpy() == free_id)
            group_3d = group[np.clip(gt, 0, num_classes - 1)]

            # argmax FN matrix (ALL frames) -- selfcheck answer key
            for b in range(3):
                for g in range(3):
                    cell = gt_occ & (bin_id == b) & (group_3d == g)
                    d = int(cell.sum())
                    if d == 0:
                        continue
                    fn_den[b, g] += d
                    fn_num[b, g] += int((cell & pred_free).sum())

            # region PR histograms (per split, per distance-bin)
            if fsplit is not None:
                masks = region_masks(gt_occ, gt_free, bin_id, group_3d, rcfg)
                for r in REGIONS:
                    rm = masks[r]
                    rm_pos = rm & gt_occ
                    rm_neg = rm & gt_free
                    for b in range(3):
                        binb = (bin_id == b)
                        pv = p_occ[rm_pos & binb]
                        nv = p_occ[rm_neg & binb]
                        if pv.size:
                            pos_hist[r][fsplit][b] += np.histogram(
                                pv, bins=NBINS, range=(0.0, 1.0))[0]
                        if nv.size:
                            neg_hist[r][fsplit][b] += np.histogram(
                                nv, bins=NBINS, range=(0.0, 1.0))[0]
                # nav-blocking (near bin only)
                near_b = (bin_id == 0)
                nf = p_occ[near_b & gt_free]
                ho = p_occ[near_b & ((group_3d == 0) | (group_3d == 1)) & gt_occ]
                if nf.size:
                    nav_free[fsplit] += np.histogram(nf, bins=NBINS, range=(0.0, 1.0))[0]
                if ho.size:
                    nav_hr[fsplit] += np.histogram(ho, bins=NBINS, range=(0.0, 1.0))[0]

            frames += 1
            if frames % 500 == 0:
                print(f'[region_pr:{args.tag}] {frames} frames', flush=True)

    # fn_matrix in the evaluator's named-cell layout
    fnm = {'num': {}, 'den': {}, 'rate': {}}
    for b in range(3):
        for g in range(3):
            k = f'{BIN_NAMES[b]}__{GROUP_NAMES[g]}'
            fnm['num'][k] = int(fn_num[b, g])
            fnm['den'][k] = int(fn_den[b, g])
            fnm['rate'][k] = (float(fn_num[b, g] / fn_den[b, g])
                              if fn_den[b, g] > 0 else None)

    regions_out = {}
    for r in REGIONS:
        regions_out[r] = {}
        for s in SPLITS:
            regions_out[r][s] = {
                'pos_hist_by_bin': pos_hist[r][s].tolist(),   # [3][NBINS]
                'neg_hist_by_bin': neg_hist[r][s].tolist(),
                'n_pos': int(pos_hist[r][s].sum()),
                'n_neg': int(neg_hist[r][s].sum())}
    nav_out = {}
    for s in SPLITS:
        nav_out[s] = {
            'free_hist': nav_free[s].tolist(),
            'hr_occ_hist': nav_hr[s].tolist(),
            'n_free': int(nav_free[s].sum()),
            'n_hr_occ': int(nav_hr[s].sum())}

    out = {'tag': args.tag, 'checkpoint': args.checkpoint, 'config': args.config,
           'frames': frames, 'nbins': NBINS, 'p_occ_range': [0.0, 1.0],
           'bin_names': BIN_NAMES,
           'max_frames': args.max_frames, 'fn_matrix': fnm,
           'regions': regions_out, 'nav': nav_out}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, 'w'))
    print(f'[region_pr:{args.tag}] wrote {args.out}  frames={frames}')

    # red-line assertion (only meaningful on a full/representative pass)
    problems = []
    for r in REGIONS:
        for s in SPLITS:
            d = regions_out[r][s]
            if d['n_pos'] == 0 or d['n_neg'] == 0:
                problems.append(f'{r}/{s}: n_pos={d["n_pos"]} n_neg={d["n_neg"]}')
    if problems:
        print('[region_pr] WARNING zero pos/neg (ok only if --max-frames tiny): '
              + '; '.join(problems))


if __name__ == '__main__':
    main()
