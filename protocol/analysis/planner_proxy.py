#!/usr/bin/env python
"""Collect fixed-path occupancy proxy statistics on canonical report scenes.

The five predefined paths use a 20 m horizon, a 4.0 x 1.85 m footprint,
20 m gentle-turn and 8 m sharp-turn radii, and 0.4 m BEV cells. These are
explicit proxy assumptions, not closed-loop navigation guarantees.

For each frame/path, m_unsafe is the minimum slab-maximum occupied score on
eligible ground-truth obstacle cells; unsafe at threshold t means m_unsafe<t.
m_over is the maximum score on eligible free cells; over-blocking means
m_over>=t. Histograms support evaluation at tune-selected near-free FP
thresholds. Report unsafe and over-block rates together. The near-free FP
budget and path over-block rate are distinct quantities."""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'tools'))
import streaming_safety_evaluator as ev            # noqa: E402
import region_pr as rpr                            # noqa: E402  (build_static, region_masks)

NBINS = 200
L_HORIZON = 20.0
FOOT_L, FOOT_W = 4.0, 1.85
# curvatures (1/R): gentle R=20m, sharp R=8m; left +, right -
PRIMITIVES = {
    'straight':     0.0,
    'left_gentle':  1.0 / 20.0,
    'left_sharp':   1.0 / 8.0,
    'right_gentle': -1.0 / 20.0,
    'right_sharp':  -1.0 / 8.0,
}


def file_hash(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()[:12]


def primitive_poses(kappa, L=L_HORIZON, ds=0.1):
    poses = []
    s = 0.0
    while s <= L + 1e-9:
        if abs(kappa) < 1e-9:
            poses.append((s, 0.0, 0.0))
        else:
            yaw = kappa * s
            poses.append((np.sin(yaw) / kappa, (1 - np.cos(yaw)) / kappa, yaw))
        s += ds
    return poses


def sweep_mask(kappa, proto, grid):
    """Static BEV footprint-sweep mask [X,Y] for a primitive (ego at origin)."""
    X, Y, _ = grid
    pr = [float(v) for v in proto['pc_range']]
    vs = float(proto['voxel_size'])
    m = np.zeros((X, Y), bool)
    step = vs / 2.0
    dls = np.arange(-FOOT_L / 2, FOOT_L / 2 + 1e-6, step)
    dws = np.arange(-FOOT_W / 2, FOOT_W / 2 + 1e-6, step)
    for (x, y, yaw) in primitive_poses(kappa):
        c, s = np.cos(yaw), np.sin(yaw)
        for dl in dls:
            for dw in dws:
                px = x + dl * c - dw * s
                py = y + dl * s + dw * c
                ix = int((px - pr[0]) / vs)
                iy = int((py - pr[1]) / vs)
                if 0 <= ix < X and 0 <= iy < Y:
                    m[ix, iy] = True
    return m


def slab_z_mask(proto, Z):
    pr = [float(v) for v in proto['pc_range']]
    vs = float(proto['voxel_size'])
    zc = (np.arange(Z) + 0.5) * vs + pr[2]
    z_lo, z_hi = [float(v) for v in proto['z_plan']]
    return (zc >= z_lo) & (zc <= z_hi)


def accum_hist(hist_u, hist_o, cnt, fmask, maxp, gt_occ_bev, gt_free_bev):
    """Update unsafe/over histograms + counts for one (frame,primitive,region)."""
    occ_cells = fmask & gt_occ_bev
    free_cells = fmask & gt_free_bev
    cnt['n_paths'] += 1
    if occ_cells.any():
        m_unsafe = float(maxp[occ_cells].min())
        cnt['n_gtocc'] += 1
        b = min(int(m_unsafe * NBINS), NBINS - 1)
        hist_u[b] += 1
    if free_cells.any():
        m_over = float(maxp[free_cells].max())
        cnt['n_gtfree'] += 1
        b = min(int(m_over * NBINS), NBINS - 1)
        hist_o[b] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--protocol', default='configs/eval/safety_protocol.yaml')
    ap.add_argument('--split-json', default='protocol/split_scene.json')
    ap.add_argument('--regions', default='protocol/regions.yaml')
    ap.add_argument('--out', required=True)
    ap.add_argument('--subset', type=int, default=None,
                    help='cap report-split frames processed (selfcheck)')
    ap.add_argument('--force', choices=['none', 'allocc', 'allfree', 'gt'],
                    default='none', help='override p_occ for selfcheck sanity')
    ap.add_argument('--selfcheck', action='store_true')
    args = ap.parse_args()

    proto = ev.resolve_protocol(args.protocol)
    free_id = int(proto['free_id'])
    rcfg = yaml.safe_load(open(args.regions))['regions']
    split = json.load(open(args.split_json))
    report_set = set(split['report_scenes'])

    print(f'[planner_proxy:{args.tag}] frozen split={args.split_json} '
          f'(md5 {file_hash(args.split_json)}), regions={args.regions} '
          f'(md5 {file_hash(args.regions)}), force={args.force}')

    eargs = rpr.make_eval_args(args.config, args.checkpoint, args.protocol)
    cfg, dataset, data_loader, model = ev.build_eval_objects(eargs)
    captured, _h, _n = ev.register_occ_head_hook(model)
    num_classes = int(getattr(cfg.model.occ_head, 'num_classes', proto['num_classes']))
    dz = int(getattr(cfg.model.occ_head, 'Dz', 16))

    sweeps = None
    slab_z = None
    bin_id = group = None
    scopes = ['all', 'near_dynamic']
    hu = {sc: np.zeros(NBINS, np.int64) for sc in scopes}
    ho = {sc: np.zeros(NBINS, np.int64) for sc in scopes}
    cnt = {sc: {'n_paths': 0, 'n_gtocc': 0, 'n_gtfree': 0} for sc in scopes}
    frames = 0
    processed = 0

    with torch.no_grad():
        for data in data_loader:
            sc_tok = dataset.data_infos[frames].get('scene_token')
            in_report = sc_tok in report_set
            if not in_report:
                frames += 1
                continue
            if args.subset is not None and processed >= args.subset:
                frames += 1
                continue

            captured.clear()
            _ = model(return_loss=False, rescale=True, **data)
            logits = ev.normalize_logits(captured['logits'], num_classes, dz)[0]
            X, Y, Z = (int(v) for v in logits.shape[:3])

            if sweeps is None:
                grid = (X, Y, Z)
                bin_id, group = rpr.build_static(proto, grid)
                slab_z = slab_z_mask(proto, Z)
                sweeps = {k: sweep_mask(kappa, proto, grid)
                          for k, kappa in PRIMITIVES.items()}
                if args.selfcheck:
                    for k, mm in sweeps.items():
                        print(f'[selfcheck] sweep {k}: {int(mm.sum())} BEV cells')

            gt_np, mask_np, _ = ev.load_gt(dataset, frames)
            gt = gt_np.astype(np.int64)
            label_valid = (gt >= 0) & (gt < num_classes)
            mask = mask_np.astype(bool) if mask_np is not None else label_valid
            valid = mask & label_valid
            gt_occ = (gt != free_id) & valid
            gt_free = (gt == free_id) & valid

            if args.force == 'allocc':
                p_occ = np.ones((X, Y, Z), np.float32)
            elif args.force == 'allfree':
                p_occ = np.zeros((X, Y, Z), np.float32)
            elif args.force == 'gt':
                p_occ = gt_occ.astype(np.float32)
            else:
                prob_free = torch.softmax(logits, dim=-1)[..., free_id]
                p_occ = (1.0 - prob_free).detach().cpu().numpy()

            group_3d = group[np.clip(gt, 0, num_classes - 1)]
            nd_region = rpr.region_masks(gt_occ, gt_free, bin_id, group_3d,
                                         rcfg)['near_dynamic']

            slab = slab_z[None, None, :]
            valid_slab = valid & slab
            maxp = np.where(valid_slab, p_occ, -1.0).max(axis=2)      # [X,Y]
            valid_bev = valid_slab.any(axis=2)
            gt_occ_bev = (gt_occ & slab).any(axis=2)
            gt_free_bev = valid_bev & ~gt_occ_bev
            nd_bev = (nd_region & slab).any(axis=2)

            for _prim, fmask in sweeps.items():
                accum_hist(hu['all'], ho['all'], cnt['all'], fmask, maxp,
                           gt_occ_bev, gt_free_bev)
                accum_hist(hu['near_dynamic'], ho['near_dynamic'],
                           cnt['near_dynamic'], fmask & nd_bev, maxp,
                           gt_occ_bev, gt_free_bev)

            frames += 1
            processed += 1
            if processed % 400 == 0:
                print(f'[planner_proxy:{args.tag}] {processed} report frames',
                      flush=True)

    out = {'tag': args.tag, 'checkpoint': args.checkpoint, 'config': args.config,
           'force': args.force, 'nbins': NBINS, 'report_frames': processed,
           'primitives': list(PRIMITIVES), 'horizon_m': L_HORIZON,
           'footprint_m': [FOOT_L, FOOT_W],
           'split_md5': file_hash(args.split_json),
           'regions_md5': file_hash(args.regions),
           'scopes': {}}
    for sc in scopes:
        out['scopes'][sc] = {
            'unsafe_hist': hu[sc].tolist(), 'over_hist': ho[sc].tolist(),
            'n_paths': cnt[sc]['n_paths'], 'n_gtocc': cnt[sc]['n_gtocc'],
            'n_gtfree': cnt[sc]['n_gtfree']}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, 'w'))
    print(f'[planner_proxy:{args.tag}] wrote {args.out} report_frames={processed}')

    if args.selfcheck:
        def rate_unsafe(sc, t):
            k = int(t * NBINS)
            n = cnt[sc]['n_paths']
            return float(hu[sc][:k].sum() / n) if n else float('nan')

        def rate_over(sc, t):
            k = int(t * NBINS)
            n = cnt[sc]['n_paths']
            return float(ho[sc][k:].sum() / n) if n else float('nan')

        u = rate_unsafe('all', 0.5)
        o = rate_over('all', 0.5)
        print(f'[selfcheck] force={args.force} @t=0.5  unsafe_rate={u:.4f} '
              f'over_block_rate={o:.4f}  (all scope, n_paths={cnt["all"]["n_paths"]})')
        print(f'[selfcheck] in[0,1]: unsafe={0 <= u <= 1} over={0 <= o <= 1}')


if __name__ == '__main__':
    main()
