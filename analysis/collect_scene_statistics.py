#!/usr/bin/env python
"""Collect per-scene sufficient statistics for one checkpoint.

Uses the streaming evaluator and canonical tune/report split. Records
1,200-bin regional and near-field score histograms, report-set fixed-path
statistics, degradation counters and full-validation native FN counts.
Optional predicted-group histograms and risk-field summaries are diagnostics.
Threshold selection uses 200 bins; manuscript regional AUPRC uses 1,200 bins.
Output: <out>.npz arrays and <out>.json metadata."""
import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(REPO, 'tools'))
sys.path.insert(0, os.path.join(REPO, 'protocol', 'analysis'))
import streaming_safety_evaluator as ev            # noqa: E402
import region_pr as rpr                            # noqa: E402
import planner_proxy as pp                         # noqa: E402

NB = 1200                       # 6 x frozen 200-bin grid
NB0 = 200                       # frozen grid
REGIONS = ['far', 'near_dynamic', 'boundary']
SPLIT_ID = {'tune': 0, 'report': 1, None: 2}
PRIMS = list(pp.PRIMITIVES)

# risk-field constants mirroring BEVOCCHead2D_RiskOcc (frozen E6 recipe)
RISK_DMAX = 40.0
RC_VULNERABLE, RC_DYNAMIC, RC_STATIC, RC_FREE = 1.0, 0.6, 0.3, 0.0
LAMBDA_D = LAMBDA_C = 0.5


def risk_class_values(proto):
    vul = set(proto['_vulnerable_ids'])
    dyn = set(proto['_dynamic_ids'])
    nc = int(proto['num_classes'])
    free_id = int(proto['free_id'])
    rc = np.full(nc, RC_STATIC, np.float64)
    for c in dyn:
        rc[c] = RC_DYNAMIC
    for c in vul:
        rc[c] = RC_VULNERABLE
    rc[free_id] = RC_FREE
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--protocol', default='configs/eval/safety_protocol.yaml')
    ap.add_argument('--split-json', default='protocol/split_scene.json')
    ap.add_argument('--regions', default='protocol/regions.yaml')
    ap.add_argument('--out', required=True, help='output stem (.npz/.json added)')
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--pred-group', action='store_true')
    ap.add_argument('--risk-field', action='store_true')
    args = ap.parse_args()

    proto = ev.resolve_protocol(args.protocol)
    free_id = int(proto['free_id'])
    rcfg = yaml.safe_load(open(args.regions))['regions']
    split = json.load(open(args.split_json))
    tune_set = set(split['tune_scenes'])
    report_set = set(split['report_scenes'])

    eargs = rpr.make_eval_args(args.config, args.checkpoint, args.protocol)
    cfg, dataset, data_loader, model = ev.build_eval_objects(eargs)
    captured, _h, _n = ev.register_occ_head_hook(model)
    num_classes = int(getattr(cfg.model.occ_head, 'num_classes',
                              proto['num_classes']))
    dz = int(getattr(cfg.model.occ_head, 'Dz', 16))

    scene_ids = {}
    scene_tokens = []
    scene_split = []

    # growable per-scene stores (lists indexed by scene id)
    reg_pos, reg_neg = [], []          # each: [3reg,3bin,NB] int64
    nav_free, nav_hr = [], []          # each: [NB]
    deg_tot, deg_lt2, deg_lt3 = [], [], []   # each: [3reg]
    # planner path records
    pl_scene, pl_prim = [], []
    pl_mu_all, pl_mo_all, pl_mu_nd, pl_mo_nd = [], [], [], []

    pg_pos = np.zeros((2, 3, 4, NB), np.int64)   # split x bin x predgroup x NB
    pg_neg = np.zeros((2, 3, 4, NB), np.int64)

    fn_num = np.zeros((3, 3), np.int64)
    fn_den = np.zeros((3, 3), np.int64)

    rf = {'sum_valid': 0.0, 'n_valid': 0, 'sum_occ': 0.0, 'n_occ': 0}
    rd_3d = None

    bin_id = group = None
    sweeps = slab_z = None
    frames = 0

    def scene_idx(tok):
        if tok not in scene_ids:
            scene_ids[tok] = len(scene_tokens)
            scene_tokens.append(tok)
            scene_split.append(SPLIT_ID['tune'] if tok in tune_set else
                               SPLIT_ID['report'] if tok in report_set else 2)
            reg_pos.append(np.zeros((3, 3, NB), np.int64))
            reg_neg.append(np.zeros((3, 3, NB), np.int64))
            nav_free.append(np.zeros(NB, np.int64))
            nav_hr.append(np.zeros(NB, np.int64))
            deg_tot.append(np.zeros(3, np.int64))
            deg_lt2.append(np.zeros(3, np.int64))
            deg_lt3.append(np.zeros(3, np.int64))
        return scene_ids[tok]

    with torch.no_grad():
        for data in data_loader:
            if args.max_frames is not None and frames >= args.max_frames:
                break
            tok = dataset.data_infos[frames].get('scene_token')
            si = scene_idx(tok)
            ssplit = scene_split[si]

            captured.clear()
            _ = model(return_loss=False, rescale=True, **data)
            logits = ev.normalize_logits(captured['logits'], num_classes, dz)[0]
            X, Y, Z = (int(v) for v in logits.shape[:3])

            if bin_id is None:
                grid = (X, Y, Z)
                bin_id, group = rpr.build_static(proto, grid)
                slab_z = pp.slab_z_mask(proto, Z)
                sweeps = {k: pp.sweep_mask(kappa, proto, grid)
                          for k, kappa in pp.PRIMITIVES.items()}
                if args.risk_field:
                    pr = [float(v) for v in proto['pc_range']]
                    vs = float(proto['voxel_size'])
                    xs = (np.arange(X) + 0.5) * vs + pr[0]
                    ys = (np.arange(Y) + 0.5) * vs + pr[1]
                    xg, yg = np.meshgrid(xs, ys, indexing='ij')
                    bev = np.sqrt(xg ** 2 + yg ** 2)
                    rd = np.clip((RISK_DMAX - bev) / RISK_DMAX, 0.0, 1.0)
                    rd_3d = np.repeat(rd[:, :, None], Z, axis=2)
                    rc_lut = risk_class_values(proto)

            gt_np, mask_np, _ = ev.load_gt(dataset, frames)
            gt = gt_np.astype(np.int64)
            label_valid = (gt >= 0) & (gt < num_classes)
            mask = mask_np.astype(bool) if mask_np is not None else label_valid
            valid = mask & label_valid
            gt_occ = (gt != free_id) & valid
            gt_free = (gt == free_id) & valid

            prob = torch.softmax(logits, dim=-1)
            p_occ = (1.0 - prob[..., free_id]).detach().cpu().numpy()
            pred_free = (logits.argmax(dim=-1).detach().cpu().numpy() == free_id)
            group_3d = group[np.clip(gt, 0, num_classes - 1)]

            q = np.minimum((p_occ * NB).astype(np.int64), NB - 1)

            # fn_matrix (all frames)
            for b in range(3):
                for g in range(3):
                    cell = gt_occ & (bin_id == b) & (group_3d == g)
                    d = int(cell.sum())
                    if d:
                        fn_den[b, g] += d
                        fn_num[b, g] += int((cell & pred_free).sum())

            if args.risk_field:
                rc_3d = rc_lut[np.clip(gt, 0, num_classes - 1)]
                R = LAMBDA_D * rd_3d + LAMBDA_C * rc_3d
                rf['sum_valid'] += float(R[valid].sum())
                rf['n_valid'] += int(valid.sum())
                rf['sum_occ'] += float(R[gt_occ].sum())
                rf['n_occ'] += int(gt_occ.sum())

            if ssplit != 2:
                masks = rpr.region_masks(gt_occ, gt_free, bin_id, group_3d, rcfg)
                for ri, r in enumerate(REGIONS):
                    rm = masks[r]
                    rm_pos = rm & gt_occ
                    rm_neg = rm & gt_free
                    # combined code: bin*2 + is_pos over region voxels
                    sel = rm_pos | rm_neg
                    if sel.any():
                        code = (bin_id[sel] * 2 + rm_pos[sel]) * NB + q[sel]
                        h = np.bincount(code, minlength=6 * NB).reshape(3, 2, NB)
                        reg_neg[si][ri] += h[:, 0, :]
                        reg_pos[si][ri] += h[:, 1, :]
                    n_pos = int(rm_pos.sum())
                    if n_pos:
                        deg_tot[si][ri] += n_pos
                        deg_lt2[si][ri] += int((rm_pos & (p_occ < 0.01)).sum())
                        deg_lt3[si][ri] += int((rm_pos & (p_occ < 0.001)).sum())

                near_b = (bin_id == 0)
                nfm = near_b & gt_free
                hrm = near_b & ((group_3d == 0) | (group_3d == 1)) & gt_occ
                if nfm.any():
                    nav_free[si] += np.bincount(q[nfm], minlength=NB)
                if hrm.any():
                    nav_hr[si] += np.bincount(q[hrm], minlength=NB)

                if args.pred_group:
                    # predicted class group from argmax over NON-free classes
                    occ_logits = logits.detach().clone()
                    occ_logits[..., free_id] = -1e9
                    pred_cls = occ_logits.argmax(dim=-1).cpu().numpy()
                    pgrp = group[np.clip(pred_cls, 0, num_classes - 1)]
                    for is_pos, m0 in ((1, gt_occ), (0, gt_free)):
                        code = ((bin_id[m0] * 4 + pgrp[m0]) * NB + q[m0])
                        h = np.bincount(code, minlength=12 * NB).reshape(3, 4, NB)
                        if is_pos:
                            pg_pos[ssplit] += h
                        else:
                            pg_neg[ssplit] += h

                # planner (report split only, frozen step-6 semantics)
                if ssplit == 1:
                    nd_region = masks['near_dynamic']
                    slab = slab_z[None, None, :]
                    valid_slab = valid & slab
                    maxp = np.where(valid_slab, p_occ, -1.0).max(axis=2)
                    valid_bev = valid_slab.any(axis=2)
                    gt_occ_bev = (gt_occ & slab).any(axis=2)
                    gt_free_bev = valid_bev & ~gt_occ_bev
                    nd_bev = (nd_region & slab).any(axis=2)
                    for pi, prim in enumerate(PRIMS):
                        fmask = sweeps[prim]
                        for tag_nd, fm in ((0, fmask), (1, fmask & nd_bev)):
                            occ_cells = fm & gt_occ_bev
                            free_cells = fm & gt_free_bev
                            mu = float(maxp[occ_cells].min()) if occ_cells.any() else np.nan
                            mo = float(maxp[free_cells].max()) if free_cells.any() else np.nan
                            if tag_nd == 0:
                                pl_scene.append(si)
                                pl_prim.append(pi)
                                pl_mu_all.append(mu)
                                pl_mo_all.append(mo)
                            else:
                                pl_mu_nd.append(mu)
                                pl_mo_nd.append(mo)

            frames += 1
            if frames % 500 == 0:
                print(f'[g0_stats:{args.tag}] {frames} frames', flush=True)

    S = len(scene_tokens)
    out_npz = args.out + '.npz'
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(
        out_npz,
        scene_tokens=np.array(scene_tokens),
        scene_split=np.array(scene_split, np.int64),
        region_pos=np.stack(reg_pos) if S else np.zeros((0, 3, 3, NB), np.int64),
        region_neg=np.stack(reg_neg) if S else np.zeros((0, 3, 3, NB), np.int64),
        nav_free=np.stack(nav_free) if S else np.zeros((0, NB), np.int64),
        nav_hr=np.stack(nav_hr) if S else np.zeros((0, NB), np.int64),
        deg_tot=np.stack(deg_tot) if S else np.zeros((0, 3), np.int64),
        deg_lt_0p01=np.stack(deg_lt2) if S else np.zeros((0, 3), np.int64),
        deg_lt_0p001=np.stack(deg_lt3) if S else np.zeros((0, 3), np.int64),
        pl_scene=np.array(pl_scene, np.int64),
        pl_prim=np.array(pl_prim, np.int64),
        pl_m_unsafe_all=np.array(pl_mu_all, np.float64),
        pl_m_over_all=np.array(pl_mo_all, np.float64),
        pl_m_unsafe_nd=np.array(pl_mu_nd, np.float64),
        pl_m_over_nd=np.array(pl_mo_nd, np.float64),
        pg_pos=pg_pos, pg_neg=pg_neg,
        fn_num=fn_num, fn_den=fn_den)

    meta = {'tag': args.tag, 'checkpoint': args.checkpoint,
            'config': args.config, 'frames': frames, 'nbins': NB,
            'frozen_grid': NB0, 'regions': REGIONS, 'primitives': PRIMS,
            'n_scenes': S, 'pred_group': bool(args.pred_group)}
    if args.risk_field:
        meta['risk_field'] = {
            'E_R_valid': rf['sum_valid'] / max(rf['n_valid'], 1),
            'E_R_occupied': rf['sum_occ'] / max(rf['n_occ'], 1),
            'n_valid': rf['n_valid'], 'n_occ': rf['n_occ'],
            'formula': '0.5*Rd(linear,Dmax=40m) + 0.5*Rc(vul=1.0,dyn=0.6,static=0.3,free=0)'}
    json.dump(meta, open(args.out + '.json', 'w'), indent=2)
    print(f'[g0_stats:{args.tag}] wrote {out_npz} frames={frames} scenes={S}')


if __name__ == '__main__':
    main()
