#!/usr/bin/env python
"""Aggregate the historical 200-bin regional diagnostic outputs.

Select thresholds on tune scenes and report regional discrimination and recall
on disjoint report scenes. Inherited fields containing "blocking" refer to
near-field free-voxel FP rates, not path over-blocking. The final manuscript's
1,200-bin AUPRC is recomputed by analysis/verify_release.py."""
import argparse
import json
import os

import numpy as np

REGIONS = ['far', 'near_dynamic', 'boundary']
ALPHAS = [0.05, 0.10, 0.20]      # region FP-rate budgets
BETAS = [0.05, 0.10]             # nav-blocking budgets
A = 'reports/formal/analysis'


def load(tag):
    return json.load(open(f'{A}/region_pr_{tag}.json'))


def region_hist(d, region, split):
    """Return (pos[NBINS], neg[NBINS]) summed over distance bins."""
    r = d['regions'][region][split]
    pos = np.array(r['pos_hist_by_bin'], float).sum(axis=0)
    neg = np.array(r['neg_hist_by_bin'], float).sum(axis=0)
    return pos, neg


def region_hist_by_bin(d, region, split):
    r = d['regions'][region][split]
    return np.array(r['pos_hist_by_bin'], float), np.array(r['neg_hist_by_bin'], float)


def auprc(pos, neg):
    tp = np.cumsum(pos[::-1]); fp = np.cumsum(neg[::-1])
    P = pos.sum()
    if P == 0:
        return float('nan')
    rec = tp / P
    prec = tp / np.maximum(tp + fp, 1e-12)
    rec = np.concatenate([[0.0], rec]); prec = np.concatenate([[1.0], prec])
    return float((np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(prec, rec))


def k_at_fp(neg, alpha):
    """Smallest threshold-bin k with FP-rate = sum(neg[k:])/sum(neg) <= alpha
    (highest recall while staying within the FP budget). Returns k in [0,NBINS]."""
    tot = neg.sum()
    if tot == 0:
        return None
    cum_hi = np.cumsum(neg[::-1])[::-1]         # cum_hi[k] = sum(neg[k:])
    fpr = cum_hi / tot
    ks = np.where(fpr <= alpha)[0]
    return int(ks[0]) if ks.size else len(neg)


def recall_at_k(pos, k):
    tot = pos.sum()
    if tot == 0:
        return float('nan')
    return float(pos[k:].sum() / tot) if k < len(pos) else 0.0


def fp_at_k(neg, k):
    tot = neg.sum()
    if tot == 0:
        return float('nan')
    return float(neg[k:].sum() / tot) if k < len(neg) else 0.0


def fn_at_fixed_fp(d, region, alpha):
    """Threshold on tune (region FP<=alpha), recall on report. Whole region."""
    pos_t, neg_t = region_hist(d, region, 'tune')
    pos_r, neg_r = region_hist(d, region, 'report')
    k = k_at_fp(neg_t, alpha)
    if k is None:
        return None
    return {'recall': recall_at_k(pos_r, k),
            'report_fp': fp_at_k(neg_r, k)}


def e0_fixed_fp(d_e1, region, alpha):
    """E0-spatial: per-distance-bin threshold (tune, each bin FP<=alpha), region
    recall on report. Uses E1 vanilla data. Single-bin region -> equals E1."""
    pos_t, neg_t = region_hist_by_bin(d_e1, region, 'tune')     # [3,NBINS]
    pos_r, neg_r = region_hist_by_bin(d_e1, region, 'report')
    tp_r = 0.0; P_r = 0.0
    for b in range(3):
        if neg_t[b].sum() == 0 and pos_t[b].sum() == 0:
            continue
        k = k_at_fp(neg_t[b], alpha)
        if k is None:                       # no negatives in this bin -> pass all
            k = 0
        tp_r += pos_r[b][k:].sum()
        P_r += pos_r[b].sum()
    return float(tp_r / P_r) if P_r > 0 else float('nan')


def recall_at_blocking(d, beta):
    """Threshold on tune so near free FP (blocking)<=beta; report recall of
    near high-risk (dynamic_union) occupied."""
    nf_t = np.array(d['nav']['tune']['free_hist'], float)
    hr_t = np.array(d['nav']['tune']['hr_occ_hist'], float)  # noqa: F841
    nf_r = np.array(d['nav']['report']['free_hist'], float)  # noqa: F841
    hr_r = np.array(d['nav']['report']['hr_occ_hist'], float)
    k = k_at_fp(nf_t, beta)
    if k is None:
        return None
    return {'recall': recall_at_k(hr_r, k),
            'report_blocking_fp': fp_at_k(np.array(d['nav']['report']['free_hist'], float), k)}


def agg(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return None
    return {'mean': float(np.mean(v)), 'min': float(np.min(v)),
            'max': float(np.max(v)), 'n': len(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--e6', nargs='+', default=['E6_seed0', 'E6_seed1', 'E6_seed2'])
    ap.add_argument('--cetv', nargs='+', default=['CETVERSKY_seed0', 'CETVERSKY_seed1'])
    ap.add_argument('--e1', nargs='+', default=['E1_seed0', 'E1_seed1', 'E1_seed2'])
    ap.add_argument('--out', default=f'{A}/step5_metrics.json')
    args = ap.parse_args()

    methods = {'E6': args.e6, 'CE+Tv': args.cetv, 'E1': args.e1}
    D = {m: [load(t) for t in tags] for m, tags in methods.items()}

    out = {'alphas': ALPHAS, 'betas': BETAS, 'auprc': {}, 'fn_at_fp': {},
           'recall_at_blocking': {}, 'verdict': {}}

    # AUPRC (report)
    for r in REGIONS:
        out['auprc'][r] = {m: agg([auprc(*region_hist(d, r, 'report')) for d in D[m]])
                           for m in methods}

    # FN@fixedFP (recall) per region per alpha; E0-spatial from E1 data
    for r in REGIONS:
        out['fn_at_fp'][r] = {}
        for a in ALPHAS:
            row = {}
            for m in methods:
                row[m] = agg([(fn_at_fixed_fp(d, r, a) or {}).get('recall') for d in D[m]])
            row['E0-spatial'] = agg([e0_fixed_fp(d, r, a) for d in D['E1']])
            out['fn_at_fp'][r][f'{a:.2f}'] = row

    # Recall@fixed blocking per beta (E0-spatial == E1 here: near single-bin)
    for b in BETAS:
        row = {}
        for m in methods:
            row[m] = agg([(recall_at_blocking(d, b) or {}).get('recall') for d in D[m]])
        row['E0-spatial'] = row['E1']
        out['recall_at_blocking'][f'{b:.2f}'] = row

    # Verdict: after FP control, is E6 >= CE+Tv AND >= E0-spatial? (use alpha=0.10)
    a_key = '0.10'
    for r in REGIONS:
        e6 = out['fn_at_fp'][r][a_key]['E6']
        ce = out['fn_at_fp'][r][a_key]['CE+Tv']
        e0 = out['fn_at_fp'][r][a_key]['E0-spatial']
        if not (e6 and ce and e0):
            out['verdict'][r] = 'INSUFFICIENT'
            continue
        d_ce = e6['mean'] - ce['mean']
        d_e0 = e6['mean'] - e0['mean']
        ok = (d_ce >= 0) and (d_e0 >= 0)
        out['verdict'][r] = {
            'recall@FP0.10_E6_mean': e6['mean'], 'CE+Tv_mean': ce['mean'],
            'E0_mean': e0['mean'], 'E6-CE+Tv': d_ce, 'E6-E0': d_e0,
            'E6>=both': bool(ok)}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, 'w'), indent=2)

    # ---- pretty print ----
    def fmt(a):
        return f'{a["mean"]:.4f}[{a["min"]:.4f}-{a["max"]:.4f}]' if a else 'n/a'

    print('\n==== AUPRC (report split) ====')
    print(f'{"region":13}{"E6":>22}{"CE+Tv":>22}{"E1":>22}')
    for r in REGIONS:
        a = out['auprc'][r]
        print(f'{r:13}{fmt(a["E6"]):>22}{fmt(a["CE+Tv"]):>22}{fmt(a["E1"]):>22}')

    print('\n==== Recall @ fixed region-FP (threshold on tune, recall on report) ====')
    for r in REGIONS:
        print(f'-- region: {r}')
        print(f'  {"FP":>5}{"E6":>22}{"CE+Tv":>22}{"E1":>22}{"E0-spatial":>22}')
        for a in ALPHAS:
            row = out['fn_at_fp'][r][f'{a:.2f}']
            print(f'  {a:5.2f}{fmt(row["E6"]):>22}{fmt(row["CE+Tv"]):>22}'
                  f'{fmt(row["E1"]):>22}{fmt(row["E0-spatial"]):>22}')

    print('\n==== Recall @ fixed nav-blocking (near high-risk occ recall) ====')
    print(f'  {"block":>6}{"E6":>22}{"CE+Tv":>22}{"E1(=E0)":>22}')
    for b in BETAS:
        row = out['recall_at_blocking'][f'{b:.2f}']
        print(f'  {b:6.2f}{fmt(row["E6"]):>22}{fmt(row["CE+Tv"]):>22}{fmt(row["E1"]):>22}')

    print('\n==== VERDICT (FP-controlled @ region-FP=0.10): E6 >= CE+Tv AND >= E0-spatial? ====')
    for r in REGIONS:
        v = out['verdict'][r]
        if isinstance(v, dict):
            print(f'  {r:13} E6={v["recall@FP0.10_E6_mean"]:.4f}  '
                  f'CE+Tv={v["CE+Tv_mean"]:.4f} (Δ{v["E6-CE+Tv"]:+.4f})  '
                  f'E0={v["E0_mean"]:.4f} (Δ{v["E6-E0"]:+.4f})  '
                  f'-> E6>=both: {v["E6>=both"]}')
        else:
            print(f'  {r:13} {v}')
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
