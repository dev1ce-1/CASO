#!/usr/bin/env python
"""Aggregate the historical 200-bin fixed-path proxy diagnostics.

Thresholds are chosen on tune scenes under a near-field free-voxel FP budget.
Unsafe and over-block rates are evaluated on report scenes at those thresholds.
Budget matching does not imply equal report-set NearFP or equal over-blocking.
For final manuscript results use analysis/verify_release.py and the complete
canonical statistics in evidence/planner/. E0-spatial coincides with E1 in
this diagnostic's single near-field threshold setting."""
import json
import numpy as np

A = 'reports/formal/analysis'
METHODS = {'E6': ['E6_seed0', 'E6_seed1', 'E6_seed2'],
           'CE+Tv': ['CETVERSKY_seed0', 'CETVERSKY_seed1'],
           'E1': ['E1_seed0', 'E1_seed1', 'E1_seed2']}
BETAS = [0.05, 0.10]
SCOPES = ['all', 'near_dynamic']


def k_at_fp(neg, alpha):
    tot = neg.sum()
    if tot == 0:
        return len(neg)
    cum = np.cumsum(neg[::-1])[::-1]
    fpr = cum / tot
    ks = np.where(fpr <= alpha)[0]
    return int(ks[0]) if ks.size else len(neg)


def load_pl(tag):
    return json.load(open(f'{A}/planner_{tag}.json'))


def load_rp(tag):
    return json.load(open(f'{A}/region_pr_{tag}.json'))


def rates_at_budget(pl, rp, beta, scope):
    """Return (unsafe_rate, over_block_rate, realized_report_blocking) at budget."""
    navfree_tune = np.array(rp['nav']['tune']['free_hist'], float)
    k = k_at_fp(navfree_tune, beta)
    s = pl['scopes'][scope]
    uh = np.array(s['unsafe_hist'], float)
    oh = np.array(s['over_hist'], float)
    n = s['n_paths']
    unsafe = float(uh[:k].sum() / n) if n else float('nan')
    over = float(oh[k:].sum() / n) if n else float('nan')
    navfree_rep = np.array(rp['nav']['report']['free_hist'], float)
    blk = float(navfree_rep[k:].sum() / navfree_rep.sum()) if navfree_rep.sum() else float('nan')
    return unsafe, over, blk


def agg(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return None
    return {'mean': float(np.mean(v)), 'min': float(np.min(v)), 'max': float(np.max(v))}


def fmt(a):
    return f'{a["mean"]:.4f}[{a["min"]:.4f}-{a["max"]:.4f}]' if a else 'n/a'


def main():
    data = {m: [(load_pl(t), load_rp(t)) for t in tags] for m, tags in METHODS.items()}

    out = {'betas': BETAS, 'main': {}, 'curve': {}, 'verdict': {}}

    # main table
    for scope in SCOPES:
        out['main'][scope] = {}
        for beta in BETAS:
            row = {}
            for m in METHODS:
                us = [rates_at_budget(pl, rp, beta, scope)[0] for pl, rp in data[m]]
                ov = [rates_at_budget(pl, rp, beta, scope)[1] for pl, rp in data[m]]
                bk = [rates_at_budget(pl, rp, beta, scope)[2] for pl, rp in data[m]]
                row[m] = {'unsafe': agg(us), 'over_block': agg(ov), 'report_blocking': agg(bk)}
            row['E0-spatial'] = row['E1']   # single near-field threshold
            out['main'][scope][f'{beta:.2f}'] = row

    # trade-off curve: sweep beta, record (report_blocking, unsafe) per method (all scope)
    betas_curve = [0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]
    for scope in SCOPES:
        out['curve'][scope] = {}
        for m in METHODS:
            pts = []
            for beta in betas_curve:
                us = agg([rates_at_budget(pl, rp, beta, scope)[0] for pl, rp in data[m]])
                bk = agg([rates_at_budget(pl, rp, beta, scope)[2] for pl, rp in data[m]])
                ov = agg([rates_at_budget(pl, rp, beta, scope)[1] for pl, rp in data[m]])
                if us and bk:
                    pts.append({'beta': beta, 'blocking': bk['mean'],
                                'unsafe': us['mean'], 'over_block': ov['mean'],
                                'unsafe_min': us['min'], 'unsafe_max': us['max']})
            out['curve'][scope][m] = pts

    # verdict @ beta=0.10: E6 unsafe <= CE+Tv AND <= E0-spatial (=E1)?  lower=safer
    for scope in SCOPES:
        row = out['main'][scope]['0.10']
        e6 = row['E6']['unsafe']; ce = row['CE+Tv']['unsafe']; e0 = row['E0-spatial']['unsafe']
        if not (e6 and ce and e0):
            out['verdict'][scope] = 'INSUFFICIENT'; continue
        out['verdict'][scope] = {
            'E6_unsafe': e6['mean'], 'CE+Tv_unsafe': ce['mean'], 'E0_unsafe': e0['mean'],
            'E6_over_block': row['E6']['over_block']['mean'],
            'CE+Tv_over_block': row['CE+Tv']['over_block']['mean'],
            'E6-CE+Tv': e6['mean'] - ce['mean'], 'E6-E0': e6['mean'] - e0['mean'],
            'E6<=both(safer)': bool(e6['mean'] <= ce['mean'] and e6['mean'] <= e0['mean'])}

    json.dump(out, open(f'{A}/step6_metrics.json', 'w'), indent=2)

    print('==== §6.6 planner-proxy @ matched nav-blocking budget ====')
    print('(unsafe & over_block ALWAYS paired; E0-spatial=E1 at near-field single threshold)')
    for scope in SCOPES:
        print(f'\n---- scope: {scope} ----')
        for beta in BETAS:
            print(f'  budget beta={beta:.2f}  (realized report-blocking in [])')
            row = out['main'][scope][f'{beta:.2f}']
            print(f'    {"model":11}{"unsafe_rate":>24}{"over_block_rate":>24}{"blocking":>12}')
            for m in ['E6', 'CE+Tv', 'E1', 'E0-spatial']:
                r = row[m]
                bk = r['report_blocking']['mean'] if r['report_blocking'] else float('nan')
                print(f'    {m:11}{fmt(r["unsafe"]):>24}{fmt(r["over_block"]):>24}{bk:>12.4f}')

    print('\n==== unsafe-vs-blocking trade-off (all scope, mean) ====')
    print(f'  {"blk":>6}{"E6":>10}{"CE+Tv":>10}{"E1":>10}')
    # align by beta index
    for i, beta in enumerate(betas_curve):
        def g(m):
            p = out['curve']['all'][m]
            return p[i]['unsafe'] if i < len(p) else float('nan')
        def gb(m):
            p = out['curve']['all'][m]
            return p[i]['blocking'] if i < len(p) else float('nan')
        print(f'  b={beta:.2f} blk≈{gb("E6"):.3f}  E6={g("E6"):.4f}  '
              f'CE+Tv={g("CE+Tv"):.4f}  E1={g("E1"):.4f}')

    print('\n==== VERDICT (@ budget 0.10): E6 unsafe <= CE+Tv AND <= E0-spatial? (lower=safer) ====')
    for scope in SCOPES:
        v = out['verdict'][scope]
        if isinstance(v, dict):
            print(f'  {scope:13} unsafe: E6={v["E6_unsafe"]:.4f} '
                  f'CE+Tv={v["CE+Tv_unsafe"]:.4f}(Δ{v["E6-CE+Tv"]:+.4f}) '
                  f'E0={v["E0_unsafe"]:.4f}(Δ{v["E6-E0"]:+.4f}) '
                  f'| over_block E6={v["E6_over_block"]:.4f} CE+Tv={v["CE+Tv_over_block"]:.4f} '
                  f'-> E6<=both: {v["E6<=both(safer)"]}')
        else:
            print(f'  {scope:13} {v}')
    print(f'\nwrote {A}/step6_metrics.json')


if __name__ == '__main__':
    main()
