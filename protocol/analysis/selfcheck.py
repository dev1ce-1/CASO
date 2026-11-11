#!/usr/bin/env python
"""Compare regional and reference native FN rates for the same checkpoint.

Checks the full-validation distance-by-class FN matrix within a specified
tolerance. Matching rates are a consistency check, not proof that every
aspect of the prediction and ground-truth pipeline is correct."""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rpr', required=True, help='region_pr output json (FULL val)')
    ap.add_argument('--ref', required=True, help='original evaluator json, same ckpt')
    ap.add_argument('--tol', type=float, default=0.01, help='max relative error on FN rate')
    args = ap.parse_args()

    rpr = json.load(open(args.rpr))
    ref = json.load(open(args.ref))
    rate_r = rpr['fn_matrix']['rate']
    den_r = rpr['fn_matrix']['den']
    num_r = rpr['fn_matrix']['num']
    rate_ref = ref['fn_matrix']
    den_ref = ref.get('fn_matrix_den', {})
    num_ref = ref.get('fn_matrix_num', {})

    print(f'region_pr frames = {rpr["frames"]}   (ref = full official val)')
    print(f'{"cell":18}{"rpr_rate":>10}{"ref_rate":>10}{"relerr%":>9}'
          f'{"rpr_den":>10}{"ref_den":>10}  verdict')
    all_ok = True
    for k in sorted(rate_ref):
        rr = rate_r.get(k)
        rf = rate_ref.get(k)
        if rr is None or rf is None:
            print(f'{k:18}  MISSING in region_pr'); all_ok = False; continue
        rel = abs(rr - rf) / rf if rf else abs(rr - rf)
        den_match = den_r.get(k) == den_ref.get(k)
        ok = rel <= args.tol
        all_ok = all_ok and ok
        flag = 'OK' if ok else 'FAIL'
        if not den_match:
            flag += f' den({den_r.get(k)}!={den_ref.get(k)})'
        print(f'{k:18}{rr:10.5f}{rf:10.5f}{rel*100:9.3f}'
              f'{den_r.get(k):10}{den_ref.get(k, "?"):>10}  {flag}')

    print('\nOVERALL fn_matrix reproduction:', 'PASS' if all_ok else 'FAIL')
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
