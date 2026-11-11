"""Verify release integrity and recompute paper metrics from saved scene statistics.

Run with Python >=3.8 and NumPy. No CUDA, checkpoints or dataset required.
This verifies stored evidence, not the prediction/ground-truth generation process.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    'E1': ['E1_seed0', 'E1_seed1', 'E1_seed2'],
    'E6': ['E6_seed0', 'E6_seed1', 'E6_seed2'],
    'CE+Tv': ['CETVERSKY_seed0', 'CETVERSKY_seed1', 'CETVERSKY_seed2'],
    'CTRL-U': ['CTRL_U_seed0', 'CTRL_U_seed1'],
    'CTRL-F': ['CTRL_F_seed0', 'CTRL_F_seed1'],
}

def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf8'))

def require(condition, message):
    if not condition:
        raise ValueError(message)

def auprc(pos, neg):
    tp = np.cumsum(pos[::-1])
    fp = np.cumsum(neg[::-1])
    require(pos.sum() > 0, 'Empty positive histogram')
    recall = np.r_[0., tp / pos.sum()]
    precision = np.r_[1., tp / np.maximum(tp + fp, 1e-12)]
    integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return float(integrate(precision, recall))

def verify():
    manifest = read('SOURCE_MANIFEST.json')
    for entry in manifest['files']:
        path = ROOT / entry['path']
        require(path.is_file(), 'Missing source file: ' + entry['path'])
        require(hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'],
                'Source checksum mismatch: ' + entry['path'])
    checksums = ROOT / 'SHA256SUMS'
    if checksums.exists():
        for line in checksums.read_text().splitlines():
            digest, name = line.split('  ', 1)
            require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest,
                    'Release checksum mismatch: ' + name)

    # Static syntax and configuration inheritance checks do not import ML code.
    python_count = 0
    for path in ROOT.rglob('*.py'):
        if '.git' in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        python_count += 1
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_base_' for t in node.targets):
                bases = ast.literal_eval(node.value)
                for base in ([bases] if isinstance(bases, str) else bases):
                    target = (path.parent / base).resolve()
                    if 'mmdetection3d' not in target.parts:
                        require(target.is_file(), 'Missing config base: ' + str(target))

    split = read('protocol/split_scene.json')
    tune, report = set(split['tune_scenes']), set(split['report_scenes'])
    require(len(tune) == len(report) == 75 and not tune & report, 'Invalid 75/75 split')
    for token in tune | report:
        score = int(hashlib.sha1((split['salt'] + '::' + token).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        require((score < .5) == (token in tune), 'Split algorithm mismatch: ' + token)

    reference = read('evidence/planner/bootstrap_ci.json')
    per_seed = {}
    for tags in GROUPS.values():
        for tag in tags:
            with np.load(ROOT / 'evidence/scene_statistics' / ('g0_' + tag + '.npz'), allow_pickle=False) as data:
                tokens = data['scene_tokens']
                require(len(tokens) == 150 and set(tokens) == tune | report, tag + ': scene universe')
                ti = np.array([t in tune for t in tokens])
                ri = np.array([t in report for t in tokens])
                require(np.array_equal(data['scene_split'], np.where(ti, 0, 1)), tag + ': split labels')
                # Axis 1 is region (far, near_dynamic, boundary); axis 2 is distance.
                pos = data['region_pos'][ri, 1].sum(axis=(0, 1)).astype(float)
                neg = data['region_neg'][ri, 1].sum(axis=(0, 1)).astype(float)
                free_tune = data['nav_free'][ti].sum(0).reshape(200, 6).sum(1)
                free_report = data['nav_free'][ri].sum(0).reshape(200, 6).sum(1)
                hr_report = data['nav_hr'][ri].sum(0).reshape(200, 6).sum(1)
                candidates = np.where(np.cumsum(free_tune[::-1])[::-1] / free_tune.sum() <= .1)[0]
                threshold = int(candidates[0]) if len(candidates) else 200
                values = {
                    'near_dynamic_auprc_mean_of_seed': auprc(pos, neg),
                    'nav_recall_at_fp_0p10': float(hr_report[threshold:].sum() / hr_report.sum()),
                    'realized_blocking_at_fp_0p10': float(free_report[threshold:].sum() / free_report.sum()),
                }
            with np.load(ROOT / 'evidence/planner/per_scene' / (tag + '.npz'), allow_pickle=False) as data:
                require(set(data['canonical_scene_tokens']) == report, tag + ': planner scenes')
                require(int(data['threshold_k200']) == threshold, tag + ': planner threshold')
                require(float(data['n_paths'].sum()) == 15060, tag + ': path count')
                for key in ['unsafe_all', 'unsafe_near_dynamic', 'over_block_all', 'over_block_near_dynamic']:
                    values[key + '_at_fp_0p10'] = float(data[key + '_num'].sum() / data['n_paths'].sum())
            per_seed[tag] = values

    groups = {}
    for group, tags in GROUPS.items():
        groups[group] = {}
        for key in per_seed[tags[0]]:
            value = float(np.mean([per_seed[t][key] for t in tags]))
            expected = reference['group_point_estimates'][group][key]
            require(abs(value - expected) <= 1e-10, '{} / {}: {} != {}'.format(group, key, value, expected))
            groups[group][key] = value
    return {'status': 'PASS', 'source_files_verified': len(manifest['files']),
            'python_files_parsed': python_count, 'tune_scenes': 75, 'report_scenes': 75,
            'seeds_recomputed': len(per_seed), 'group_metrics_recomputed': sum(map(len, groups.values())),
            'group_point_estimates': groups,
            'scope': 'Saved statistics and release integrity; no inference or confidence-interval rerun.'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional destination for the verification JSON')
    args = parser.parse_args()
    result = verify()
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf8')
    print(text)
