"""Audit-only rank calculations; future-population normalizers are unknowable.

Reports an explicit finite observed-attempt population, plus metric dominance
which is independent of normalization. Never uses hidden information in a
method or shortcut implementation.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/user')
OUT = ROOT / 'redteam'
EXP = 'captaincook4d_task_graph'


def main():
    spec = json.loads((ROOT / 'scoring/reference.json').read_text())['experiments'][EXP]
    reference = json.loads((ROOT / 'scoring/scores_test.json').read_text())['experiments'][EXP]['results']['do']
    metrics = list(spec['metrics'])
    rows, invalid = {}, {}
    for status_path in sorted(OUT.glob('*/status.json')):
        folder = status_path.parent
        status = json.loads(status_path.read_text())
        score_path = folder / 'scores_test.json'
        if status.get('exit_code') != 0 or not score_path.exists():
            invalid[folder.name] = status
            continue
        obj = json.loads(score_path.read_text())
        result = obj.get('experiments', {}).get(EXP, {}).get('results', {})
        key = 'count_based' if folder.name == 'simple_baseline' else 'do'
        if key not in result or not all(np.isfinite(result[key].get(m, float('nan'))) for m in metrics):
            invalid[folder.name] = {'reason': 'nonfinite or missing metric', 'results': result}
            continue
        rows[folder.name] = {'metrics': result[key], 'manifest': obj.get('slice_manifest', {}),
                             'additional_results': {k: v for k, v in result.items() if k != key}}
    matrix = np.array([[row['metrics'][m] for m in metrics] for row in rows.values()])
    scales = matrix.std(axis=0)
    scales[scales == 0] = 1.0
    signs = np.array([1 if spec['metrics'][m]['higher_is_better'] else -1 for m in metrics])
    weights = np.array([abs(spec['metrics'][m]['coefficient']) for m in metrics])
    ref = np.array([reference[m] for m in metrics])
    for row in rows.values():
        delta = (np.array([row['metrics'][m] for m in metrics]) - ref) * signs
        contributions = spec['weight'] * weights * delta / scales
        row['audit_population_rank_delta'] = float(contributions.sum())
        row['rank_contributions'] = dict(zip(metrics, contributions.tolist()))
        row['dominates_reference'] = bool(np.all(delta >= -1e-10) and np.any(delta > 1e-10))
        row['tied_reference'] = bool(np.all(np.abs(delta) <= 1e-10))
    seed_rows = [rows[f'reference_seed_{seed}'] for seed in [7, 19, 31] if f'reference_seed_{seed}' in rows]
    f1s = [r['metrics']['f1'] for r in seed_rows]
    ranks = [r['audit_population_rank_delta'] for r in seed_rows]
    bounds = {'precision': 100, 'recall': 100, 'f1': 100, 'std_f1': 0,
              'worst_quartile_f1': 100, 'edge_density_ratio': 1, 'path_coverage': 1,
              'topological_distance': 0}
    result = {'reference': reference, 'rows': rows, 'invalid': invalid,
              'rank_definition': 'sum experiment_weight * abs(coefficient) * directed_delta / population_std',
              'normalizer_population': list(rows), 'normalizers': dict(zip(metrics, scales.tolist())),
              'actual_future_rank': 'Not identifiable until the future ranked population is known; dominance comparisons do not depend on it.',
              'reference_metrics_at_bound': [m for m in metrics if abs(reference[m] - bounds[m]) < 1e-10],
              'seed_f1s': f1s, 'seed_f1_spread': max(f1s) - min(f1s) if f1s else None,
              'seed_rank_spread': max(ranks) - min(ranks) if ranks else None,
              'seed_f1_std': float(np.std(f1s)) if f1s else None}
    (OUT / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print('CASE, F1, STD_F1, RANK_DELTA, COVERAGE, DOMINATES')
    for name, row in rows.items():
        print(name, round(row['metrics']['f1'], 6), round(row['metrics']['std_f1'], 6),
              round(row['audit_population_rank_delta'], 6), row['manifest'].get('scenario_count'),
              row['dominates_reference'])
    print('INVALID', list(invalid))
    print('SEED F1 SPREAD', result['seed_f1_spread'], 'RANK SPREAD', result['seed_rank_spread'])


if __name__ == '__main__':
    main()
