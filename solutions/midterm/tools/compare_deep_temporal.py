"""Frozen deep-temporal models versus submitted v2; no training or fusion search."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import json
from pathlib import Path
import numpy as np
from check_history_normalization import B, evaluate
from run_round1 import apk


def ap_vectors(data, recs, top_truth):
    broad, top = [], []
    for n, uid in enumerate(data['users']):
        u = str(uid); a, b = data['group_offsets'][n:n+2]
        lookup = {str(data['item_ids'][idx]): bool(y) for idx, y in
                  zip(data['row_item_indices'][a:b], data['y'][a:b])}
        items = recs[u]
        assert len(items) == 5 and len(set(items)) == 5
        assert all(i in lookup for i in items), 'Prediction outside fixed candidate set'
        hits = 0; score = 0.
        for rank, item in enumerate(items, 1):
            if lookup[item]:
                hits += 1; score += hits/rank
        denominator = min(5, int(data['truth_counts'][n]))
        broad.append(score/denominator if denominator else 0.)
        top.append(apk(top_truth[u], items))
        if data['cold'][n]:
            expected = [str(data['item_ids'][i]) for i in data['fallback_item_indices'][:5]]
            assert items == expected, 'Cold fallback differs'
    return {'all80': np.array(broad), 'top5_count80': np.array(top)}


def main():
    out = B/'evidence/deep_temporal/comparison.json'
    marker = B/'evidence/deep_temporal/results.json'
    if not marker.exists() or json.loads(marker.read_text())['status'] != 'complete':
        raise SystemExit('Deep training not complete; comparison not run')
    if out.exists():
        raise SystemExit('Preserve existing comparison')
    completed = json.loads(marker.read_text())
    variants = [f"{row['variant']}_{row['width']}features" for row in completed['results'] if row['phase'] == 'refit']
    designated = 'top5_count80_24features'
    assert designated in variants
    with np.load(B/'cache/simple_reranker.npz') as z:
        model = {k: z[k] for k in z.files}
    results = {}
    for window in ('eval', 'later'):
        with np.load(B/f'cache/context_features/{window}.npz') as z:
            d = {k: z[k] for k in z.files}
        s1 = ((d['X'][:, :9]-model['mean'])/model['scale'])@model['coefficients'][:-1]+model['coefficients'][-1]
        s2 = np.load(B/f'cache/training_filter/filtered_equal_rows_{window}_scores.npy')
        fused = np.zeros(len(s1))
        for a, b in zip(d['group_offsets'][:-1], d['group_offsets'][1:]):
            items = d['item_ids'][d['row_item_indices'][a:b]]
            for scores in (s1, s2):
                order = sorted(range(b-a), key=lambda j: (-scores[a+j], int(items[j])))
                ranks = np.empty(b-a); ranks[order] = np.arange(1, b-a+1)
                fused[a:b] += .5/(60+ranks)
        _, baseline = evaluate(d, fused)
        truth = json.loads((B/f'evidence/top5_target_audit/{window}_truth.json').read_text())['top5_count80']
        vectors = {'submitted_v2': ap_vectors(d, baseline, truth)}
        for variant in variants:
            preds = json.loads((B/f'evidence/deep_temporal/{variant}_{window}_predictions.json').read_text())
            vectors[variant] = ap_vectors(d, preds, truth)
        table = {name: {target: float(x.mean()) for target, x in vals.items()} for name, vals in vectors.items()}
        intervals = {}
        for target in ('all80', 'top5_count80'):
            delta = vectors[designated][target]-vectors['submitted_v2'][target]
            rng = np.random.default_rng(20261023)
            means = np.concatenate([delta[rng.integers(0, len(delta), size=(100, len(delta)))].mean(axis=1) for _ in range(50)])
            intervals[target] = {'mean_delta': float(delta.mean()),
                'paired_user_percentile_interval_95': np.quantile(means, [.025, .975]).tolist(),
                'users_improved': int((delta>0).sum()), 'users_worsened': int((delta<0).sum()),
                'users_equal': int((delta==0).sum())}
        results[window] = {'scores': table, 'preselected_candidate_vs_v2': intervals,
                           'cold_fallback_verified_users_per_model': int(d['cold'].sum()), 'users': len(d['users'])}
    report = {'results': results, 'preselected_candidate': designated, 'bootstrap_replicates': 5000,
              'seed': 20261023, 'fits': 0, 'submissions': 0, 'fusion_search': False,
              'limitations': 'Conditional on fixed models and repeatedly explored data; not adjusted for selection. Users may be dependent, periods overlap; intervals do not quantify Kaggle uncertainty.',
              'all80_denominator': 'min(5, full-universe truth count), never shortlist positives',
              'baseline': 'Submitted v2 equal RRF k=60 reconstructed from frozen logistic and filtered-context scores'}
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
