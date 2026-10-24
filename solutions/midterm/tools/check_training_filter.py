"""Two bounded training-group filtering experiments; no network or submission."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'

import hashlib
import json
import pickle
import resource
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from check_history_normalization import evaluate

BASE = Path(__file__).resolve().parents[1]


def main():
    out = BASE / 'evidence/training_filter'
    cache = BASE / 'cache/training_filter'
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    if (out / 'results.json').exists():
        raise SystemExit('Preserve completed training-filter experiment.')
    began = time.perf_counter()
    reference = json.loads((BASE / 'evidence/context_boosting/protocol.json').read_text())
    context = next(m for m in reference['models'] if m['name'] == 'boost_context')
    parameters = dict(learning_rate=reference['learning_rate'], max_iter=reference['iterations'],
                      l2_regularization=reference['l2'], min_samples_leaf=reference['min_samples_leaf'],
                      early_stopping=reference['early_stopping'], random_state=reference['seed'],
                      max_leaf_nodes=context['max_leaf_nodes'], max_depth=context['max_depth'])
    protocol = {
        'fit_window': ['2020-04-13', '2020-05-18'],
        'fit_history_start': '2020-03-09',
        'evaluation_windows': {'eval': ['2020-05-18', '2020-07-27'], 'later': ['2020-06-22', '2020-07-27']},
        'filter': 'Keep early-fit user groups only when retrieved candidates include at least one positive and at least one negative.',
        'filter_labels': 'early fitting-window labels only; never evaluation labels/activity',
        'variants': ['filtered_equal_rows', 'filtered_equal_users'],
        'equal_user_weight': 'N_retained_rows / (N_retained_groups * rows_in_group); sum weights=N_retained_rows',
        'parameters': parameters, 'features': 'all19 frozen context features',
        'evaluation': 'unchanged candidates, all1260 users, empty truth AP=0, full-truth AP@5 denominator, common cold popularity fallback',
        'selection_caveat': 'Both holdouts previously explored and overlapping; exploratory comparison, not untouched confirmation.',
        'statistical_caveat': 'Filtering drops valid negative-only groups and changes population/calibration; it is an explicit hypothesis, not universally correct cleaning.',
        'training_runs': 2, 'submissions': 0, 'threads': 1,
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))

    def load(name):
        with np.load(BASE / f'cache/context_features/{name}.npz', allow_pickle=False) as source:
            return {k: source[k] for k in source.files}

    fit, mainval, later = (load(name) for name in ('fit', 'eval', 'later'))
    assert np.array_equal(fit['feature_names'], mainval['feature_names'])
    assert np.array_equal(fit['feature_names'], later['feature_names'])
    assert len(mainval['users']) == len(later['users']) == 1260
    y = fit['y']
    keep = np.zeros(len(y), dtype=bool)
    groups = []
    counts = {'input_groups': len(fit['users']), 'input_rows': len(y),
              'excluded_no_positive_groups': 0, 'excluded_no_negative_groups': 0,
              'excluded_no_positive_rows': 0, 'excluded_no_negative_rows': 0}
    for n, user in enumerate(fit['users']):
        lo, hi = map(int, fit['group_offsets'][n:n + 2])
        positives = int(y[lo:hi].sum())
        if positives == 0:
            counts['excluded_no_positive_groups'] += 1
            counts['excluded_no_positive_rows'] += hi - lo
        elif positives == hi - lo:
            counts['excluded_no_negative_groups'] += 1
            counts['excluded_no_negative_rows'] += hi - lo
        else:
            keep[lo:hi] = True
            groups.append((str(user), lo, hi))
    n_rows, n_groups = int(keep.sum()), len(groups)
    assert n_rows > 0 and n_groups > 0
    weights = np.zeros(len(y), dtype=np.float64)
    for _, lo, hi in groups:
        weights[lo:hi] = n_rows / (n_groups * (hi - lo))
    selected_weights = weights[keep]
    assert np.isclose(selected_weights.sum(), n_rows)
    assert all(np.isclose(weights[lo:hi].sum(), n_rows / n_groups) for _, lo, hi in groups)
    counts.update(retained_groups=n_groups, retained_rows=n_rows,
                  input_positive_rate=float(y.mean()), retained_positive_rate=float(y[keep].mean()),
                  equal_user_weight_min=float(selected_weights.min()),
                  equal_user_weight_max=float(selected_weights.max()))
    (out / 'filter_counts.json').write_text(json.dumps(counts, indent=2))
    print(json.dumps({'phase': 'filter_complete', **counts}), flush=True)
    x, yy = fit['X'][keep], y[keep]
    assert np.isfinite(x).all() and set(np.unique(yy)) == {0, 1}
    results = []
    for variant in protocol['variants']:
        start_fit = time.perf_counter()
        model = HistGradientBoostingClassifier(**parameters)
        model.fit(x, yy, sample_weight=selected_weights if variant == 'filtered_equal_users' else None)
        result = {'method': variant, 'fit_seconds': time.perf_counter() - start_fit,
                  'fit_groups': n_groups, 'fit_rows': n_rows}
        for name, data in [('eval', mainval), ('later', later)]:
            scores = model.predict_proba(data['X'])[:, 1]
            assert np.isfinite(scores).all()
            metric, predictions = evaluate(data, scores)
            result[name] = {'map5': metric}
            (out / f'{variant}_{name}_predictions.json').write_text(json.dumps(predictions))
            np.save(cache / f'{variant}_{name}_scores.npy', scores)
        with (cache / f'{variant}.pkl').open('wb') as f:
            pickle.dump(model, f)
        results.append(result)
        (out / 'results.partial.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(result), flush=True)
    report = {'status': 'complete', 'results': results, 'filter_counts': counts,
              'wall_seconds': time.perf_counter() - began,
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
              'sklearn_version': sklearn.__version__,
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'protocol_sha256': hashlib.sha256((out / 'protocol.json').read_bytes()).hexdigest(),
              'input_sha256': {name: hashlib.sha256((BASE / f'cache/context_features/{name}.npz').read_bytes()).hexdigest()
                               for name in ('fit', 'eval', 'later')},
              'submissions': 0}
    (out / 'results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'status': 'complete', 'wall_seconds': report['wall_seconds'],
                      'peak_rss_bytes': report['peak_rss_bytes']}), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
