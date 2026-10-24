"""One filtered pooled-snapshot fit; only post-June22 evaluation is valid."""
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
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from check_history_normalization import evaluate

BASE = Path(__file__).resolve().parents[1]


def main():
    started = time.perf_counter()
    out = BASE / 'evidence/filtered_multiwindow'
    cache = BASE / 'cache/filtered_multiwindow'
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    if (out / 'results.json').exists():
        raise SystemExit('Preserve completed filtered multiwindow experiment.')
    parameters = json.loads((BASE / 'evidence/training_filter/protocol.json').read_text())['parameters']
    protocol = {
        'fit_windows': [['2020-04-13', '2020-05-18'], ['2020-05-18', '2020-06-22']],
        'evaluation_window': ['2020-06-22', '2020-07-27'],
        'filter': 'Separately within each fitting snapshot, retain user groups with both retrieved positive and negative labels.',
        'weights': 'Equal weight per retained candidate row; concatenate filtered snapshots.',
        'parameters': parameters, 'training_runs': 1, 'submissions': 0,
        'evaluation': 'Later only, all1260 users, unchanged candidates/cold fallback/full-truth AP@5 denominator.',
        'caveat': 'Original main window overlaps fitting labels and must not be evaluated. Later window was previously explored; not independent confirmation.',
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    inputs = {'early': BASE / 'cache/context_features/fit.npz',
              'recent': BASE / 'cache/multiwindow/recent_fit.npz',
              'later': BASE / 'cache/context_features/later.npz'}
    arrays, labels, counts, hashes = [], [], {}, {}
    feature_names = None
    for name in ('early', 'recent'):
        with np.load(inputs[name], allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        if feature_names is None:
            feature_names = d['feature_names']
        assert np.array_equal(feature_names, d['feature_names'])
        mask = np.zeros(len(d['y']), dtype=bool)
        excluded_no_positive = excluded_no_negative = kept_groups = 0
        for lo, hi in zip(d['group_offsets'][:-1], d['group_offsets'][1:]):
            positives = int(d['y'][lo:hi].sum())
            if positives == 0:
                excluded_no_positive += 1
            elif positives == hi - lo:
                excluded_no_negative += 1
            else:
                mask[lo:hi] = True
                kept_groups += 1
        arrays.append(d['X'][mask])
        labels.append(d['y'][mask])
        counts[name] = {'input_groups': len(d['users']), 'input_rows': len(d['y']),
                        'retained_groups': kept_groups, 'retained_rows': int(mask.sum()),
                        'excluded_no_positive_groups': excluded_no_positive,
                        'excluded_no_negative_groups': excluded_no_negative}
        hashes[name] = hashlib.sha256(inputs[name].read_bytes()).hexdigest()
        del d
    (out / 'filter_counts.json').write_text(json.dumps(counts, indent=2))
    x, y = np.concatenate(arrays), np.concatenate(labels)
    del arrays, labels
    assert np.isfinite(x).all() and set(np.unique(y)) == {0, 1}
    model = HistGradientBoostingClassifier(**parameters)
    fit_started = time.perf_counter()
    model.fit(x, y)
    fit_seconds = time.perf_counter() - fit_started
    del x, y
    with np.load(inputs['later'], allow_pickle=False) as z:
        later = {k: z[k] for k in z.files}
    assert np.array_equal(feature_names, later['feature_names'])
    assert len(later['users']) == 1260
    scores = model.predict_proba(later['X'])[:, 1]
    metric, predictions = evaluate(later, scores)
    previous = json.loads((BASE / 'evidence/multiwindow/results.json').read_text())
    pooled = next(r['later_map5'] for r in previous['results'] if r['model'] == 'early_plus_recent')
    filtered = json.loads((BASE / 'evidence/training_filter/results.json').read_text())
    early = next(r['later']['map5'] for r in filtered['results'] if r['method'] == 'filtered_equal_rows')
    hashes['later'] = hashlib.sha256(inputs['later'].read_bytes()).hexdigest()
    result = {'status': 'complete', 'later_map5': metric,
              'filtered_early_reference': early, 'unfiltered_pooled_reference': pooled,
              'delta_vs_filtered_early': metric - early, 'delta_vs_unfiltered_pooled': metric - pooled,
              'filter_counts': counts, 'fit_seconds': fit_seconds,
              'wall_seconds': time.perf_counter() - started,
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
              'input_sha256': hashes,
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'protocol_sha256': hashlib.sha256((out / 'protocol.json').read_bytes()).hexdigest(),
              'submissions': 0}
    with (cache / 'filtered_pooled.pkl').open('wb') as f:
        pickle.dump(model, f)
    np.save(cache / 'later_scores.npy', scores)
    (out / 'later_predictions.json').write_text(json.dumps(predictions))
    (out / 'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
