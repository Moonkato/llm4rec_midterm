"""Bounded temporal LambdaRank selection. Execution requires prior user approval.

No work runs on import. Inner-stop MAP uses full-universe truth denominators.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '2'
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import resource
import sqlite3
import sys
import time
import numpy as np
import lightgbm as lgb
from threadpoolctl import threadpool_limits
from build_interactions import tick

BASE = Path(__file__).resolve().parents[1]
WINDOWS = {
    'inner_fit': ('2020-04-13', '2020-05-04'),
    'inner_stop': ('2020-05-04', '2020-05-18'),
    'fit': ('2020-04-13', '2020-05-18'),
    'eval': ('2020-05-18', '2020-07-27'),
    'later': ('2020-06-22', '2020-07-27'),
}
CONFIGS = [
    dict(num_leaves=7, max_depth=3, min_child_samples=100, reg_lambda=20.),
    dict(num_leaves=15, max_depth=4, min_child_samples=100, reg_lambda=10.),
    dict(num_leaves=31, max_depth=5, min_child_samples=150, reg_lambda=20.),
]
PARAMS = dict(objective='lambdarank', metric='None', n_estimators=2000,
              learning_rate=.03, random_state=20261023, n_jobs=2, verbosity=-1,
              deterministic=True, force_col_wise=True, label_gain=[0, 1],
              lambdarank_truncation_level=10)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)


def full_truth(db, start, end):
    """Count qualifying user-airings before selecting any retrieved candidates."""
    a, b = tick(start + ' 00:00:00'), tick(end + ' 00:00:00')
    query = '''SELECT o.user,e.item,COUNT(*) FROM
      (SELECT user,event_id FROM overlaps WHERE start<? AND end>?
       GROUP BY user,event_id HAVING MAX(MAX(0,MIN(end,?)-MAX(start,?)))>
       .8*(SELECT duration FROM events WHERE events.event_id=overlaps.event_id)) o
      JOIN events e USING(event_id) GROUP BY o.user,e.item'''
    counts = defaultdict(list)
    for u, i, n in db.execute(query, (b, a, b, a)):
        counts[u].append((n, i))
    return {
        'all80': {u: {i for n, i in vals} for u, vals in counts.items()},
        'top5_count80': {u: {i for n, i in sorted(vals, key=lambda x: (-x[0], int(x[1])))[:5]}
                         for u, vals in counts.items()},
    }


class Metric:
    """Vectorized deterministic ranking with complete-truth MAP denominators."""
    def __init__(self, data, truth):
        self.d = data
        offsets = data['group_offsets']
        assert np.all(np.diff(offsets) >= 5), 'Every query must provide five candidates'
        self.g = np.repeat(np.arange(len(data['users'])), np.diff(offsets))
        self.ids = np.array([int(data['item_ids'][i]) for i in data['row_item_indices']], dtype=np.int64)
        self.y = np.array([str(i) in truth.get(str(data['users'][g]), set()) for g, i in zip(self.g, self.ids)], dtype=np.uint8)
        self.den = np.array([min(5, len(truth.get(str(u), set()))) for u in data['users']], dtype=float)
        self.cold_rows = np.flatnonzero(data['cold'][self.g])
        self.fallback_scores = np.zeros(len(self.cold_rows), dtype=float)
        fallback = {int(i): rank for rank, i in enumerate(data['fallback_item_indices'][:5])}
        assert len(fallback) == 5
        for n in np.flatnonzero(data['cold']):
            a, b = offsets[n:n+2]
            assert set(fallback).issubset(set(map(int, data['row_item_indices'][a:b]))), 'Cold fallback absent from shortlist'
        for k, row in enumerate(self.cold_rows):
            self.fallback_scores[k] = -fallback.get(int(data['row_item_indices'][row]), 1000000)

    def rank(self, scores):
        scores = np.array(scores, dtype=float, copy=True)
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite predictions')
        scores[self.cold_rows] = self.fallback_scores
        order = np.lexsort((self.ids, -scores, self.g))
        group = self.g[order]
        positions = np.arange(len(order)) - self.d['group_offsets'][group] + 1
        chosen = positions <= 5
        rows, group, positions = order[chosen], group[chosen], positions[chosen]
        hit = self.y[rows]
        cumulative = np.cumsum(hit)
        starts = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])
        shift = np.r_[0, cumulative[starts[1:] - 1]]
        within_hits = cumulative - np.repeat(shift, np.diff(np.r_[starts, len(rows)]))
        sums = np.bincount(group, weights=hit * within_hits / positions, minlength=len(self.den))
        ap = np.divide(sums, self.den, out=np.zeros_like(sums), where=self.den > 0)
        return float(ap.mean()), rows

    def __call__(self, y_true, scores):
        return 'full_truth_map5', self.rank(scores)[0], True

    def predictions(self, scores):
        result = {str(u): [] for u in self.d['users']}
        for row in self.rank(scores)[1]:
            result[str(self.d['users'][self.g[row]])].append(str(self.ids[row]))
        return result


def training_rows(data, metric):
    keep = np.zeros(len(metric.y), dtype=bool)
    groups = []
    for a, b in zip(data['group_offsets'][:-1], data['group_offsets'][1:]):
        positives = int(metric.y[a:b].sum())
        if 0 < positives < b - a:
            keep[a:b] = True
            groups.append(int(b-a))
    if not groups:
        raise ValueError('No eligible training groups')
    return keep, groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-ablations', action='store_true', help='Two additional full-fit base19 models; no outer selection')
    args = parser.parse_args()
    out, cache = BASE/'evidence/deep_temporal', BASE/'cache/deep_temporal'
    out.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        raise SystemExit('Preserve existing run: protocol already exists')
    protocol = dict(windows=WINDOWS, configurations=CONFIGS, parameters=PARAMS,
        features='base19 plus five aligned weekly features', max_fits=12,
        planned_fits=10 if args.baseline_ablations else 8, wall_seconds_limit=2700,
        rss_bytes_limit=4*1024**3, threads=2, early_stopping_rounds=100,
        targets=['all80', 'top5_count80'], top5='Full universe >80% airing count, numeric ID ties, before shortlist',
        selection='Within target select configuration and iteration solely by inner_stop full-truth MAP@5; deterministic first config on ties',
        refit='Original Apr13-May18 labels at selected fixed iteration; no early stopping on outer data',
        optional_ablation='Base19 refit at same chosen configuration/iteration, no additional selection',
        candidates='Existing fixed snapshots; cold fallback unchanged',
        limits='Runtime/RSS monitored before loading/fitting and each boosting iteration; not OS hard allocation cap',
        reuse='All observations previously explored; improved temporal separation does not create untouched confirmation',
        horizon_mismatch='Inner fit21days / stop14days versus outer70/35days and final84days; temporal separation does not remove horizon distribution shift',
        fragment_caveat='Maximum stored clipped overlap; original sessions not rejoined', submissions=0)
    write(out/'protocol.json', protocol)
    began = time.perf_counter(); fits = 0; results = []; inputs = {}
    def guard(env=None):
        if time.perf_counter()-began > 2700 or rss() > 4*1024**3:
            raise RuntimeError('Prespecified wall-time or RSS budget exceeded')
    guard.order = -100
    guard.before_iteration = True
    try:
        datasets, metrics = {}, {}
        db = sqlite3.connect('file:'+str(BASE/'cache/interactions.sqlite')+'?mode=ro', uri=True)
        for name, window in WINDOWS.items():
            guard()
            folder = 'deep_context' if name.startswith('inner_') else 'context_features'
            source = BASE/f'cache/{folder}/{name}.npz'
            extra_path = BASE/f'cache/deep_weekly/{name}_row_features.npy'
            with np.load(source, allow_pickle=False) as z:
                d = {k: z[k] for k in z.files}
            extra = np.load(extra_path, allow_pickle=False)
            assert d['X'].shape[1] == 19 and extra.shape == (len(d['y']), 5)
            assert np.isfinite(extra).all()
            d['X'] = np.concatenate([d['X'], extra], axis=1)
            truths = full_truth(db, *window)
            metrics[name] = {v: Metric(d, t) for v, t in truths.items()}
            assert np.array_equal(metrics[name]['all80'].y, d['y']), name+' labels differ'
            assert np.array_equal(metrics[name]['all80'].den, np.minimum(5, d['truth_counts'])), name+' denominator differs'
            datasets[name] = d
            for p in (source, extra_path):
                inputs[str(p.relative_to(BASE))] = hashlib.sha256(p.read_bytes()).hexdigest()
        db.close()
        write(out/'inputs.json', inputs)
        selected = {}
        for variant in protocol['targets']:
            d, m = datasets['inner_fit'], metrics['inner_fit'][variant]
            stop, sm = datasets['inner_stop'], metrics['inner_stop'][variant]
            keep, groups = training_rows(d, m)
            candidates = []
            for config_id, config in enumerate(CONFIGS):
                guard(); assert fits < 12
                model = lgb.LGBMRanker(**PARAMS, **config)
                fits += 1; started = time.perf_counter()
                model.fit(d['X'][keep], m.y[keep], group=groups,
                    eval_set=[(stop['X'], sm.y)], eval_group=[np.diff(stop['group_offsets'])],
                    eval_metric=sm, callbacks=[guard, lgb.early_stopping(100, first_metric_only=True, verbose=False)])
                score = sm.rank(model.predict(stop['X'], num_iteration=model.best_iteration_))[0]
                row = dict(phase='inner', variant=variant, config_id=config_id,
                    best_iteration=int(model.best_iteration_), inner_stop_map5=score,
                    fit_seconds=time.perf_counter()-started)
                candidates.append(row); results.append(row)
                write(out/'results.partial.json', results); print(json.dumps(row), flush=True)
            selected[variant] = max(candidates, key=lambda r: (r['inner_stop_map5'], -r['config_id']))
        write(out/'selection.json', selected)
        for variant, winner in selected.items():
            d, m = datasets['fit'], metrics['fit'][variant]
            keep, groups = training_rows(d, m)
            for width in ([24, 19] if args.baseline_ablations else [24]):
                guard(); assert fits < 12
                params = dict(PARAMS); params['n_estimators'] = winner['best_iteration']
                model = lgb.LGBMRanker(**params, **CONFIGS[winner['config_id']])
                fits += 1; started = time.perf_counter()
                model.fit(d['X'][keep, :width], m.y[keep], group=groups, callbacks=[guard])
                key = f'{variant}_{width}features'
                model.booster_.save_model(str(cache/f'{key}.txt'))
                row = dict(phase='refit', variant=variant, width=width,
                           best_iteration=winner['best_iteration'], fit_seconds=time.perf_counter()-started)
                for name in ('eval', 'later'):
                    guard(); scores = model.predict(datasets[name]['X'][:, :width])
                    row[name] = {v: metric.rank(scores)[0] for v, metric in metrics[name].items()}
                    np.save(cache/f'{key}_{name}_scores.npy', scores)
                    write(out/f'{key}_{name}_predictions.json', metrics[name]['all80'].predictions(scores))
                results.append(row); write(out/'results.partial.json', results); print(json.dumps(row), flush=True)
        write(out/'results.json', dict(status='complete', results=results, fits=fits,
            wall_seconds=time.perf_counter()-began, peak_rss_bytes=rss(), lightgbm_version=lgb.__version__,
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    except Exception as error:
        write(out/'failure.json', dict(status='incomplete', error=type(error).__name__, detail=str(error),
            fits=fits, wall_seconds=time.perf_counter()-began, peak_rss_bytes=rss()))
        raise


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
