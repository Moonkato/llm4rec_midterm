"""Fixed v3 LambdaRank with six user-OOF event features; one fit only."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from threadpoolctl import threadpool_limits
from run_deep_temporal import BASE, PARAMS, CONFIGS, Metric, full_truth, training_rows, rss


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    if not parser.parse_args().run:
        raise SystemExit('Prepared only; explicit --run required')
    marker = BASE/'evidence/event_oof/results.json'
    oof_run = json.loads(marker.read_text())
    assert oof_run['status'] == 'complete' and oof_run['fits'] == 3
    oof_manifest = {r['window']: r for r in oof_run['results']}
    original_hashes = json.loads((BASE/'evidence/deep_temporal/inputs.json').read_text())
    out, cache = BASE/'evidence/event_stack', BASE/'cache/event_stack'
    if (out/'protocol.json').exists():
        raise SystemExit('Preserve previous run')
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    params = dict(PARAMS)
    params.update(n_estimators=417, n_jobs=1)
    protocol = {
        'status': 'fixed protocol before fit', 'fits': 1,
        'features': 'original v3 base19+weekly5, then six user-OOF event summaries',
        'target': 'same original full-universe hard top5_count80 with numeric ID ties',
        'selection': 'No adaptive objective choice from tie controls; no hyperparameter or fusion grid',
        'parameters': params, 'configuration': CONFIGS[1],
        'training': 'Apr13 history cutoff; labels Apr13-May18 only',
        'event_features': 'Fit rows scored only by event model excluding the entire user; eval/later by frozen original event model trained beforeMay18',
        'baseline': 'frozen v4 equal RRF60; same candidates and cold fallback',
        'primary': 'standalone 30-feature ranker; both temporal windows against v4',
        'diagnostics': 'all80 plus independent exact tie-marginal expected AP and paired bootstrap',
        'budget': 'one fit,300seconds,1.5GiB,1CPU', 'submissions': 0,
        'source_sha256': digest(Path(__file__)),
        'oof_results_sha256': digest(marker),
        'limitations': 'Reused overlapping windows; no untouched validation or guarantee about unknown Kaggle target formula',
    }
    write(out/'protocol.json', protocol)

    def guard(env=None):
        if time.perf_counter()-began > 300 or rss() > 1.5*1024**3:
            raise RuntimeError('Stack budget exceeded')
    guard.order = -100
    guard.before_iteration = True
    windows = [('fit','2020-04-13','2020-05-18'),
               ('eval','2020-05-18','2020-07-27'),
               ('later','2020-06-22','2020-07-27')]
    data, metrics, hashes = {}, {}, {}
    with sqlite3.connect('file:'+str(BASE/'cache/interactions.sqlite')+'?mode=ro', uri=True) as db:
        db.execute('PRAGMA cache_size=-32000')
        db.execute('PRAGMA temp_store=FILE')
        for name, start, end in windows:
            guard()
            base = BASE/f'cache/context_features/{name}.npz'
            weekly = BASE/f'cache/deep_weekly/{name}_row_features.npy'
            extra = BASE/f'cache/event_oof/{name}_features.npy'
            with np.load(base, allow_pickle=False) as z:
                d = {k: z[k] for k in z.files}
            w, e = np.load(weekly), np.load(extra)
            assert w.shape == (len(d['y']),5) and e.shape == (len(d['y']),6)
            assert d['X'].shape[1] == 19
            d['X'] = np.concatenate([d['X'],w,e],axis=1)
            assert np.isfinite(d['X']).all()
            truth = full_truth(db, start, end)
            metrics[name] = {t: Metric(d,v) for t,v in truth.items()}
            assert np.array_equal(metrics[name]['all80'].y, d['y'])
            assert np.array_equal(metrics[name]['all80'].den, np.minimum(5,d['truth_counts']))
            data[name] = d
            for p in (base, weekly, extra):
                hashes[str(p.relative_to(BASE))] = digest(p)
            assert hashes[str(base.relative_to(BASE))] == original_hashes[str(base.relative_to(BASE))]
            assert hashes[str(weekly.relative_to(BASE))] == original_hashes[str(weekly.relative_to(BASE))]
            assert hashes[str(base.relative_to(BASE))] == oof_manifest[name]['source_sha256']
            assert hashes[str(extra.relative_to(BASE))] == oof_manifest[name]['feature_sha256']
    write(out/'inputs.json', hashes)
    frozen = lgb.Booster(model_file=str(BASE/'cache/deep_temporal/top5_count80_24features.txt'))
    for name in ('eval','later'):
        actual = frozen.predict(data[name]['X'][:,:24], num_threads=1)
        saved = np.load(BASE/f'cache/deep_temporal/top5_count80_24features_{name}_scores.npy')
        assert np.allclose(actual,saved,rtol=1e-10,atol=1e-10), 'v3 feature alignment changed'
    del frozen
    keep, groups = training_rows(data['fit'],metrics['fit']['top5_count80'])
    write(out/'training.json', {'queries':len(groups), 'rows':int(keep.sum()),
                              'features':30, 'same_v3_group_selection':True})
    model = lgb.LGBMRanker(**params, **CONFIGS[1])
    guard()
    fit_start = time.perf_counter()
    model.fit(data['fit']['X'][keep],metrics['fit']['top5_count80'].y[keep],
              group=groups, callbacks=[guard])
    fit_seconds = time.perf_counter()-fit_start
    model.booster_.save_model(str(cache/'model.txt'))
    results = []
    baseline = {r['window']:r['equal_rrf60_v3'] for r in
                json.loads((BASE/'evidence/airing_model/results.json').read_text())['results']}
    for name in ('eval','later'):
        guard()
        scores = model.predict(data[name]['X'], num_threads=1)
        np.save(cache/f'{name}_scores.npy',scores)
        pred = metrics[name]['all80'].predictions(scores)
        write(out/f'{name}_predictions.json',pred)
        values = {t:m.rank(scores)[0] for t,m in metrics[name].items()}
        row = {'window':name,'scores':values,'v4':baseline[name],
               'delta_vs_v4':{t:values[t]-baseline[name][t] for t in values}}
        results.append(row)
        print(json.dumps(row),flush=True)
    write(out/'results.json', {'status':'complete','results':results,'fits':1,
          'fit_seconds':fit_seconds,'wall_seconds':time.perf_counter()-began,
          'peak_rss_bytes':rss(),'submissions':0,'source_sha256':digest(Path(__file__))})


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
