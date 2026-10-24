"""Three authorized fixed regularity ablations; no tuning or submissions."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import numpy as np
import lightgbm as lgb
from threadpoolctl import threadpool_limits
from run_deep_temporal import BASE, WINDOWS, CONFIGS, PARAMS, Metric, full_truth, training_rows, rss, write
# Imported helper module declares its own thread defaults; override them for this round.
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'


def main():
    out, cache = BASE/'evidence/regularity_round', BASE/'cache/regularity_round'
    out.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        raise SystemExit('Preserve previous round')
    if not (BASE/'evidence/regularity_features/results.json').exists():
        raise SystemExit('Feature build is not complete')
    feature_manifest = BASE/'evidence/regularity_features/protocol.json'
    feature_protocol = json.loads(feature_manifest.read_text())
    groups = feature_protocol['groups']
    variants = {'regularity_trend': sorted(groups['regularity']+groups['trend']),
                'completion_share': sorted(groups['intensity']+groups['completion'])}
    variants['all'] = sorted(variants['regularity_trend']+variants['completion_share'])
    for cols in variants.values():
        assert cols and min(cols) >= 0 and len(set(cols)) == len(cols)
    params = dict(PARAMS); params.update(n_estimators=417, n_jobs=1)
    protocol = dict(target='top5_count80', config=CONFIGS[1], parameters=params,
        base_features=24, extra_column_groups=variants, source_feature_protocol=str(feature_manifest.relative_to(BASE)),
        fit_window=WINDOWS['fit'], evaluation_windows={n: WINDOWS[n] for n in ('eval', 'later')},
        max_fits=3, wall_limit_seconds=600, rss_limit_bytes=int(1.5*1024**3), threads=1,
        baseline='Frozen v3 deep_temporal/top5_count80_24features',
        selection='Three prespecified ablations, fixed config and417trees; no hyperparameter tuning',
        primary_candidate='all', ablation_role='Other two variants explanatory only; no selection by outer scores',
        truth='Full universe all80 and top5 by qualifying airing count before shortlist; numeric ID ties',
        validation='Reused overlapping exploratory windows, not independent confirmation',
        resource_guard='Every boosting iteration and between preprocessing phases; monitored RSS, not OS allocation ceiling',
        submissions=0)
    write(out/'protocol.json', protocol)
    began=time.perf_counter(); fit_count=0; results=[]; hashes={}
    def guard(env=None):
        if time.perf_counter()-began > 600 or rss() > int(1.5*1024**3):
            raise RuntimeError('Regularity round time or memory budget exceeded')
    guard.order=-100; guard.before_iteration=True
    try:
        datasets={}; extras={}; metrics={}
        db=sqlite3.connect('file:'+str(BASE/'cache/interactions.sqlite')+'?mode=ro', uri=True)
        for name in ('fit','eval','later'):
            guard()
            paths=[BASE/f'cache/context_features/{name}.npz',
                   BASE/f'cache/deep_weekly/{name}_row_features.npy',
                   BASE/f'cache/regularity_features/{name}_row_features.npy']
            with np.load(paths[0],allow_pickle=False) as z: d={k:z[k] for k in z.files}
            weekly=np.load(paths[1],allow_pickle=False); extra=np.load(paths[2],allow_pickle=False)
            assert d['X'].shape[1]==19 and weekly.shape==(len(d['y']),5)
            assert extra.ndim==2 and extra.shape[0]==len(d['y']) and np.isfinite(extra).all()
            assert max(variants['all']) < extra.shape[1]
            d['X']=np.concatenate([d['X'],weekly],axis=1)
            truth=full_truth(db,*WINDOWS[name]); metrics[name]={t:Metric(d,v) for t,v in truth.items()}
            assert np.array_equal(metrics[name]['all80'].y,d['y'])
            assert np.array_equal(metrics[name]['all80'].den,np.minimum(5,d['truth_counts']))
            datasets[name]=d; extras[name]=extra
            for path in paths:hashes[str(path.relative_to(BASE))]=hashlib.sha256(path.read_bytes()).hexdigest()
        db.close(); write(out/'inputs.json',hashes)
        baseline={}
        for name in ('eval','later'):
            scores=np.load(BASE/f'cache/deep_temporal/top5_count80_24features_{name}_scores.npy')
            baseline[name]={t:m.rank(scores)[0] for t,m in metrics[name].items()}
        write(out/'baseline.json',baseline)
        keep,queries=training_rows(datasets['fit'],metrics['fit']['top5_count80'])
        for variant,cols in variants.items():
            guard(); assert fit_count<3
            x=np.concatenate([datasets['fit']['X'],extras['fit'][:,cols]],axis=1)
            model=lgb.LGBMRanker(**params,**CONFIGS[1]); fit_count+=1; started=time.perf_counter()
            model.fit(x[keep],metrics['fit']['top5_count80'].y[keep],group=queries,callbacks=[guard]); del x
            row=dict(variant=variant,features=24+len(cols),fit_seconds=time.perf_counter()-started)
            model.booster_.save_model(str(cache/f'{variant}.txt'))
            for name in ('eval','later'):
                guard(); x=np.concatenate([datasets[name]['X'],extras[name][:,cols]],axis=1)
                scores=model.predict(x); del x
                values={t:m.rank(scores)[0] for t,m in metrics[name].items()}
                row[name]={'scores':values,'delta_vs_v3':{t:values[t]-baseline[name][t] for t in values}}
                np.save(cache/f'{variant}_{name}_scores.npy',scores)
                write(out/f'{variant}_{name}_predictions.json',metrics[name]['all80'].predictions(scores))
            results.append(row); write(out/'results.partial.json',results); print(json.dumps(row),flush=True)
        write(out/'results.json',dict(status='complete',results=results,baseline=baseline,fits=fit_count,
            wall_seconds=time.perf_counter()-began,peak_rss_bytes=rss(),lightgbm_version=lgb.__version__,
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    except Exception as error:
        write(out/'failure.json',dict(status='incomplete',error=type(error).__name__,detail=str(error),
            fits=fit_count,wall_seconds=time.perf_counter()-began,peak_rss_bytes=rss()))
        raise


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
