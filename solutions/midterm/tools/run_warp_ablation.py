"""One prespecified WARP scalar-feature ablation; root approval required to run."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import hashlib,json,sqlite3,time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from threadpoolctl import threadpool_limits
from run_deep_temporal import BASE,WINDOWS,CONFIGS,PARAMS,Metric,full_truth,training_rows,rss,write


def ap_credit(metric,scores):
    """Additive seen/unseen credit of ordinary AP; no segment renormalization."""
    _,rows=metric.rank(scores);count=len(metric.d['users'])
    seen=np.zeros(count);unseen=np.zeros(count);hits=np.zeros(count,dtype=int);ranks=np.zeros(count,dtype=int)
    for row in rows:
        g=metric.g[row];ranks[g]+=1
        if metric.y[row]:
            hits[g]+=1
            if metric.den[g]:
                credit=hits[g]/ranks[g]/metric.den[g]
                (seen if metric.d['X'][row,0]>0 else unseen)[g]+=credit
    assert np.isclose((seen+unseen).mean(),metric.rank(scores)[0])
    return {'seen':seen,'unseen':unseen,'all':seen+unseen}


def paired_bootstrap(delta):
    rng=np.random.default_rng(20261023)
    means=np.concatenate([delta[rng.integers(0,len(delta),size=(100,len(delta)))].mean(axis=1) for _ in range(50)])
    return {'mean_delta':float(delta.mean()),'paired_user_percentile95':np.quantile(means,[.025,.975]).tolist(),
            'users_improved':int((delta>0).sum()),'users_worsened':int((delta<0).sum()),'users_equal':int((delta==0).sum())}


def main():
    out,cache=BASE/'evidence/warp_ablation',BASE/'cache/warp_ablation'
    cache.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise SystemExit('Preserve previous ablation')
    marker=BASE/'evidence/warp_features/results.json'
    if not marker.exists():raise SystemExit('WARP feature build not complete')
    plan=json.loads((out/'plan.json').read_text())
    params=dict(PARAMS);params.update(n_estimators=417,n_jobs=2)
    feature_protocol=json.loads((BASE/'evidence/warp_features/protocol.json').read_text())
    protocol=dict(plan,parameters=params,config=CONFIGS[1],feature_protocol=feature_protocol,
                  feature_completion_sha256=hashlib.sha256(marker.read_bytes()).hexdigest(),status='running')
    write(out/'protocol.json',protocol);began=time.perf_counter();fits=0;manifest={}
    def guard(env=None):
        if time.perf_counter()-began>600 or rss()>2*1024**3:raise RuntimeError('WARP ablation resource budget exceeded')
    guard.order=-100;guard.before_iteration=True
    try:
        ds={};metrics={};db=sqlite3.connect('file:'+str(BASE/'cache/interactions.sqlite')+'?mode=ro',uri=True)
        db.execute('PRAGMA temp_store=FILE');db.execute('PRAGMA cache_size=-32000')
        for name in ('fit','eval','later'):
            guard();paths=[BASE/f'cache/context_features/{name}.npz',BASE/f'cache/deep_weekly/{name}_row_features.npy',BASE/f'cache/warp_features/{name}_row_features.npy']
            with np.load(paths[0],allow_pickle=False) as z:d={k:z[k] for k in z.files}
            weekly=np.load(paths[1],allow_pickle=False);warp=np.load(paths[2],allow_pickle=False)
            assert d['X'].shape[1]==19 and weekly.shape==(len(d['y']),5) and warp.shape==(len(d['y']),8)
            assert np.isfinite(warp).all()
            d['X']=np.concatenate([d['X'],weekly,warp],axis=1);assert np.isfinite(d['X']).all()
            truths=full_truth(db,*WINDOWS[name]);metrics[name]={t:Metric(d,v) for t,v in truths.items()}
            assert np.array_equal(metrics[name]['all80'].y,d['y'])
            assert np.array_equal(metrics[name]['all80'].den,np.minimum(5,d['truth_counts']))
            ds[name]=d
            for p in paths:manifest[str(p.relative_to(BASE))]=hashlib.sha256(p.read_bytes()).hexdigest()
        db.close();write(out/'inputs.json',manifest)
        keep,groups=training_rows(ds['fit'],metrics['fit']['top5_count80'])
        guard();model=lgb.LGBMRanker(**params,**CONFIGS[1]);started=time.perf_counter();fits+=1
        model.fit(ds['fit']['X'][keep],metrics['fit']['top5_count80'].y[keep],group=groups,callbacks=[guard])
        fit_seconds=time.perf_counter()-started;model.booster_.save_model(str(cache/'v3_plus_warp32.txt'))
        results={}
        for name in ('eval','later'):
            guard();scores=model.predict(ds[name]['X'],num_threads=2)
            baseline=np.load(BASE/f'cache/deep_temporal/top5_count80_24features_{name}_scores.npy')
            np.save(cache/f'v3_plus_warp32_{name}_scores.npy',scores)
            write(out/f'v3_plus_warp32_{name}_predictions.json',metrics[name]['all80'].predictions(scores))
            result={}
            for target,metric in metrics[name].items():
                new=ap_credit(metric,scores);old=ap_credit(metric,baseline)
                result[target]={'baseline_map5':float(old['all'].mean()),'candidate_map5':float(new['all'].mean()),
                    'paired_bootstrap':paired_bootstrap(new['all']-old['all']),
                    'additive_ap_credit':{s:{'baseline':float(old[s].mean()),'candidate':float(new[s].mean()),
                        'delta':float((new[s]-old[s]).mean())} for s in ('seen','unseen')}}
            results[name]=result;write(out/'results.partial.json',results);print(json.dumps({'window':name,'results':result}),flush=True)
        success=(results['eval']['top5_count80']['paired_bootstrap']['mean_delta']>=.01 and results['later']['top5_count80']['paired_bootstrap']['mean_delta']>=0)
        write(out/'results.json',{'status':'complete','primary':'v3_plus_warp32','results':results,
            'prespecified_point_estimate_success_rule_met':success,'fits':fits,'fit_seconds':fit_seconds,
            'fit_rows':int(keep.sum()),'fit_groups':len(groups),'bootstrap_replicates':5000,'bootstrap_seed':20261023,
            'slice_interpretation':'Seen/unseen are additive shares of ordinary full-denominator AP credit, NOT independently normalized segment MAP. Seen iff historical count30>0.',
            'caveat':'Reused windows, conditional paired-user intervals, not adjusted for selection or dependence; not Kaggle confidence bounds.',
            'wall_seconds':time.perf_counter()-began,'peak_rss_bytes':rss(),'lightgbm_version':lgb.__version__,
            'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    except Exception as error:
        write(out/'failure.json',{'status':'incomplete','error':type(error).__name__,'detail':str(error),
            'fits':fits,'wall_seconds':time.perf_counter()-began,'peak_rss_bytes':rss()});raise


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
