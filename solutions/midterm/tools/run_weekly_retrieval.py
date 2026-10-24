"""One authorized fixed refit after weekly-schedule candidate expansion."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import hashlib,json,sqlite3,time
from pathlib import Path
import numpy as np
from scipy import sparse
import lightgbm as lgb
from threadpoolctl import threadpool_limits
from build_context_features import build_window
from run_deep_temporal import BASE,WINDOWS,CONFIGS,PARAMS,Metric,full_truth,training_rows,rss,write
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'


def oracle(metric):
    offsets=metric.d['group_offsets']; hits=np.add.reduceat(metric.y,offsets[:-1])
    values=np.divide(np.minimum(5,hits),metric.den,out=np.zeros(len(hits)),where=metric.den>0)
    return {'oracle_map5':float(values.mean()),'retrieved_relevant_pairs':int(hits.sum())}


def main():
    out,cache=BASE/'evidence/weekly_retrieval',BASE/'cache/weekly_retrieval'
    out.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise SystemExit('Preserve previous round')
    if not (BASE/'evidence/weekly_candidates/results.json').exists():raise SystemExit('Weekly candidates not complete')
    params=dict(PARAMS);params.update(n_estimators=417,n_jobs=1)
    protocol={'primary':'retrained_expanded24','diagnostic':'frozen_v3_expanded24','fits':1,'parameters':params,'config':CONFIGS[1],
      'expansion':'Union existing candidates with top30 positive historical all-weekly and top30 recent28-weekly forecast expected seconds; no truth-based selection',
      'features':'Identical base19 plus mapped historical weekly5, no new regularity features','target':'top5_count80 in full universe before shortlist',
      'windows':{n:WINDOWS[n] for n in ('fit','eval','later')},'fallback':'Unchanged original cold historical popularity',
      'limits':{'wall_seconds':600,'rss_bytes':int(1.5*1024**3),'threads':1},'selection':'Fixed configuration and iteration, no tuning or blend search',
      'validation':'Previously explored overlapping windows; diagnostic not independent confirmation','submissions':0}
    write(out/'protocol.json',protocol);began=time.perf_counter();fits=0;results=[];manifest={}
    def guard(env=None):
        if time.perf_counter()-began>600 or rss()>int(1.5*1024**3):raise RuntimeError('Prespecified resource budget exceeded')
    guard.order=-100;guard.before_iteration=True
    try:
        datasets={};metrics={};original_metrics={};coverage={}
        db=sqlite3.connect('file:'+str(BASE/'cache/interactions.sqlite')+'?mode=ro',uri=True)
        db.execute('PRAGMA cache_size=-32000');db.execute('PRAGMA temp_store=FILE')
        for name in ('fit','eval','later'):
            guard();source=BASE/'cache/weekly_candidates';extra_path=BASE/f'evidence/weekly_candidates/{name}_extra_candidates.json'
            extras=json.loads(extra_path.read_text());start,end=WINDOWS[name]
            build_window(db,name,start+' 00:00:00',end+' 00:00:00',name=='fit',cache,out,extra_candidates=extras)
            guard()
            with np.load(cache/f'{name}.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
            with np.load(BASE/f'cache/context_features/{name}.npz',allow_pickle=False) as z:original={k:z[k] for k in z.files}
            with np.load(source/f'{name}_catalog.npz',allow_pickle=False) as z:catalog={k:z[k] for k in z.files}
            assert np.array_equal(d['users'],original['users']) and np.array_equal(d['item_ids'],original['item_ids'])
            assert np.array_equal(d['truth_counts'],original['truth_counts']) and np.array_equal(d['cold'],original['cold'])
            assert np.array_equal(d['fallback_item_indices'],original['fallback_item_indices'])
            user_lookup={str(u):i for i,u in enumerate(catalog['users'])};item_lookup={str(i):j for j,i in enumerate(catalog['item_ids'])}
            matrices={key:sparse.load_npz(source/f'{name}_{key}.npz') for key in ('expected','recent_expected','mean_fraction','activity')}
            weekly=np.zeros((len(d['y']),5),dtype=np.float32)
            forecast_days=float(np.asarray(catalog['forecast_days']).item())
            old_weekly=np.load(BASE/f'cache/deep_weekly/{name}_row_features.npy')
            old_rows_retained=0
            for n,u in enumerate(d['users']):
                a,b=d['group_offsets'][n:n+2];oa,ob=original['group_offsets'][n:n+2]
                ix=np.array([item_lookup[str(d['item_ids'][i])] for i in d['row_item_indices'][a:b]])
                j=user_lookup[str(u)];v=matrices['expected'][j,ix].toarray().ravel();r=matrices['recent_expected'][j,ix].toarray().ravel()
                weekly[a:b,0]=v/(60*forecast_days)
                weekly[a:b,1]=matrices['mean_fraction'][j,ix].toarray().ravel()/catalog['airings'][ix]
                weekly[a:b,2]=r/(60*forecast_days)
                weekly[a:b,3]=matrices['activity'][j,ix].toarray().ravel()/catalog['totaldur'][ix]
                weekly[a:b,4]=np.minimum(10,np.divide(r,v,out=np.zeros_like(r),where=v>0))
                expanded={int(k):a+t for t,k in enumerate(d['row_item_indices'][a:b])}
                old_indices=np.array([expanded[int(k)] for k in original['row_item_indices'][oa:ob]])
                assert np.allclose(d['X'][old_indices],original['X'][oa:ob],rtol=1e-5,atol=1e-6)
                assert np.allclose(weekly[old_indices],old_weekly[oa:ob],rtol=1e-6,atol=1e-6), 'Mapped weekly values differ on old candidates'
                old_rows_retained+=len(old_indices)
            assert np.isfinite(weekly).all() and np.all(weekly>=0)
            np.save(cache/f'{name}_weekly.npy',weekly);d['X']=np.concatenate([d['X'],weekly],axis=1)
            truths=full_truth(db,start,end)
            metrics[name]={v:Metric(d,t) for v,t in truths.items()};original_metrics[name]={v:Metric(original,t) for v,t in truths.items()}
            assert np.array_equal(metrics[name]['all80'].y,d['y'])
            coverage[name]={'old_rows':old_rows_retained,'expanded_rows':len(d['y']),'targets':{}}
            for target,truth in truths.items():
                old,new=oracle(original_metrics[name][target]),oracle(metrics[name][target])
                total=sum(len(truth.get(str(u),set())) for u in d['users'])
                for vals in (old,new):vals['full_truth_pairs']=total;vals['micro_candidate_recall']=vals['retrieved_relevant_pairs']/total if total else 0
                coverage[name]['targets'][target]={'original':old,'expanded':new}
            datasets[name]=d
            for path in [extra_path,source/f'{name}_catalog.npz',*[source/f'{name}_{k}.npz' for k in matrices]]:
                manifest[str(path.relative_to(BASE))]=hashlib.sha256(path.read_bytes()).hexdigest()
            del matrices,original,old_weekly
        db.close();write(out/'coverage.json',coverage);write(out/'inputs.json',manifest)
        baseline={}
        for name in ('eval','later'):
            s=np.load(BASE/f'cache/deep_temporal/top5_count80_24features_{name}_scores.npy')
            baseline[name]={t:m.rank(s)[0] for t,m in original_metrics[name].items()}
        write(out/'baseline.json',baseline)
        frozen=lgb.Booster(model_file=str(BASE/'cache/deep_temporal/top5_count80_24features.txt'))
        for key,model in [('frozen_v3_expanded24',frozen),('retrained_expanded24',None)]:
            guard();row={'model':key}
            if model is None:
                keep,groups=training_rows(datasets['fit'],metrics['fit']['top5_count80']);assert fits==0
                row['retained_training_groups']=len(groups);row['retained_training_rows']=int(keep.sum())
                model=lgb.LGBMRanker(**params,**CONFIGS[1]);fits+=1;t=time.perf_counter()
                model.fit(datasets['fit']['X'][keep],metrics['fit']['top5_count80'].y[keep],group=groups,callbacks=[guard])
                row['fit_seconds']=time.perf_counter()-t;model.booster_.save_model(str(cache/f'{key}.txt'))
            for name in ('eval','later'):
                guard();scores=model.predict(datasets[name]['X'],num_threads=1);values={t:m.rank(scores)[0] for t,m in metrics[name].items()}
                row[name]={'scores':values,'delta_vs_original_v3':{t:values[t]-baseline[name][t] for t in values}}
                np.save(cache/f'{key}_{name}_scores.npy',scores);write(out/f'{key}_{name}_predictions.json',metrics[name]['all80'].predictions(scores))
            results.append(row);write(out/'results.partial.json',results);print(json.dumps(row),flush=True)
        write(out/'results.json',{'status':'complete','results':results,'fits':fits,'wall_seconds':time.perf_counter()-began,
            'peak_rss_bytes':rss(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    except Exception as error:
        write(out/'failure.json',{'status':'incomplete','error':type(error).__name__,'detail':str(error),'fits':fits,'wall_seconds':time.perf_counter()-began,'peak_rss_bytes':rss()});raise


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
