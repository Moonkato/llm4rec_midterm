"""Full-history temporal companion; prepare only until separately authorized."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import hashlib,json,sqlite3,time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from threadpoolctl import threadpool_limits
from rolling_context_helpers import build_window,weekly_window
from run_deep_temporal import BASE,CONFIGS,PARAMS,Metric,full_truth,training_rows,rss,write
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
WINDOWS={'early':('2020-03-09','2020-04-13','2020-05-18'),
         'recent':('2020-03-09','2020-05-18','2020-06-22'),
         'later':('2020-03-09','2020-06-22','2020-07-27')}


def main():
    out,cache=BASE/'evidence/full_history_temporal',BASE/'cache/full_history_temporal'
    out.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise SystemExit('Preserve previous round')
    params=dict(PARAMS);params.update(n_estimators=417,n_jobs=1,lambdarank_norm=True)
    protocol={'windows_origin_cutoff_end':WINDOWS,'features':'Samebase19+weekly5, full historical originMarch9 for every snapshot,35day forecast' ,
        'primary':'pooled_early_recent','diagnostic':'recent_only','frozen_control':'v3 on EXACT SAME original full-history later features/candidates/fallback',
        'separate_reference':'Original v3 on105day history equals controlled comparator exactly',
        'target':'Full-universe top5_count80 before shortlist, numeric ties',
        'parameters':params,'config':CONFIGS[1],'fits':2,'threads':1,'wall_limit_seconds':300,'rss_limit_bytes':int(1.5*1024**3),
        'ranker_weighting':'lambdarank_norm=True; constant weight per query, NOT inverse query size. Pooled each query weight total_queries/(2*queries_in_snapshot); sums of query coefficients equal across snapshots. No claim exactly equal gradient norms.',
        'selection':'Fixed model/iteration, no grid. Evaluate only Jun22-Jul27, never overlapping May18-Jul27.',
        'fallback':'All comparisons share original105day historical unique-viewer popularity and cold flags',
        'limitations':'All observations previously explored.35day forecast diagnostic does not validate84day forecast; historical length35/70/105days differs by snapshot, as actual cumulative-history deployment.',
        'submissions':0}
    write(out/'protocol.json',protocol);began=time.perf_counter();fits=0;manifest={}
    def guard(env=None):
        if time.perf_counter()-began>300 or rss()>int(1.5*1024**3):raise RuntimeError('Temporal round budget exceeded')
    guard.order=-100;guard.before_iteration=True
    try:
        ds={};metrics={};preparation=[];db=sqlite3.connect('file:'+str(BASE/'cache/interactions.sqlite')+'?mode=ro',uri=True)
        db.execute('PRAGMA cache_size=-32000');db.execute('PRAGMA temp_store=FILE')
        for name,(origin,cut,end) in WINDOWS.items():
            guard()
            if name in ('early','later'):
                oldname='fit' if name=='early' else 'later'
                base=BASE/f'cache/context_features/{oldname}.npz';wp=BASE/f'cache/deep_weekly/{oldname}_row_features.npy'
                with np.load(base,allow_pickle=False) as z:d={k:z[k] for k in z.files}
                weekly=np.load(wp,allow_pickle=False)
            else:
                base=BASE/'cache/multiwindow/recent_fit.npz'
                with np.load(base,allow_pickle=False) as z:d={k:z[k] for k in z.files}
                weekly,used=weekly_window(db,d,origin+' 00:00:00',cut+' 00:00:00',end+' 00:00:00',BASE.parents[1]/'midterm/data/dataset11-30.csv',guard)
                wp=cache/f'{name}_weekly.npy';np.save(wp,weekly)
                preparation.append({'window':name,'rows':len(d['y']),'users':len(d['users']),'raw_sessions_used':used,'source':'existing full-history recent19'})
            assert d['X'].shape[1]==19 and weekly.shape==(len(d['y']),5)
            d['X']=np.concatenate([d['X'],weekly],axis=1);assert np.isfinite(d['X']).all()
            truths=full_truth(db,cut,end);metrics[name]={t:Metric(d,v) for t,v in truths.items()}
            assert np.array_equal(metrics[name]['all80'].y,d['y'])
            assert np.array_equal(metrics[name]['all80'].den,np.minimum(5,d['truth_counts']))
            ds[name]=d
            for p in (base,wp):manifest[str(p.relative_to(BASE))]=hashlib.sha256(p.read_bytes()).hexdigest()
        db.close();write(out/'preparation.json',preparation);write(out/'inputs.json',manifest)
        # Verify constant target cohort and full truth against original heldout snapshot.
        with np.load(BASE/'cache/context_features/later.npz',allow_pickle=False) as z:original={k:z[k] for k in z.files}
        assert np.array_equal(original['users'],ds['later']['users'])
        assert np.array_equal(original['truth_counts'],ds['later']['truth_counts'])
        full_scores=np.load(BASE/'cache/deep_temporal/top5_count80_24features_later_scores.npy')
        # Original scores are evaluated by original snapshot labels and separately stored full top-five truth.
        with sqlite3.connect('file:'+str(BASE/'cache/interactions.sqlite')+'?mode=ro',uri=True) as db2:truth=full_truth(db2,'2020-06-22','2020-07-27')
        original_metrics={t:Metric(original,v) for t,v in truth.items()}
        reference={t:m.rank(full_scores)[0] for t,m in original_metrics.items()}
        frozen=lgb.Booster(model_file=str(BASE/'cache/deep_temporal/top5_count80_24features.txt'))
        frozen_scores=frozen.predict(ds['later']['X'],num_threads=1)
        control={t:m.rank(frozen_scores)[0] for t,m in metrics['later'].items()}
        assert np.allclose(frozen_scores,full_scores,rtol=1e-10,atol=1e-10), 'Full-history control differs from v3'
        np.save(cache/'frozen_v3_full_history_later_scores.npy',frozen_scores)
        write(out/'frozen_v3_full_history_later_predictions.json',metrics['later']['all80'].predictions(frozen_scores))
        write(out/'references.json',{'frozen_v3_full_history_control':control,'original_v3_full_history_reference':reference,
            'controlled_cold_users':int(ds['later']['cold'].sum()),'original_cold_users':int(original['cold'].sum())})
        prepared={};group_info={}
        for name in ('early','recent'):
            keep,groups=training_rows(ds[name],metrics[name]['top5_count80'])
            prepared[name]=(ds[name]['X'][keep],metrics[name]['top5_count80'].y[keep],groups)
            group_info[name]={'input_queries':len(ds[name]['users']),'retained_queries':len(groups),'retained_rows':int(keep.sum())}
        write(out/'training_groups.json',group_info);results=[]
        for key,names in [('recent_only',['recent']),('pooled_early_recent',['early','recent'])]:
            guard();assert fits<2
            total_queries=sum(len(prepared[n][2]) for n in names)
            x=np.concatenate([prepared[n][0] for n in names]);y=np.concatenate([prepared[n][1] for n in names])
            groups=sum((prepared[n][2] for n in names),[])
            weights=np.concatenate([np.full(len(prepared[n][1]),total_queries/(len(names)*len(prepared[n][2]))) for n in names])
            assert sum(groups)==len(y)==len(weights)
            model=lgb.LGBMRanker(**params,**CONFIGS[1]);fits+=1;started=time.perf_counter()
            model.fit(x,y,group=groups,sample_weight=weights,callbacks=[guard]);fit_seconds=time.perf_counter()-started
            model.booster_.save_model(str(cache/f'{key}.txt'));scores=model.predict(ds['later']['X'],num_threads=1)
            values={t:m.rank(scores)[0] for t,m in metrics['later'].items()}
            row={'model':key,'fit_seconds':fit_seconds,'scores':values,'delta_vs_identical_full_history_control':{t:values[t]-control[t] for t in values},'fit_queries':len(groups),'fit_rows':len(y)}
            np.save(cache/f'{key}_later_scores.npy',scores);write(out/f'{key}_later_predictions.json',metrics['later']['all80'].predictions(scores))
            results.append(row);write(out/'results.partial.json',results);print(json.dumps(row),flush=True)
            del x,y,weights,model
        write(out/'results.json',{'status':'complete','results':results,'control':control,'separate_full_history_reference':reference,
            'fits':fits,'wall_seconds':time.perf_counter()-began,'peak_rss_bytes':rss(),
            'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'helper_sha256':hashlib.sha256((BASE/'tools/rolling_context_helpers.py').read_bytes()).hexdigest()})
    except Exception as error:
        write(out/'failure.json',{'status':'incomplete','error':type(error).__name__,'detail':str(error),'fits':fits,'wall_seconds':time.perf_counter()-began,'peak_rss_bytes':rss()});raise


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
