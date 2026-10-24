"""Preserve v3 and add two train-fitted categorical IDs. Requires explicit --run."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,sqlite3,time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from threadpoolctl import threadpool_limits
from run_deep_temporal import BASE,WINDOWS,PARAMS,CONFIGS,Metric,full_truth,training_rows,rss,write
from run_objective_diversity import reciprocal_ranks
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true',help='Execute the single explicitly authorized fit')
    args=parser.parse_args()
    params=dict(PARAMS);params.update(n_estimators=417,n_jobs=1,cat_smooth=20,min_data_per_group=100)
    plan={'model':'v3_24_plus_user_item_identity26','base_features':'EXACT original19 plus successful weekly5; no OOF extras',
        'parameters':params,'config':CONFIGS[1],'categorical_feature_indices':[24,25],
        'categorical_encoding':'Contiguous zero-based user/item codes fitted ONLY from retained TRAIN rows; unseen=-1; original IDs retained as strings',
        'fit_window':WINDOWS['fit'],'target':'Original hardtop5_count80 full universe, numeric ties before shortlist',
        'query_filter':'Original v3 training_rows positive-and-negative groups; same rows and query boundaries',
        'baseline_checks':'Original24 input hashes match deep_temporal/inputs.json; frozenv3 predictions exactly equal stored scores for BOTH windows',
        'outputs':'Standalone and fixed equalRRF60 with original event model; compare standalone v3 and fusion v4',
        'primary_decision':'Final prespecified2x2 combiner assessed separately by parent; these diagnostics do not select a different configuration',
        'limits':{'fits':1,'wall_seconds':300,'rss_bytes':int(1.5*1024**3),'threads':1},
        'caveat':'Repeatedly explored overlapping temporal windows; not untouched test, no hyperparameter or blend search','submissions':0}
    if not args.run:
        print(json.dumps({'status':'source_ready_not_executed','plan':plan},indent=2));return
    out,cache=BASE/'evidence/final_identity',BASE/'cache/final_identity'
    out.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise SystemExit('Preserve previous experiment')
    write(out/'protocol.json',plan);began=time.perf_counter();fits=0
    def guard(env=None):
        if time.perf_counter()-began>300 or rss()>int(1.5*1024**3):raise RuntimeError('Final identity budget exceeded')
    guard.order=-100;guard.before_iteration=True
    try:
        ds={};metrics={};manifest={};oldhash=json.loads((BASE/'evidence/deep_temporal/inputs.json').read_text())
        db=sqlite3.connect('file:'+str(BASE/'cache/interactions.sqlite')+'?mode=ro',uri=True)
        db.execute('PRAGMA cache_size=-32000');db.execute('PRAGMA temp_store=FILE')
        for name in ('fit','eval','later'):
            guard();paths=[BASE/f'cache/context_features/{name}.npz',BASE/f'cache/deep_weekly/{name}_row_features.npy']
            with np.load(paths[0],allow_pickle=False) as z:d={k:z[k] for k in z.files}
            weekly=np.load(paths[1],allow_pickle=False)
            assert d['X'].shape[1]==19 and weekly.shape==(len(d['y']),5)
            d['X']=np.concatenate([d['X'],weekly],axis=1);assert d['X'].shape[1]==24 and np.isfinite(d['X']).all()
            truth=full_truth(db,*WINDOWS[name]);metrics[name]={t:Metric(d,v) for t,v in truth.items()}
            assert np.array_equal(metrics[name]['all80'].y,d['y'])
            assert np.array_equal(metrics[name]['all80'].den,np.minimum(5,d['truth_counts']))
            ds[name]=d
            for p in paths:
                key=str(p.relative_to(BASE));manifest[key]=hashlib.sha256(p.read_bytes()).hexdigest();assert manifest[key]==oldhash[key], 'Original24 source changed'
        db.close();write(out/'inputs.json',manifest)
        frozen=lgb.Booster(model_file=str(BASE/'cache/deep_temporal/top5_count80_24features.txt'))
        oldscores={};events={};baseline={}
        for name in ('eval','later'):
            oldscores[name]=np.load(BASE/f'cache/deep_temporal/top5_count80_24features_{name}_scores.npy')
            reproduced=frozen.predict(ds[name]['X'],num_threads=1)
            assert np.array_equal(reproduced,oldscores[name]), 'Frozen v3 exact prediction parity failed'
            events[name]=np.load(BASE/f'cache/airing_model/{name}_expected_counts.npy')
            fused=(reciprocal_ranks(ds[name],oldscores[name])+reciprocal_ranks(ds[name],events[name]))/2
            baseline[name]={'v3':{t:m.rank(oldscores[name])[0] for t,m in metrics[name].items()},
                            'v4':{t:m.rank(fused)[0] for t,m in metrics[name].items()}}
        write(out/'baseline.json',baseline)
        fit=ds['fit'];keep,groups=training_rows(fit,metrics['fit']['top5_count80'])
        group_ids=metrics['fit']['all80'].g
        retained_users={str(fit['users'][g]) for g in group_ids[keep]}
        retained_items={str(fit['item_ids'][i]) for i in fit['row_item_indices'][keep]}
        user_codes={u:n for n,u in enumerate(sorted(retained_users,key=int))}
        item_codes={i:n for n,i in enumerate(sorted(retained_items,key=int))}
        write(cache/'categorical_encoders.json',{'users':user_codes,'items':item_codes,'unknown_code':-1})
        matrices={};encoding_stats={}
        for name,d in ds.items():
            row_users=np.array([user_codes.get(str(d['users'][g]),-1) for g in metrics[name]['all80'].g],dtype=np.int32)
            row_items=np.array([item_codes.get(str(d['item_ids'][i]),-1) for i in d['row_item_indices']],dtype=np.int32)
            matrices[name]=np.column_stack((d['X'],row_users,row_items)).astype(np.float32)
            assert np.array_equal(matrices[name][:,:24],d['X'])
            encoding_stats[name]={'rows':len(row_users),'unknown_user_rows':int((row_users<0).sum()),'unknown_item_rows':int((row_items<0).sum())}
        assert np.all(matrices['fit'][keep,24:]>=0)
        write(out/'encoding.json',{'user_categories':len(user_codes),'item_categories':len(item_codes),'by_window':encoding_stats,
            'training_rows':int(keep.sum()),'training_groups':len(groups),'exact_baseline_parity_both_windows':True})
        guard();model=lgb.LGBMRanker(**params,**CONFIGS[1]);fits+=1;started=time.perf_counter()
        model.fit(matrices['fit'][keep],metrics['fit']['top5_count80'].y[keep],group=groups,categorical_feature=[24,25],callbacks=[guard])
        fit_seconds=time.perf_counter()-started;model.booster_.save_model(str(cache/'identity_ranker.txt'));results={}
        for name in ('eval','later'):
            guard();scores=model.predict(matrices[name],num_threads=1)
            fused=(reciprocal_ranks(ds[name],scores)+reciprocal_ranks(ds[name],events[name]))/2
            results[name]={}
            for key,value,ref in [('standalone',scores,'v3'),('old_event_rrf60',fused,'v4')]:
                vals={t:m.rank(value)[0] for t,m in metrics[name].items()}
                results[name][key]={'scores':vals,'delta_vs_'+ref:{t:vals[t]-baseline[name][ref][t] for t in vals}}
                np.save(cache/f'{key}_{name}_scores.npy',value);write(out/f'{key}_{name}_predictions.json',metrics[name]['all80'].predictions(value))
            print(json.dumps({'window':name,'results':results[name]}),flush=True)
        write(out/'results.json',{'status':'complete','results':results,'baseline':baseline,'fits':fits,'fit_seconds':fit_seconds,
            'wall_seconds':time.perf_counter()-began,'peak_rss_bytes':rss(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    except Exception as error:
        write(out/'failure.json',{'status':'incomplete','error':type(error).__name__,'detail':str(error),'fits':fits,
            'wall_seconds':time.perf_counter()-began,'peak_rss_bytes':rss()});raise


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
