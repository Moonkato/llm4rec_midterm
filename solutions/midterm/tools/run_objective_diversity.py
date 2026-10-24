"""Two fixed objective variants and prespecified equal-rank fusions; no tuning."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import hashlib,json,sqlite3,time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from threadpoolctl import threadpool_limits
from run_deep_temporal import BASE,WINDOWS,CONFIGS,PARAMS,Metric,full_truth,training_rows,rss,write
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'


def reciprocal_ranks(d,scores):
    """IDs break ties numerically; larger scores always mean preferred."""
    out=np.zeros(len(scores),dtype=np.float64)
    for a,b in zip(d['group_offsets'][:-1],d['group_offsets'][1:]):
        ids=np.array([int(d['item_ids'][i]) for i in d['row_item_indices'][a:b]])
        order=np.lexsort((ids,-scores[a:b])); ranks=np.empty(b-a)
        ranks[order]=np.arange(1,b-a+1);out[a:b]=1/(60+ranks)
    return out


def main():
    out,cache=BASE/'evidence/objective_diversity',BASE/'cache/objective_diversity'
    out.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise SystemExit('Preserve previous round')
    common=dict(PARAMS);common.update(n_estimators=417,n_jobs=1)
    common.pop('objective');common.pop('lambdarank_truncation_level');common.pop('label_gain')
    common.update(CONFIGS[1]);rank_params=dict(common,objective='rank_xendcg',label_gain=[0,1])
    binary_params=dict(common,objective='binary',metric='binary_logloss')
    protocol={'primary':'rrf60_all3','rank_parameters':rank_params,'binary_parameters':binary_params,
        'objective_support':'Installed lib_lightgbm.dylib contains rank_xendcg and xe_ndcg objective names (strings inspected, no install/network)',
        'fit_window':WINDOWS['fit'],'features':'Frozen v3 original base19 plus aligned weekly5',
        'target':'Full-universe top5_count80 before candidate restriction; numeric IDs break count ties',
        'training_groups':'Same positive-and-negative groups as v3; rank queries contiguous; binary total weight equal per retained user, mean row weight1',
        'fixed_fusions':['Equal RRF60 v3 + rank_xendcg','Equal RRF60 v3 + binary_equal_user','Primary equal RRF60 v3 + rank_xendcg + binary_equal_user'],
        'selection':'Primary declared before fitting; others diagnostics, no weight/iteration/config search',
        'evaluation':'All80 and top5_count80 on both previously explored overlapping windows, full-truth denominators, unchanged cold fallback',
        'limits':{'fits':2,'wall_seconds':300,'rss_bytes':1024**3,'threads':1},'submissions':0}
    write(out/'protocol.json',protocol);began=time.perf_counter();fits=0;rows=[];manifest={}
    def guard(env=None):
        if time.perf_counter()-began>300 or rss()>1024**3:raise RuntimeError('Prespecified resource budget exceeded')
    guard.order=-100;guard.before_iteration=True
    try:
        ds={};metrics={};db=sqlite3.connect('file:'+str(BASE/'cache/interactions.sqlite')+'?mode=ro',uri=True)
        db.execute('PRAGMA temp_store=FILE');db.execute('PRAGMA cache_size=-32000')
        for name in ('fit','eval','later'):
            guard();paths=[BASE/f'cache/context_features/{name}.npz',BASE/f'cache/deep_weekly/{name}_row_features.npy']
            with np.load(paths[0],allow_pickle=False) as z:d={k:z[k] for k in z.files}
            w=np.load(paths[1],allow_pickle=False)
            assert d['X'].shape[1]==19 and w.shape==(len(d['y']),5)
            d['X']=np.concatenate([d['X'],w],axis=1);assert np.isfinite(d['X']).all()
            truths=full_truth(db,*WINDOWS[name]);metrics[name]={t:Metric(d,v) for t,v in truths.items()}
            assert np.array_equal(metrics[name]['all80'].y,d['y'])
            assert np.array_equal(metrics[name]['all80'].den,np.minimum(5,d['truth_counts']))
            ds[name]=d
            for p in paths:manifest[str(p.relative_to(BASE))]=hashlib.sha256(p.read_bytes()).hexdigest()
        db.close();write(out/'inputs.json',manifest)
        keep,groups=training_rows(ds['fit'],metrics['fit']['top5_count80']);n=int(keep.sum())
        weights=np.repeat(n/(len(groups)*np.asarray(groups,dtype=float)),groups)
        assert len(weights)==n and np.isclose(weights.mean(),1)
        group_totals=np.add.reduceat(weights,np.r_[0,np.cumsum(groups)[:-1]])
        assert np.allclose(group_totals,n/len(groups))
        write(out/'training_weights.json',{'rows':n,'groups':len(groups),'mean_row_weight':float(weights.mean()),
            'min_weight':float(weights.min()),'max_weight':float(weights.max()),'equal_group_total':float(n/len(groups))})
        scores={name:{'v3':np.load(BASE/f'cache/deep_temporal/top5_count80_24features_{name}_scores.npy')} for name in ('eval','later')}
        for key in ('rank_xendcg','binary_equal_user'):
            guard();assert fits<2;started=time.perf_counter();fits+=1
            if key=='rank_xendcg':
                model=lgb.LGBMRanker(**rank_params)
                model.fit(ds['fit']['X'][keep],metrics['fit']['top5_count80'].y[keep],group=groups,callbacks=[guard])
            else:
                model=lgb.LGBMClassifier(**binary_params)
                model.fit(ds['fit']['X'][keep],metrics['fit']['top5_count80'].y[keep],sample_weight=weights,callbacks=[guard])
            row={'model':key,'fit_seconds':time.perf_counter()-started}
            model.booster_.save_model(str(cache/f'{key}.txt'))
            for name in ('eval','later'):
                guard()
                value=model.predict_proba(ds[name]['X'],num_threads=1)[:,list(model.classes_).index(1)] if key=='binary_equal_user' else model.predict(ds[name]['X'],num_threads=1)
                assert np.isfinite(value).all();scores[name][key]=value
                np.save(cache/f'{key}_{name}_scores.npy',value)
                row[name]={t:m.rank(value)[0] for t,m in metrics[name].items()}
                write(out/f'{key}_{name}_predictions.json',metrics[name]['all80'].predictions(value))
            rows.append(row);write(out/'results.partial.json',rows);print(json.dumps(row),flush=True)
        baseline={};fusion_results=[]
        for name in ('eval','later'):
            baseline[name]={t:m.rank(scores[name]['v3'])[0] for t,m in metrics[name].items()}
        write(out/'baseline.json',baseline)
        for key,components in [('rrf60_v3_rank',['v3','rank_xendcg']),('rrf60_v3_binary',['v3','binary_equal_user']),('rrf60_all3',['v3','rank_xendcg','binary_equal_user'])]:
            row={'model':key}
            for name in ('eval','later'):
                guard();value=sum(reciprocal_ranks(ds[name],scores[name][c]) for c in components)/len(components)
                values={t:m.rank(value)[0] for t,m in metrics[name].items()}
                row[name]={'scores':values,'delta_vs_v3':{t:values[t]-baseline[name][t] for t in values}}
                np.save(cache/f'{key}_{name}_scores.npy',value);write(out/f'{key}_{name}_predictions.json',metrics[name]['all80'].predictions(value))
            fusion_results.append(row);print(json.dumps(row),flush=True)
        write(out/'results.json',{'status':'complete','standalone':rows,'fusions':fusion_results,'baseline':baseline,
            'fits':fits,'wall_seconds':time.perf_counter()-began,'peak_rss_bytes':rss(),
            'lightgbm_version':lgb.__version__,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    except Exception as error:
        write(out/'failure.json',{'status':'incomplete','error':type(error).__name__,'detail':str(error),'fits':fits,
            'wall_seconds':time.perf_counter()-began,'peak_rss_bytes':rss()});raise


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
