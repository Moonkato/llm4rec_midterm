"""Two controlled CE fits, hard versus fractional ties. Prepare before root start."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import hashlib,json,sqlite3,time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from threadpoolctl import threadpool_limits
from run_deep_temporal import BASE,WINDOWS,CONFIGS,Metric,rss,write
from tie_control_utils import count_qualified,tie_membership,probability,expected_ap
from run_objective_diversity import reciprocal_ranks
from run_round1 import apk
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'


def main():
    out,cache=BASE/'evidence/tie_controls',BASE/'cache/tie_controls'
    out.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise SystemExit('Preserve prior controls')
    params=dict(objective='cross_entropy',metric='cross_entropy',n_estimators=417,learning_rate=.03,
        random_state=20261023,n_jobs=1,verbosity=-1,deterministic=True,force_col_wise=True,**CONFIGS[1])
    protocol={'variants':['hard_numeric_top5','fractional_boundary_membership'],'parameters':params,
        'features':'Identical original19+weekly5, fixed original candidate rows/users',
        'fit_rows':'ALL original fit rows, including queries with no positive candidates; no variant-dependent filtering',
        'weights':'Constant per user1/nrows, globally normalized to mean1; equal total row-weight per original user',
        'labels':'Full-universe counts before shortlist; >80% single stored overlap proxy. Fractional labels1 for strictly above fifthcount, remaining_slots/tie_group_size at boundary,0 below.',
        'expected_metric':'Exact expectation of ordinary AP under uniformly sampled boundary-tie subsets without replacement; fulltruth denominator and pairwise inclusion probabilities',
        'fixed_diagnostics':'Each standalone and equalRRF60 with frozen original event model; no blend search',
        'baseline':'Frozen v3 and original equalRRF60 event+v3 (v4), same cold fallback',
        'selection':'Controls secondary; final OOF stack remains prespecified hardtop5 LambdaRank irrespective control results',
        'limits':{'fits':2,'wall_seconds':300,'rss_bytes':int(1.5*1024**3),'threads':1},
        'caveat':'Exploratory reused overlapping holdouts; tie metric is sensitivity, not verified official Kaggle scorer','submissions':0}
    write(out/'protocol.json',protocol);began=time.perf_counter();fits=0;manifest={};results=[]
    def guard(env=None):
        if time.perf_counter()-began>300 or rss()>int(1.5*1024**3):raise RuntimeError('Tie control budget exceeded')
    guard.order=-100;guard.before_iteration=True
    try:
        ds={};infos={};metrics={};labels={}
        db=sqlite3.connect('file:'+str(BASE/'cache/interactions.sqlite')+'?mode=ro',uri=True)
        db.execute('PRAGMA cache_size=-32000');db.execute('PRAGMA temp_store=FILE')
        original_hashes=json.loads((BASE/'evidence/deep_temporal/inputs.json').read_text())
        for name in ('fit','eval','later'):
            guard();paths=[BASE/f'cache/context_features/{name}.npz',BASE/f'cache/deep_weekly/{name}_row_features.npy']
            with np.load(paths[0],allow_pickle=False) as z:d={k:z[k] for k in z.files}
            w=np.load(paths[1],allow_pickle=False);assert d['X'].shape[1]==19 and w.shape==(len(d['y']),5)
            d['X']=np.concatenate([d['X'],w],axis=1);ds[name]=d
            counts=count_qualified(db,*WINDOWS[name]);info={str(u):tie_membership(counts.get(str(u),{})) for u in d['users']};infos[name]=info
            alltruth={u:set(vals) for u,vals in counts.items()};hardtruth={u:x['hard'] for u,x in info.items()}
            metrics[name]={'all80':Metric(d,alltruth),'hard_top5':Metric(d,hardtruth)}
            assert np.array_equal(metrics[name]['all80'].y,d['y'])
            assert np.array_equal(metrics[name]['all80'].den,np.minimum(5,d['truth_counts']))
            labels[name]=np.array([probability(str(d['item_ids'][idx]),info[str(d['users'][g])]) for g,idx in zip(metrics[name]['all80'].g,d['row_item_indices'])],dtype=np.float32)
            assert np.all((labels[name]>=0)&(labels[name]<=1))
            for p in paths:
                key=str(p.relative_to(BASE));manifest[key]=hashlib.sha256(p.read_bytes()).hexdigest();assert manifest[key]==original_hashes[key]
        db.close();write(out/'inputs.json',manifest)
        lengths=np.diff(ds['fit']['group_offsets']);n=len(ds['fit']['y']);g=len(lengths)
        weights=np.repeat(n/(g*lengths.astype(float)),lengths)
        assert np.isclose(weights.mean(),1) and np.allclose(np.add.reduceat(weights,ds['fit']['group_offsets'][:-1]),n/g)
        write(out/'weights.json',{'rows':n,'users':g,'mean_weight':float(weights.mean()),'query_weight_sum':n/g,
            'hard_positive_rows':int(metrics['fit']['hard_top5'].y.sum()),'fractional_positive_mass':float(labels['fit'].sum()),
            'fractional_boundary_rows':int(((labels['fit']>0)&(labels['fit']<1)).sum())})
        def assess(name,scores):
            pred=metrics[name]['all80'].predictions(scores)
            return {'all80_map5':metrics[name]['all80'].rank(scores)[0],
                    'hard_top5_map5':metrics[name]['hard_top5'].rank(scores)[0],
                    'exact_tie_expected_ap5':float(np.mean([expected_ap(pred[str(u)],infos[name][str(u)]) for u in ds[name]['users']]))},pred
        event={};baselines={}
        for name in ('eval','later'):
            event[name]=np.load(BASE/f'cache/airing_model/{name}_expected_counts.npy')
            v3=np.load(BASE/f'cache/deep_temporal/top5_count80_24features_{name}_scores.npy')
            fused=(reciprocal_ranks(ds[name],v3)+reciprocal_ranks(ds[name],event[name]))/2
            baselines[name]={'v3':assess(name,v3)[0],'v4':assess(name,fused)[0]}
        write(out/'baselines.json',baselines)
        for variant,target in [('hard_numeric_top5',metrics['fit']['hard_top5'].y.astype(np.float32)),('fractional_boundary_membership',labels['fit'])]:
            guard();assert fits<2;model=lgb.LGBMRegressor(**params);fits+=1;started=time.perf_counter()
            model.fit(ds['fit']['X'],target,sample_weight=weights,callbacks=[guard]);row={'variant':variant,'fit_seconds':time.perf_counter()-started}
            model.booster_.save_model(str(cache/f'{variant}.txt'))
            for name in ('eval','later'):
                guard();scores=model.predict(ds[name]['X'],num_threads=1);assert np.isfinite(scores).all()
                assert np.all(scores>=0) and np.all(scores<=1)
                row[name]={}
                for suffix,value in [('standalone',scores),('event_rrf60',(reciprocal_ranks(ds[name],scores)+reciprocal_ranks(ds[name],event[name]))/2)]:
                    vals,pred=assess(name,value);row[name][suffix]=vals
                    np.save(cache/f'{variant}_{suffix}_{name}_scores.npy',value);write(out/f'{variant}_{suffix}_{name}_predictions.json',pred)
            results.append(row);write(out/'results.partial.json',results);print(json.dumps(row),flush=True)
        write(out/'results.json',{'status':'complete','results':results,'baselines':baselines,'fits':fits,
            'wall_seconds':time.perf_counter()-began,'peak_rss_bytes':rss(),'lightgbm_version':lgb.__version__,
            'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'helper_sha256':hashlib.sha256((BASE/'tools/tie_control_utils.py').read_bytes()).hexdigest()})
    except Exception as error:
        write(out/'failure.json',{'status':'incomplete','error':type(error).__name__,'detail':str(error),'fits':fits,
            'wall_seconds':time.perf_counter()-began,'peak_rss_bytes':rss()});raise


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
