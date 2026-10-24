"""Bounded gradient boosting comparison on frozen chronological features."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='2'
import json,time,pickle,hashlib,resource,sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import sklearn
BASE=Path(__file__).resolve().parents[1]

def main():
    state=json.loads((BASE/'run_state.json').read_text())
    if state.get('next_training_request',{}).get('status')!='approved_context_round':raise SystemExit('Explicit context training approval not recorded.')
    out=BASE/'evidence/context_boosting';out.mkdir(exist_ok=True)
    if (out/'results.json').exists():raise SystemExit('Preserve completed round.')
    config={'seed':20261022,'learning_rate':.06,'iterations':150,'l2':10.,'min_samples_leaf':50,'early_stopping':False,'models':[{'name':'boost_old9','features':9,'max_leaf_nodes':15,'max_depth':3},{'name':'boost_context','features':None,'max_leaf_nodes':15,'max_depth':3},{'name':'boost_context_deeper','features':None,'max_leaf_nodes':31,'max_depth':5}],'selection':'primary window MAP; later window robustness overlaps primary, not independent test','new_submissions':0}
    (out/'protocol.json').write_text(json.dumps(config,indent=2))
    start=time.perf_counter()
    # Names provided explicitly so the feature-builder output cannot be guessed silently.
    paths=[Path(x) for x in sys.argv[1:]]
    if len(paths)!=3:raise SystemExit('Pass fit.npz main.npz later.npz paths.')
    assert [p.name for p in paths]==['fit.npz','eval.npz','later.npz'], 'Unexpected window order'
    loaded=[]
    for p in paths:
        with np.load(p,allow_pickle=False) as archive:loaded.append({k:archive[k] for k in archive.files})
    fit,mainval,late=loaded
    assert all(np.array_equal(fit['feature_names'],d['feature_names']) for d in [mainval,late])
    for d in [fit,mainval,late]:assert np.isfinite(d['X']).all()
    results=[]
    def evaluate(name,d,scores):
        aps=[];pred={};oracle=[]
        for n,u in enumerate(d['users']):
            a,b=d['group_offsets'][n:n+2];idx=d['row_item_indices'][a:b];ys=d['y'][a:b];items=d['item_ids'];den=min(5,int(d['truth_counts'][n]))
            if d['cold'][n]:
                lookup={int(j):k for k,j in enumerate(idx)};r=[lookup[int(j)] for j in d['fallback_item_indices'][:5]]
            else:r=sorted(range(b-a),key=lambda j:(-scores[a+j],int(items[idx[j]])))[:5]
            hits=0;ap=0.
            for rank,j in enumerate(r,1):
                if ys[j]:hits+=1;ap+=hits/rank
            aps.append(ap/den if den else 0.);oracle.append(min(5,int(ys.sum()))/den if den else 0.)
            pred[str(u)]=[str(items[idx[j]]) for j in r]
        return {'map5':float(np.mean(aps)),'candidate_oracle_map5':float(np.mean(oracle)),'empty_truth_users':int((d['truth_counts']==0).sum())},pred
    frozen=np.load(BASE/'cache/simple_reranker.npz')
    row={'method':'frozen_logistic_expanded_candidates','fit_seconds':0}
    for name,d in [('main',mainval),('later',late)]:
        scores=((d['X'][:,:9]-frozen['mean'])/frozen['scale'])@frozen['coefficients'][:-1]+frozen['coefficients'][-1]
        row[name],pred=evaluate(name,d,scores);(out/(row['method']+'_'+name+'_predictions.json')).write_text(json.dumps(pred))
    results.append(row);print(json.dumps(row),flush=True)
    for c in config['models']:
        n=c['features'] or fit['X'].shape[1];began=time.perf_counter()
        model=HistGradientBoostingClassifier(learning_rate=.06,max_iter=150,max_leaf_nodes=c['max_leaf_nodes'],max_depth=c['max_depth'],min_samples_leaf=50,l2_regularization=10.,early_stopping=False,random_state=20261022)
        model.fit(fit['X'][:,:n],fit['y']);row={'method':c['name'],'fit_seconds':time.perf_counter()-began,'features':n}
        for name,d in [('main',mainval),('later',late)]:
            scores=model.predict_proba(d['X'][:,:n])[:,1];row[name],pred=evaluate(name,d,scores);(out/(c['name']+'_'+name+'_predictions.json')).write_text(json.dumps(pred))
        with (BASE/'cache'/f"{c['name']}.pkl").open('wb') as f:pickle.dump(model,f)
        results.append(row);(out/'results.partial.json').write_text(json.dumps(results,indent=2));print(json.dumps(row),flush=True)
    manifest={'results':results,'wall_seconds':time.perf_counter()-start,'sklearn_version':sklearn.__version__,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'input_files':[str(p) for p in paths],'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'results.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'completed':True,'wall_seconds':manifest['wall_seconds']}),flush=True)
if __name__=='__main__':main()
