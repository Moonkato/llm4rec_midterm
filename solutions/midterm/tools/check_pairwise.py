"""Two bounded linear pairwise logistic fits on the early window only."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import hashlib
import json
import resource
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from check_history_normalization import evaluate

BASE=Path(__file__).resolve().parents[1]


def main():
    out=BASE/'evidence/pairwise';cache=BASE/'cache/pairwise'
    if (out/'results.json').exists():raise SystemExit('Preserve completed experiment')
    out.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    protocol={'seed':20261022,'fits':2,'variants':['random3','random1_hard2'],
        'features':'all19 frozen context features, early-fit-only column standardization',
        'fit_window':'history March9-Apr13, labels Apr13-May18',
        'positive_cap_per_user':20,'negatives_per_positive':3,
        'hard_definition':'largest historical personal_schedule_score (feature5), numeric item ID ties',
        'sampling':'uniform positive subset if >20; negative sampling with replacement only when pool too small; mixture excludes hard IDs from random draw when possible',
        'user_weight':'equal total pair weight for each user with both labels; users without pos or neg omitted',
        'objective':'weighted mean logaddexp(0,-difference@w)+0.5*0.001*||w||^2',
        'optimizer':'L-BFGS-B max100 iterations, no intercept',
        'evaluation':'unchanged eval/later candidates, truth denominators, cold popularity fallback and numeric ID ties',
        'selection':'exploratory on previously evaluated overlapping holdouts; not independent confirmation',
        'normalization':'no history-length normalization in this ablation',
        'submissions':0}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    start=time.perf_counter()
    def load(name):
        with np.load(BASE/f'cache/context_features/{name}.npz',allow_pickle=False) as z:return {k:z[k] for k in z.files}
    fit=load('fit');mu=fit['X'].mean(axis=0,dtype=np.float64);sd=fit['X'].std(axis=0,dtype=np.float64);sd[sd==0]=1
    x=(fit['X'].astype(np.float64)-mu)/sd
    results=[]
    for variant in protocol['variants']:
        rng=np.random.default_rng(protocol['seed']);pos_rows=[];neg_rows=[];weights=[];used=0
        for n,u in enumerate(fit['users']):
            a,b=fit['group_offsets'][n:n+2];positive=np.flatnonzero(fit['y'][a:b])+a;negative=np.flatnonzero(fit['y'][a:b]==0)+a
            if len(positive)==0 or len(negative)==0:continue
            used+=1
            if len(positive)>20:positive=np.sort(rng.choice(positive,20,replace=False))
            hard=sorted(negative,key=lambda j:(-fit['X'][j,5],int(fit['item_ids'][fit['row_item_indices'][j]])))[:2]
            if len(hard)<2:hard=hard*2
            remaining=np.array([j for j in negative if j not in hard],dtype=np.int64)
            for p in positive:
                if variant=='random3':neg=rng.choice(negative,3,replace=len(negative)<3)
                else:neg=[int(rng.choice(remaining if len(remaining) else negative)),*hard]
                pos_rows.extend([int(p)]*3);neg_rows.extend(map(int,neg));weights.extend([1/(3*len(positive))]*3)
        assert len(pos_rows)<=300000
        differences=x[np.asarray(pos_rows)]-x[np.asarray(neg_rows)]
        pair_weights=np.asarray(weights,dtype=np.float64);pair_weights/=pair_weights.sum()
        assert np.isfinite(differences).all()
        began=time.perf_counter()
        def loss(w):
            z=differences@w
            value=np.dot(pair_weights,np.logaddexp(0,-z))+.0005*np.dot(w,w)
            gradient=-(differences.T@(pair_weights*expit(-z)))+.001*w
            return value,gradient
        result=minimize(loss,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':100})
        row={'method':variant,'paircount':len(pos_rows),'fit_users':used,'fit_seconds':time.perf_counter()-began,
             'converged':bool(result.success),'message':str(result.message),'iterations':int(result.nit),
             'objective':float(result.fun),'coefficients_standardized':result.x.tolist()}
        np.savez(cache/f'{variant}.npz',coefficients=result.x,mean=mu,scale=sd,feature_names=fit['feature_names'])
        del differences
        for window in ('eval','later'):
            data=load(window)
            assert np.array_equal(data['feature_names'],fit['feature_names'])
            scores=((data['X']-mu)/sd)@result.x
            assert np.isfinite(scores).all()
            metric,predictions=evaluate(data,scores)
            row[window]={'map5':metric,'empty_truth_users':int(np.sum(data['truth_counts']==0))}
            (out/f'{variant}_{window}_predictions.json').write_text(json.dumps(predictions)+'\n')
            del data
        results.append(row);print(json.dumps(row),flush=True)
        (out/'results.partial.json').write_text(json.dumps(results,indent=2)+'\n')
    manifest={'results':results,'wall_seconds':time.perf_counter()-start,
        'peak_rss_platform_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'fit_npz_sha256':hashlib.sha256((BASE/'cache/context_features/fit.npz').read_bytes()).hexdigest()}
    (out/'results.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
