"""Paired user bootstrap of frozen ranker vs submitted v2; no model fitting."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import json
import numpy as np
from check_history_normalization import B, evaluate
from run_round1 import apk

def main():
    out=B/'evidence/ranker_uncertainty.json'
    if out.exists():
        raise SystemExit('Preserve completed audit')
    results={}
    with np.load(B/'cache/simple_reranker.npz') as z:
        model={k:z[k] for k in z.files}
    for window in ('eval','later'):
        with np.load(B/f'cache/context_features/{window}.npz') as z:
            d={k:z[k] for k in z.files}
        s1=((d['X'][:,:9]-model['mean'])/model['scale'])@model['coefficients'][:-1]+model['coefficients'][-1]
        s2=np.load(B/f'cache/training_filter/filtered_equal_rows_{window}_scores.npy')
        fused=np.zeros(len(s1))
        for a,b in zip(d['group_offsets'][:-1],d['group_offsets'][1:]):
            items=d['item_ids'][d['row_item_indices'][a:b]]
            for scores in (s1,s2):
                order=sorted(range(b-a),key=lambda j:(-scores[a+j],int(items[j])))
                ranks=np.empty(b-a);ranks[order]=np.arange(1,b-a+1)
                fused[a:b]+=.5/(60+ranks)
        _,baseline=evaluate(d,fused)
        candidate=json.loads((B/f'evidence/lambdarank_targets/top5_count80_{window}_predictions.json').read_text())
        truth=json.loads((B/f'evidence/top5_target_audit/{window}_truth.json').read_text())['top5_count80']
        delta=np.array([apk(truth[u],candidate[u])-apk(truth[u],baseline[u]) for u in d['users']])
        rng=np.random.default_rng(20261023)
        means=np.concatenate([delta[rng.integers(0,len(delta),size=(100,len(delta)))].mean(axis=1) for _ in range(50)])
        results[window]={'mean_difference':float(delta.mean()),'paired_user_percentile_interval_95':np.quantile(means,[.025,.975]).tolist(),
                         'users_improved':int((delta>0).sum()),'users_worsened':int((delta<0).sum()),'users_equal':int((delta==0).sum())}
    out.write_text(json.dumps({'metric':'MAP@5, top5_count80 proxy','candidate':'frozen LightGBM top5_count80','baseline':'submitted v2 equal RRF',
        'bootstrap_replicates':5000,'seed':20261023,'results':results,'fits':0,'submissions':0,
        'limitations':'Exploratory intervals condition on repeatedly explored data and fixed fitted models; not adjusted for selection. Users may not be independent, periods overlap. Does not quantify Kaggle uncertainty.'},indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':
    main()
