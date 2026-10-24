"""Calibrate history-length-dependent features to the 35-day fitting history."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
import json,pickle,time,hashlib
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parents[1]

def normalize(x,days,variant):
    x=x.copy();columns=[0,1,5,7,9,10,13]
    for j in columns:x[:,j]=np.log1p(np.expm1(x[:,j])*35/days)
    if variant=='counts_and_decay':
        for j,half in [(2,60),(11,7),(12,28)]:
            factor=(-np.expm1(-np.log(2)*35/half))/(-np.expm1(-np.log(2)*days/half))
            x[:,j]=np.log1p(np.expm1(x[:,j])*factor)
    return x

def evaluate(d,scores):
    aps=[];pred={}
    for n,u in enumerate(d['users']):
        a,b=d['group_offsets'][n:n+2];idx=d['row_item_indices'][a:b];y=d['y'][a:b];den=min(5,int(d['truth_counts'][n]))
        if d['cold'][n]:
            lookup={int(i):k for k,i in enumerate(idx)};top=[lookup[int(i)] for i in d['fallback_item_indices'][:5]]
        else:top=sorted(range(b-a),key=lambda j:(-scores[a+j],int(d['item_ids'][idx[j]])))[:5]
        hits=0;ap=0
        for rank,j in enumerate(top,1):
            if y[j]:hits+=1;ap+=hits/rank
        aps.append(ap/den if den else 0);pred[str(u)]=[str(d['item_ids'][idx[j]]) for j in top]
    return float(np.mean(aps)),pred

def main():
    out=B/'evidence/history_normalization';out.mkdir(exist_ok=True)
    if (out/'results.json').exists():raise SystemExit('Preserve completed experiment')
    protocol={'fit_days':35,'eval_days':70,'later_days':105,'variants':['counts_only','counts_and_decay'],'count_columns':[0,1,5,7,9,10,13],'decay_half_lives':{'2':60,'11':7,'12':28},'assumption':'constant per-day viewing rate within history; normalization does not assume observed future views','models':['frozen_logistic','boost_context'],'selection':'exploratory on existing overlapping windows','fits':0,'submissions':0,'explanation':'Fit history is already35days, so the training transformation is identity and no refitting is required.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));start=time.perf_counter()
    m=np.load(B/'cache/simple_reranker.npz')
    with (B/'cache/boost_context.pkl').open('rb') as f:boost=pickle.load(f)
    results=[]
    for name,days in [('eval',70),('later',105)]:
        with np.load(B/f'cache/context_features/{name}.npz') as f:d={k:f[k] for k in f.files}
        for variant in protocol['variants']:
            x=normalize(d['X'],days,variant)
            scores={'frozen_logistic':((x[:,:9]-m['mean'])/m['scale'])@m['coefficients'][:-1]+m['coefficients'][-1],'boost_context':boost.predict_proba(x)[:,1]}
            for method,score in scores.items():
                metric,pred=evaluate(d,score);row={'window':name,'normalization':variant,'model':method,'map5':metric};results.append(row);print(json.dumps(row),flush=True)
                (out/f'{method}_{variant}_{name}_predictions.json').write_text(json.dumps(pred))
    (out/'results.json').write_text(json.dumps({'results':results,'wall_seconds':time.perf_counter()-start,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
if __name__=='__main__':main()
