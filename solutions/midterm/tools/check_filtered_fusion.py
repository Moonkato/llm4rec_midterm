"""Fixed small rank-fusion sweep; no fits or submissions."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
import json,pickle
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parents[1];O=B/'evidence/filtered_fusion';O.mkdir(exist_ok=True)
protocol={'weights_boost':[.25,.5,.75],'rrf_constant':60,'components':['frozen_logistic','filtered_equal_rows'],'selection':'exploratory same windows, no independent confirmation','training_runs':0}
(O/'protocol.json').write_text(json.dumps(protocol,indent=2))
with (B/'cache/training_filter/filtered_equal_rows.pkl').open('rb') as f:model=pickle.load(f)
m=np.load(B/'cache/simple_reranker.npz');results={}
for name in ['eval','later']:
 with np.load(B/f'cache/context_features/{name}.npz') as f:d={k:f[k] for k in f.files}
 scores=[((d['X'][:,:9]-m['mean'])/m['scale'])@m['coefficients'][:-1]+m['coefficients'][-1],model.predict_proba(d['X'])[:,1]]
 aps={str(w):[] for w in protocol['weights_boost']}
 for n,u in enumerate(d['users']):
  a,b=d['group_offsets'][n:n+2];idx=d['row_item_indices'][a:b];y=d['y'][a:b];den=min(5,int(d['truth_counts'][n]));parts=[]
  for score in scores:
   order=sorted(range(b-a),key=lambda j:(-score[a+j],int(d['item_ids'][idx[j]])));r=np.empty(b-a);r[order]=np.arange(1,b-a+1);parts.append(1/(60+r))
  for w in protocol['weights_boost']:
   fused=(1-w)*parts[0]+w*parts[1]
   if d['cold'][n]:
    lookup={int(i):k for k,i in enumerate(idx)};top=[lookup[int(i)] for i in d['fallback_item_indices'][:5]]
   else:top=sorted(range(b-a),key=lambda j:(-fused[j],int(d['item_ids'][idx[j]])))[:5]
   hit=0;v=0
   for rank,j in enumerate(top,1):
    if y[j]:hit+=1;v+=hit/rank
   aps[str(w)].append(v/den if den else 0)
 results[name]={w:float(np.mean(v)) for w,v in aps.items()}
(O/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results))
