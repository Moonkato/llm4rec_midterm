"""One fixed equal-weight rank mixture after normalization ablation."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
import json,pickle,time
import numpy as np
from check_history_normalization import B,normalize,evaluate
out=B/'evidence/normalized_fusion';out.mkdir(exist_ok=True)
protocol={'weight':.5,'rrf_constant':60,'components':['original_frozen_logistic','boost_context_counts_and_decay_normalized35days'],'selection':'one followup based on previous holdout ablation; exploratory, not independent confirmation','fits':0,'submissions':0}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2));start=time.perf_counter()
m=np.load(B/'cache/simple_reranker.npz')
with (B/'cache/boost_context.pkl').open('rb') as f:boost=pickle.load(f)
results={}
for name,days in [('eval',70),('later',105)]:
 with np.load(B/f'cache/context_features/{name}.npz') as f:d={k:f[k] for k in f.files}
 x=d['X'];scores=[((x[:,:9]-m['mean'])/m['scale'])@m['coefficients'][:-1]+m['coefficients'][-1],boost.predict_proba(normalize(x,days,'counts_and_decay'))[:,1]]
 fused=np.zeros(len(x))
 for n in range(len(d['users'])):
  a,b=d['group_offsets'][n:n+2];idx=d['row_item_indices'][a:b]
  for s in scores:
   order=sorted(range(b-a),key=lambda j:(-s[a+j],int(d['item_ids'][idx[j]])));r=np.empty(b-a);r[order]=np.arange(1,b-a+1);fused[a:b]+=.5/(60+r)
 metric,pred=evaluate(d,fused);results[name]=metric;(out/f'{name}_predictions.json').write_text(json.dumps(pred))
(out/'results.json').write_text(json.dumps({'map5':results,'wall_seconds':time.perf_counter()-start},indent=2));print(json.dumps(results))
