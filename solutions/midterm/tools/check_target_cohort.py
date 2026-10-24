"""One pooled-window ablation restricted to known submission IDs; no future labels."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
import json,pickle,time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from check_history_normalization import B,evaluate
out=B/'evidence/target_cohort';out.mkdir(exist_ok=True)
if (out/'results.json').exists():raise SystemExit('Preserve completed experiment')
config=json.loads((B/'evidence/multiwindow/protocol.json').read_text())
protocol={'source_protocol':'evidence/multiwindow/protocol.json','change':'Restrict training rows to template user IDs, which are supplied before test; no test views accessed','models':1,'selection':'exploratory late holdout already inspected','submissions':0}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2));start=time.perf_counter()
target=set(json.loads((B/'evidence/round1/catalog.json').read_text())['users']);xs=[];ys=[]
for p in [B/'cache/context_features/fit.npz',B/'cache/multiwindow/recent_fit.npz']:
 with np.load(p) as f:d={k:f[k] for k in f.files}
 mask=np.zeros(len(d['y']),dtype=bool)
 for j,u in enumerate(d['users']):
  if str(u) in target:
   a,b=d['group_offsets'][j:j+2];mask[a:b]=True
 xs.append(d['X'][mask]);ys.append(d['y'][mask])
x=np.concatenate(xs);y=np.concatenate(ys)
model=HistGradientBoostingClassifier(**config['parameters']);model.fit(x,y)
with np.load(B/'cache/context_features/later.npz') as f:d={k:f[k] for k in f.files}
score=model.predict_proba(d['X'])[:,1];metric,pred=evaluate(d,score)
result={'fit_rows':len(y),'later_map5':metric,'wall_seconds':time.perf_counter()-start,'same_parameters_as_multiwindow':True}
(out/'results.json').write_text(json.dumps(result,indent=2));(out/'predictions.json').write_text(json.dumps(pred))
with (B/'cache/multiwindow/target_cohort.pkl').open('wb') as f:pickle.dump(model,f)
print(json.dumps(result))
