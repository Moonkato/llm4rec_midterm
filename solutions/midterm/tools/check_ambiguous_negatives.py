"""Training-only removal of partial-view negatives; evaluation labels unchanged."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import json,pickle,sqlite3,time,hashlib
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from check_history_normalization import B,evaluate
from build_interactions import tick

def main():
 out=B/'evidence/ambiguous_negatives';out.mkdir(exist_ok=True)
 if (out/'results.json').exists():raise SystemExit('Preserve completed run')
 protocol={'fit_window':['2020-04-13','2020-05-18'],'filter':'Drop training candidate rows labelled0 if any fit-label-window single fragment covers >30% of airing; retain all strict>80% positives. No validation label changes.','parameters':{'learning_rate':.06,'max_iter':150,'max_leaf_nodes':15,'max_depth':3,'min_samples_leaf':50,'l2_regularization':10.,'early_stopping':False,'random_state':20261022},'hypothesis':'partial viewing conveys interest and may be an uncertain negative for interest ranking, although strict target still treats it negative','selection':'exploratory on reused main/later windows','submissions':0}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2));start=time.perf_counter()
 with np.load(B/'cache/context_features/fit.npz') as f:d={k:f[k] for k in f.files}
 db=sqlite3.connect('file:'+str(B/'cache/interactions.sqlite')+'?mode=ro',uri=True);a=tick('2020-04-13 00:00:00');b=tick('2020-05-18 00:00:00')
 partial=set(db.execute('''SELECT DISTINCT o.user,e.item FROM overlaps o JOIN events e USING(event_id) WHERE o.start<? AND o.end>? AND MAX(0,MIN(o.end,?)-MAX(o.start,?))>.3*e.duration''',(b,a,b,a)));db.close()
 keep=np.ones(len(d['y']),dtype=bool)
 for j,u in enumerate(d['users']):
  lo,hi=d['group_offsets'][j:j+2]
  for row in range(lo,hi):
   if not d['y'][row] and (str(u),str(d['item_ids'][d['row_item_indices'][row]])) in partial:keep[row]=False
 model=HistGradientBoostingClassifier(**protocol['parameters']);model.fit(d['X'][keep],d['y'][keep]);results={}
 for name in ['eval','later']:
  with np.load(B/f'cache/context_features/{name}.npz') as f:v={k:f[k] for k in f.files}
  metric,pred=evaluate(v,model.predict_proba(v['X'])[:,1]);results[name]=metric;(out/f'{name}_predictions.json').write_text(json.dumps(pred))
 with (B/'cache/ambiguous_negatives.pkl').open('wb') as f:pickle.dump(model,f)
 result={'map5':results,'training_rows_before':len(keep),'training_rows_after':int(keep.sum()),'ambiguous_negatives_removed':int((~keep).sum()),'wall_seconds':time.perf_counter()-start,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};(out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
