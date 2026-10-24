"""Recompute submitted CSV predictions from local frozen models and actual history."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import csv,json,pickle,sqlite3,time,hashlib
from pathlib import Path
import numpy as np
from export_candidate import predict
from build_interactions import tick
from check_history_normalization import B

def main():
 began=time.perf_counter();root=B.parents[1]
 with (root/'midterm/data/submission.csv').open() as f:users=[r['user_id'] for r in csv.DictReader(f)]
 m=np.load(B/'cache/simple_reranker.npz');db=sqlite3.connect('file:'+str(B/'cache/interactions.sqlite')+'?mode=ro',uri=True)
 v1,checks=predict(db,users,m,tick('2020-07-27 00:00:00'),tick('2020-10-19 00:00:00'));db.close()
 with np.load(B/'cache/context_features/test.npz') as f:d={k:f[k] for k in f.files}
 with (B/'cache/training_filter/filtered_equal_rows.pkl').open('rb') as f:boost=pickle.load(f)
 scores=[((d['X'][:,:9]-m['mean'])/m['scale'])@m['coefficients'][:-1]+m['coefficients'][-1],boost.predict_proba(d['X'])[:,1]];v2={}
 for n,u in enumerate(d['users']):
  a,b=d['group_offsets'][n:n+2];idx=d['row_item_indices'][a:b];fused=np.zeros(b-a)
  for score in scores:
   order=sorted(range(b-a),key=lambda j:(-score[a+j],int(d['item_ids'][idx[j]])));rank=np.empty(b-a);rank[order]=np.arange(1,b-a+1);fused+=.5/(60+rank)
  if d['cold'][n]:top=[str(d['item_ids'][i]) for i in d['fallback_item_indices'][:5]]
  else:top=[str(d['item_ids'][idx[j]]) for j in sorted(range(b-a),key=lambda j:(-fused[j],int(d['item_ids'][idx[j]])))[:5]]
  v2[str(u)]=top
 result={}
 for name,pred in [('logistic_v1',v1),('filtered_context_v2',v2)]:
  path=B/f'candidates/{name}.csv'
  with path.open() as f:rows=list(csv.DictReader(f))
  mismatch=[r['user_id'] for r in rows if r['tv_show_id'].split()!=pred[r['user_id']]]
  assert [r['user_id'] for r in rows]==users and len(rows)==1260
  result[name]={'mismatches':len(mismatch),'all_1260_ranked_lists_match':not mismatch,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
  assert not mismatch
 result['wall_seconds']=time.perf_counter()-began
 (B/'evidence/submissions/reproduction_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
