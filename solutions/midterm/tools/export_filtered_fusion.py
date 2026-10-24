"""Export selected frozen equal-RRF candidate; no training or network."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import csv,json,pickle,hashlib,time
import numpy as np
from check_history_normalization import B

def main():
 start=time.perf_counter();dest=B/'candidates/filtered_context_v2.csv';out=B/'evidence/candidate_filtered_context_v2.json'
 if dest.exists() or out.exists():raise SystemExit('Preserve candidate')
 with np.load(B/'cache/context_features/test.npz') as f:d={k:f[k] for k in f.files}
 m=np.load(B/'cache/simple_reranker.npz')
 with (B/'cache/training_filter/filtered_equal_rows.pkl').open('rb') as f:boost=pickle.load(f)
 scores=[((d['X'][:,:9]-m['mean'])/m['scale'])@m['coefficients'][:-1]+m['coefficients'][-1],boost.predict_proba(d['X'])[:,1]]
 recs=[]
 for n,u in enumerate(d['users']):
  a,b=d['group_offsets'][n:n+2];idx=d['row_item_indices'][a:b];fused=np.zeros(b-a)
  for score in scores:
   order=sorted(range(b-a),key=lambda j:(-score[a+j],int(d['item_ids'][idx[j]])));rank=np.empty(b-a);rank[order]=np.arange(1,b-a+1);fused+=.5/(60+rank)
  if d['cold'][n]:top=[str(d['item_ids'][i]) for i in d['fallback_item_indices'][:5]]
  else:top=[str(d['item_ids'][idx[j]]) for j in sorted(range(b-a),key=lambda j:(-fused[j],int(d['item_ids'][idx[j]])))[:5]]
  assert len(top)==len(set(top))==5 and '0' not in top
  recs.append({'user_id':str(u),'tv_show_id':' '.join(top)})
 root=B.parents[1]
 with (root/'midterm/data/submission.csv').open() as f:template=list(csv.DictReader(f))
 assert len(recs)==1260 and [r['user_id'] for r in recs]==[r['user_id'] for r in template]
 with dest.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=['user_id','tv_show_id']);w.writeheader();w.writerows(recs)
 with (B/'candidates/logistic_v1.csv').open() as f:previous={r['user_id']:r['tv_show_id'] for r in csv.DictReader(f)}
 manifest={'status':'prepared, not submitted','proposed_submission_name':'midterm_02_filtered_context_rrf','file':str(dest.relative_to(root)),'rows':len(recs),'changed_ranked_lists_vs_v1':sum(previous[r['user_id']]!=r['tv_show_id'] for r in recs),'local_map5':{'main':.5854757495590829,'later':.49270568783068786},'formula':'0.5/(60+rank_logistic)+0.5/(60+rank_filtered_boost)','training':'both models frozen fit Apr13May18 labels; prediction features from full history beforeJuly27','test_labels_used':False,'selection_caveat':'selected on repeatedly explored overlapping holdouts; does not guarantee Kaggle gain','hashes':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [dest,B/'cache/simple_reranker.npz',B/'cache/training_filter/filtered_equal_rows.pkl',B/'cache/context_features/test.npz',__file__ and __import__('pathlib').Path(__file__)]},'wall_seconds':time.perf_counter()-start}
 out.write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
