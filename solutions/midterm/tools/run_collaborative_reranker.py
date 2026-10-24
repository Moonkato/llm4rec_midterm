"""Two controlled rerankers on collaborative-expanded candidate generation."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import json,pickle,time,hashlib,resource
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from check_history_normalization import B,evaluate

def main():
 out=B/'evidence/collaborative_reranker';out.mkdir(exist_ok=True);cache=B/'cache/collaborative_reranker';cache.mkdir(exist_ok=True)
 if (out/'results.json').exists():raise SystemExit('Preserve completed round')
 assert (B/'evidence/collaborative_features/results.json').exists(),'Feature preparation incomplete'
 parameters=json.loads((B/'evidence/training_filter/protocol.json').read_text())['parameters']
 protocol={'models':['expanded_candidates_old19','expanded_candidates_collaborative23'],'parameters':parameters,'filter':'early fit usergroups with bothlabels only, equal rowweight','candidate_expansion':'top20user neighbors and top20itemassociation in addition to previous candidates','labels':'unchanged strict>80% truth; all1260 users','selection':'exploratory repeatedly used temporal windows; no independently unseen validation claim','submissions':0}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2));start=time.perf_counter();datasets={}
 for name in ['fit','eval','later']:
  with np.load(B/f'cache/collaborative_features/{name}.npz') as f:datasets[name]={k:f[k] for k in f.files}
 fit=datasets['fit']
 assert fit['X'].shape[1]==23 and all(np.array_equal(fit['feature_names'],d['feature_names']) for d in datasets.values())
 keep=np.zeros(len(fit['y']),dtype=bool)
 for a,b in zip(fit['group_offsets'][:-1],fit['group_offsets'][1:]):
  p=int(fit['y'][a:b].sum())
  if 0<p<b-a:keep[a:b]=True
 results=[]
 for name,n in zip(protocol['models'],[19,23]):
  t=time.perf_counter();model=HistGradientBoostingClassifier(**parameters);model.fit(fit['X'][keep,:n],fit['y'][keep]);row={'method':name,'fit_rows':int(keep.sum()),'fit_seconds':time.perf_counter()-t}
  for window in ['eval','later']:
   d=datasets[window];scores=model.predict_proba(d['X'][:,:n])[:,1];metric,pred=evaluate(d,scores);row[window]={'map5':metric};np.save(cache/f'{name}_{window}_scores.npy',scores);(out/f'{name}_{window}_predictions.json').write_text(json.dumps(pred))
  with (cache/f'{name}.pkl').open('wb') as f:pickle.dump(model,f)
  results.append(row);(out/'results.partial.json').write_text(json.dumps(results,indent=2));print(json.dumps(row),flush=True)
 (out/'results.json').write_text(json.dumps({'results':results,'wall_seconds':time.perf_counter()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'code_sha256':hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()},indent=2))
if __name__=='__main__':main()
