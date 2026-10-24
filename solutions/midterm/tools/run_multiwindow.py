"""Recent-only versus pooled-window training, evaluated strictly after June22."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
import json,pickle,sqlite3,time,resource,hashlib
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from build_context_features import build_window,BASE
from check_history_normalization import evaluate

def main():
 out=BASE/'evidence/multiwindow';cache=BASE/'cache/multiwindow';out.mkdir(exist_ok=True);cache.mkdir(exist_ok=True)
 if (out/'results.json').exists():raise SystemExit('Preserve completed round')
 protocol={'early_fit':['2020-04-13','2020-05-18'],'recent_fit':['2020-05-18','2020-06-22'],'evaluation':['2020-06-22','2020-07-27'],'fit_rows':'all historical users, distinct chronological snapshots with overlapping users/history; no synthetic labels','models':['recent_only','early_plus_recent'],'parameters':{'learning_rate':.06,'max_iter':150,'max_leaf_nodes':15,'max_depth':3,'min_samples_leaf':50,'l2_regularization':10.,'early_stopping':False,'random_state':20261022},'validation_caveat':'later window previously inspected but never included in fit labels; original May18July27 window cannot be used to evaluate these fitted models','submissions':0}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2));start=time.perf_counter()
 db=sqlite3.connect('file:'+str(BASE/'cache/interactions.sqlite')+'?mode=ro',uri=True)
 prep=build_window(db,'recent_fit','2020-05-18 00:00:00','2020-06-22 00:00:00',True,cache,out);db.close()
 def load(p):
  with np.load(p) as f:return {k:f[k] for k in f.files}
 early=load(BASE/'cache/context_features/fit.npz');recent=load(cache/'recent_fit.npz');later=load(BASE/'cache/context_features/later.npz')
 assert np.array_equal(early['feature_names'],recent['feature_names']) and np.array_equal(early['feature_names'],later['feature_names'])
 results=[]
 for name,sources in [('recent_only',[recent]),('early_plus_recent',[early,recent])]:
  x=np.concatenate([d['X'] for d in sources]);y=np.concatenate([d['y'] for d in sources]);t=time.perf_counter()
  model=HistGradientBoostingClassifier(**protocol['parameters']);model.fit(x,y)
  score=model.predict_proba(later['X'])[:,1];metric,pred=evaluate(later,score)
  row={'model':name,'fit_rows':len(y),'later_map5':metric,'fit_and_score_seconds':time.perf_counter()-t};results.append(row);print(json.dumps(row),flush=True)
  with (cache/f'{name}.pkl').open('wb') as f:pickle.dump(model,f)
  np.save(cache/f'{name}_later_scores.npy',score);(out/f'{name}_later_predictions.json').write_text(json.dumps(pred))
  (out/'results.partial.json').write_text(json.dumps(results,indent=2))
 (out/'results.json').write_text(json.dumps({'results':results,'preprocessing':prep,'wall_seconds':time.perf_counter()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
if __name__=='__main__':main()
