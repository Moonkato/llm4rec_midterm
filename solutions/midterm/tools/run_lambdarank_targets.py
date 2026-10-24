"""Two fixed LambdaRank fits with different documented preference proxies."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
import json,sqlite3,time,hashlib,resource
from collections import defaultdict
import numpy as np
import lightgbm as lgb
from check_history_normalization import B,evaluate
from build_interactions import tick
from run_round1 import apk

def main():
 out=B/'evidence/lambdarank_targets';cache=B/'cache/lambdarank_targets';out.mkdir(exist_ok=True);cache.mkdir(exist_ok=True)
 if (out/'results.json').exists():raise SystemExit('Preserve completed round')
 params={'objective':'lambdarank','n_estimators':200,'learning_rate':.04,'num_leaves':15,'max_depth':4,'min_child_samples':50,'reg_lambda':10.,'random_state':20261023,'n_jobs':2,'verbosity':-1,'deterministic':True,'force_col_wise':True,'label_gain':[0,1],'lambdarank_truncation_level':10}
 protocol={'fit_window':['2020-04-13','2020-05-18'],'variants':['all80','top5_count80'],'features':'23 collaborative features, historical only','parameters':params,'query_group':'user; remove groups without either class in each fit','top5_ties':'numeric item ID after descending number of >80% user-airing events','evaluation':'both all80 and4fixed top5proxies on two explored windows; no metric swapping','objective_note':'LambdaRank uses NDCG-based pair weights, not exact MAP loss; report own MAP@5','hidden_target':'not confirmed; output5 does not prove scorertruth5','submissions':0}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2));start=time.perf_counter();ds={}
 for name in ['fit','eval','later']:
  with np.load(B/f'cache/collaborative_features/{name}.npz') as f:ds[name]={k:f[k] for k in f.files}
 fit=ds['fit'];cut=tick('2020-04-13 00:00:00');end=tick('2020-05-18 00:00:00');db=sqlite3.connect('file:'+str(B/'cache/interactions.sqlite')+'?mode=ro',uri=True);counts=defaultdict(list)
 q='''SELECT o.user,e.item,COUNT(*) FROM (SELECT user,event_id FROM overlaps WHERE start<? AND end>? GROUP BY user,event_id HAVING MAX(MAX(0,MIN(end,?)-MAX(start,?)))>.8*(SELECT duration FROM events WHERE events.event_id=overlaps.event_id)) o JOIN events e USING(event_id) GROUP BY o.user,e.item'''
 for u,i,n in db.execute(q,(end,cut,end,cut)):counts[u].append((n,i))
 db.close();truth={u:{i for n,i in sorted(v,key=lambda x:(-x[0],int(x[1])))[:5]} for u,v in counts.items()};ytop=np.zeros(len(fit['y']),dtype=np.uint8)
 for j,u in enumerate(fit['users']):
  a,b=fit['group_offsets'][j:j+2];ytop[a:b]=[str(fit['item_ids'][idx]) in truth.get(str(u),set()) for idx in fit['row_item_indices'][a:b]]
 assert np.all(ytop<=fit['y'])
 (out/'fit_top5_truth.json').write_text(json.dumps({u:sorted(t,key=int) for u,t in truth.items()}))
 results=[]
 for variant,y in [('all80',fit['y']),('top5_count80',ytop)]:
  keep=np.zeros(len(y),dtype=bool);groups=[]
  for a,b in zip(fit['group_offsets'][:-1],fit['group_offsets'][1:]):
   p=int(y[a:b].sum())
   if 0<p<b-a:keep[a:b]=True;groups.append(int(b-a))
  assert sum(groups)==int(keep.sum());t=time.perf_counter();model=lgb.LGBMRanker(**params);model.fit(fit['X'][keep],y[keep],group=groups)
  row={'variant':variant,'fit_rows':int(keep.sum()),'fit_groups':len(groups),'fit_seconds':time.perf_counter()-t}
  for name in ['eval','later']:
   d=ds[name];score=model.predict(d['X']);broad,pred=evaluate(d,score);proxy=json.loads((B/f'evidence/top5_target_audit/{name}_truth.json').read_text());row[name]={'all80_map5':broad,**{p:float(np.mean([apk(t[str(u)],pred[str(u)]) for u in d['users']])) for p,t in proxy.items()}};np.save(cache/f'{variant}_{name}_scores.npy',score);(out/f'{variant}_{name}_predictions.json').write_text(json.dumps(pred))
  model.booster_.save_model(str(cache/f'{variant}.txt'));results.append(row);(out/'results.partial.json').write_text(json.dumps(results,indent=2));print(json.dumps(row),flush=True)
 (out/'results.json').write_text(json.dumps({'results':results,'lightgbm_version':lgb.__version__,'wall_seconds':time.perf_counter()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'code_sha256':hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()},indent=2))
if __name__=='__main__':main()
