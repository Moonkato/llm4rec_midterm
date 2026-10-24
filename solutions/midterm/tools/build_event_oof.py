"""Three user-disjoint event fits and six out-of-fold item summaries; no ranker."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import argparse, hashlib, json, resource, sqlite3, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import lightgbm as lgb
from threadpoolctl import threadpool_limits
from run_airing_model import B, ROOT, sha, atomic_json, WINDOWS
from airing_inference_features import histories, schedule, eventfeatures
from export_airing_fusion import check_helper_parity
from build_interactions import tick
FEATURES=['event_log_expected_count','event_mean_probability','event_max_probability','event_probability_weighted_mean','event_expected_count_percentile','event_expected_count_history80_ratio']
PARAMS=dict(objective='binary',num_leaves=15,learning_rate=.05,min_data_in_leaf=100,lambda_l2=10,max_bin=63,num_threads=1,seed=20261023,verbosity=-1,deterministic=True,force_col_wise=True)

def userfold(u):return int.from_bytes(hashlib.sha256(str(u).encode()).digest()[:8],'big')%3

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true')
 if not ap.parse_args().run:raise SystemExit('Prepared only; root must authorize --run')
 out=B/'evidence/event_oof';cache=B/'cache/event_oof'
 if (out/'protocol.json').exists() or (out/'results.json').exists():raise SystemExit('Preserve existing run')
 out.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
 started=time.perf_counter();last=[started]
 def log(r):
  r['wall_seconds']=time.perf_counter()-started;print(json.dumps(r),flush=True)
  with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
 def guard(phase=None):
  rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
  if time.perf_counter()-started>1200 or rss>3000000000:raise RuntimeError('OOF20minute3GB budget exceeded; incomplete')
  if phase and time.perf_counter()-last[0]>25:log({'phase':phase,'peak_rss_bytes':rss});last[0]=time.perf_counter()
 check_helper_parity()
 prior=json.loads((B/'evidence/airing_model/results.json').read_text());assert prior['status']=='complete' and prior['code_sha256']==sha(B/'tools/run_airing_model.py')
 inputs=[B/'cache/interactions.sqlite',ROOT/'midterm/data/dataset11-30.csv',B/'cache/airing_model/model.txt',B/'tools/airing_inference_features.py',B/'tools/run_airing_model.py',Path(__file__),*[B/f'cache/context_features/{w}.npz' for w,_,_ in WINDOWS],*[B/f'cache/airing_model/{w}_expected_counts.npy' for w in ('eval','later')]]
 protocol={'fits':3,'ranker_fits':0,'submissions':0,'features':FEATURES,'windows':WINDOWS,'parameters':PARAMS,'trees':300,
  'fold':'SHA256 UTF8 user string first8 bytes big-endian modulo3; exclude all sampled exposures of held-out users from corresponding model',
  'sample':'Same seed20261023 as original: all pool positives plus q=.01 Bernoulli negatives via binomial/uniform complement; weights1/100',
  'target':'MAX one stored clipped raw overlap >80% full event duration; no union of artificial split fragments',
  'features_used_for_event_fit':'Exact original33 formulas; histories before Apr13 only; no held-out user target labels fit by its predictor',
  'OOF_score_pool':'All original fit user-item candidate airings, irrespective of labels; each user scored only by held-out-fold model',
  'eval_later':'Frozen original all-April event model; expected-count parity required against saved outputs',
  'OOF_inference_shift':'Meta-training event predictors use2/3users; evaluation original model uses all training users. Standard OOF sample-size/calibration shift; no extra calibration fit or ensemble.',
  'summaries':'log1p sum(m*p), sum(m*p)/sum(m), max(p), sum(m*p*p)/(sum(m*p)+1e-12), within-user candidate percentile descending numeric-ID ties, sum(m*p)/(1+count80*forecastdays/historydays)',
  'interpretation':'Descriptive model-score summaries; no independence, calibrated probability-any, or true uncertainty claim',
  'boundary':'Set p=0 if exact clipped forecast fraction<=.8; original5tuple collapse and multiplicity',
  'percentile':'1-rank0/(n-1) within original candidate pool; stable numeric item ID ties; one-item pool1',
  'days':'Physical elapsed normalized tick seconds/86400, consistent with original history rate features',
  'budget':'20min3GB1CPU','input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in inputs}}
 atomic_json(out/'protocol.json',protocol)
 db=sqlite3.connect('file:'+str(B/'cache/interactions.sqlite')+'?mode=ro',uri=True);db.execute('PRAGMA cache_size=-64000');db.execute('PRAGMA temp_store=FILE')
 channels=sorted((r[0] for r in db.execute('SELECT DISTINCT channel FROM events')),key=int);chmap={c:j for j,c in enumerate(channels)}
 def load(w):
  with np.load(B/f'cache/context_features/{w}.npz',allow_pickle=False) as z:return {k:z[k] for k in z.files}
 fit=load('fit');cut,end=map(tick,WINDOWS[0][1:]);events,groups=schedule(cut,end,db);h=histories(fit,cut,chmap,guard,log)
 folds=np.array([userfold(u) for u in fit['users']],dtype=np.int8)
 atomic_json(out/'folds.json',{str(u):int(f) for u,f in zip(fit['users'],folds)})
 positions={i:{eid:k for k,(eid,t) in enumerate(ev)} for i,ev in events.items()};positive=defaultdict(lambda:defaultdict(list))
 query='''SELECT o.user,e.item,o.event_id,e.duration,MAX(MAX(0,MIN(o.end,?)-MAX(o.start,?))) FROM overlaps o JOIN events e USING(event_id) WHERE o.start<? AND o.end>? GROUP BY o.user,o.event_id'''
 for u,i,eid,dur,s in db.execute(query,(end,cut,end,cut)):
  if s>.8*dur:positive[u][i].append(eid)
 rng=np.random.default_rng(20261023);samples=[];labels=[];owners=[];pairrows=[];exposures=0
 for j,u0 in enumerate(fit['users']):
  u=str(u0);a,b=fit['group_offsets'][j:j+2]
  for row in range(a,b):
   i=str(fit['item_ids'][fit['row_item_indices'][row]]);n=len(events[i]);exposures+=n
   pos=np.array(sorted(positions[i][eid] for eid in positive.get(u,{}).get(i,[]) if eid in positions[i]),dtype=np.int64)
   countneg=n-len(pos);k=rng.binomial(countneg,.01);neg=rng.choice(countneg,size=k,replace=False) if k else np.empty(0,dtype=np.int64)
   neg=neg+np.searchsorted(pos-np.arange(len(pos)),neg,side='right');assert not np.intersect1d(pos,neg).size
   chosen=np.concatenate([pos,neg]);labels.extend(np.concatenate([np.ones(len(pos),dtype=np.uint8),np.zeros(k,dtype=np.uint8)]))
   samples.extend((i,int(ix)) for ix in chosen);owners.extend([j]*len(chosen));pairrows.extend([row]*len(chosen))
  if j%100==0:guard('sample_training')
 y=np.asarray(labels,dtype=np.uint8);owners=np.asarray(owners,dtype=np.int32);weights=np.where(y,1.,100.).astype(np.float32);sample_folds=folds[owners]
 x=np.empty((len(y),33),dtype=np.float32);lo=0
 while lo<len(y):
  hi=lo+1
  while hi<len(y) and pairrows[hi]==pairrows[lo]:hi+=1
  row=pairrows[lo];u=str(fit['users'][owners[lo]]);tuples=[events[i][ix][1] for i,ix in samples[lo:hi]]
  x[lo:hi]=eventfeatures(fit['X'][row],u,tuples,h,cut,chmap);lo=hi
  if lo%10000<200:guard('assemble_training')
 assert np.isfinite(x).all()
 sampling={'rows':len(y),'positives':int(y.sum()),'negatives':int((y==0).sum()),'full_exposures':exposures}
 assert sampling['rows']==prior['sampling']['sampled_rows'] and sampling['positives']==prior['sampling']['pool_positive_airings']
 atomic_json(out/'sampling.json',sampling);log({'phase':'sample_ready',**sampling})
 del samples,labels,pairrows,positive,positions,events,owners
 models={};fit_reports=[]
 for fold in range(3):
  train=sample_folds!=fold;held_users=set(map(str,fit['users'][folds==fold]));train_users=set(map(str,fit['users'][folds!=fold]));assert not held_users&train_users
  dataset=lgb.Dataset(x[train],label=y[train],weight=weights[train],categorical_feature=[19,20],free_raw_data=True)
  def progress(env):
   if env.iteration%20==0:guard('training_fold')
  began=time.perf_counter();model=lgb.train(PARAMS,dataset,num_boost_round=300,callbacks=[progress]);path=cache/f'fold{fold}_model.txt';model.save_model(str(path));models[fold]=model
  record={'fold':fold,'train_users':len(train_users),'heldout_users':len(held_users),'train_rows':int(train.sum()),'heldout_sample_rows':int((~train).sum()),'user_overlap':0,'fit_seconds':time.perf_counter()-began,'model_sha256':sha(path)}
  fit_reports.append(record);atomic_json(out/'fits.json',fit_reports);log({'phase':'fit_complete',**record});del dataset,train,held_users,train_users
 del x,y,weights,sample_folds
 frozen=lgb.Booster(model_file=str(B/'cache/airing_model/model.txt'));assert frozen.num_feature()==33 and frozen.num_trees()==300
 results=[]
 for name,cut_s,end_s in WINDOWS:
  if name=='fit':d=fit
  else:
   del h,groups
   d=load(name);cut,end=tick(cut_s),tick(end_s);ev,groups=schedule(cut,end,db);del ev;h=histories(d,cut,chmap,guard,log)
  nrows=len(d['X']);sums=np.zeros(nrows);squares=np.zeros(nrows);maximum=np.zeros(nrows);airings=np.zeros(nrows);assembled=np.zeros(nrows,dtype=np.uint8)
  pending=[];dest=[];multiplicity=[];feasible=[];model_ids=[];scored=0
  def flush():
   nonlocal scored
   if not pending:return
   xx=np.concatenate(pending);which=np.asarray(model_ids,dtype=np.int8);p=np.empty(len(xx))
   for fold in np.unique(which):
    mask=which==fold;predictor=models[int(fold)] if name=='fit' else frozen;p[mask]=predictor.predict(xx[mask],num_threads=1)
   p[~np.asarray(feasible,dtype=bool)]=0
   ix=np.asarray(dest,dtype=np.int64);m=np.asarray(multiplicity);np.add.at(sums,ix,m*p);np.add.at(squares,ix,m*p*p);np.maximum.at(maximum,ix,p);np.add.at(airings,ix,m);scored+=len(p)
   pending.clear();dest.clear();multiplicity.clear();feasible.clear();model_ids.clear();guard('aggregate_forecast')
  for j,u0 in enumerate(d['users']):
   u=str(u0);a,b=d['group_offsets'][j:j+2];fold=userfold(u) if name=='fit' else -1
   for row in range(a,b):
    i=str(d['item_ids'][d['row_item_indices'][row]]);tuples=list(groups[i]);pending.append(eventfeatures(d['X'][row],u,tuples,h,cut,chmap))
    dest.extend([row]*len(tuples));multiplicity.extend(groups[i].values());feasible.extend(t[4]>.8 for t in tuples);model_ids.extend([fold]*len(tuples));assembled[row]+=1
    if len(dest)>=65536:flush()
   if j%100==0:guard('assemble_forecast')
  flush();assert np.all(assembled==1) and np.all(airings>0)
  parity=None
  if name!='fit':
   reference=np.load(B/f'cache/airing_model/{name}_expected_counts.npy');assert reference.shape==sums.shape
   parity=float(np.max(np.abs(sums-reference)));assert np.allclose(sums,reference,rtol=1e-10,atol=1e-10)
  features=np.empty((nrows,6),dtype=np.float32);features[:,0]=np.log1p(sums);features[:,1]=sums/airings;features[:,2]=maximum;features[:,3]=squares/(sums+1e-12)
  for a,b in zip(d['group_offsets'][:-1],d['group_offsets'][1:]):
   ids=np.asarray([int(i) for i in d['item_ids'][d['row_item_indices'][a:b]]]);order=np.lexsort((ids,-sums[a:b]));rank=np.empty(b-a);rank[order]=np.arange(b-a);features[a:b,4]=1-rank/max(1,b-a-1)
  features[:,5]=sums/(1+np.expm1(d['X'][:,1].astype(np.float64))*(end-cut)/cut)
  assert np.isfinite(features).all() and np.all(features>=0)
  path=cache/f'{name}_features.npy';tmp=path.with_suffix('.npy.tmp')
  with tmp.open('wb') as f:np.save(f,features)
  tmp.replace(path);np.save(cache/f'{name}_expected_counts.npy',sums)
  record={'window':name,'rows':nrows,'features':FEATURES,'scored_collapsed_rows':scored,'all_rows_assembled_once':True,'frozen_expected_count_max_error':parity,'feature_sha256':sha(path),'source_sha256':sha(B/f'cache/context_features/{name}.npz')}
  results.append(record);atomic_json(out/'results.partial.json',results);log(record)
  del sums,squares,maximum,airings,assembled,features
 db.close();atomic_json(out/'results.json',{'status':'complete','fits':3,'fit_reports':fit_reports,'results':results,'feature_names':FEATURES,'wall_seconds':time.perf_counter()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),'code_sha256':sha(Path(__file__))})

if __name__=='__main__':
 with threadpool_limits(limits=1):main()
