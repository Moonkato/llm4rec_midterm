"""Selected fixed factorial arm, inference only; explicit selection evidence required."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import argparse,csv,json,resource,sqlite3,sys,time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
import lightgbm as lgb
from threadpoolctl import threadpool_limits
from run_airing_model import B,ROOT,sha,atomic_json
from airing_inference_features import histories,eventfeatures
from final_replay_features import FinalReplay
from export_airing_fusion import check_helper_parity
from check_history_normalization import evaluate
from build_interactions import tick
ARMS=['v3_old_event','identity_old_event','v3_replay','identity_replay']

def rrf(d,a,b):
 out=np.zeros(len(a))
 for start,end in zip(d['group_offsets'][:-1],d['group_offsets'][1:]):
  ids=np.asarray([int(i) for i in d['item_ids'][d['row_item_indices'][start:end]]])
  for scores in (a,b):
   order=np.lexsort((ids,-scores[start:end]));rank=np.empty(end-start);rank[order]=np.arange(1,end-start+1);out[start:end]+=.5/(60+rank)
 return out

def readpred(path):
 with path.open(newline='') as f:return {r['user_id']:r['tv_show_id'].split() for r in csv.DictReader(f)}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');ap.add_argument('--arm',choices=ARMS);ap.add_argument('--selection-file',type=Path)
 args=ap.parse_args()
 if not args.run:raise SystemExit('Prepared only; explicit --run, --arm and approved selection evidence required')
 if args.arm is None or args.selection_file is None:ap.error('--arm and --selection-file required')
 decision=json.loads(args.selection_file.read_text());assert decision.get('approved_for_export') is True and decision.get('selected_arm')==args.arm
 identity=args.arm.startswith('identity');replay=args.arm.endswith('replay')
 out=B/'evidence/candidate_final_factorial_v5';cache=B/'cache/final_factorial_candidate';dest=B/'candidates/final_factorial_v5.csv'
 if dest.exists() or (out/'protocol.json').exists():raise SystemExit('Preserve existing candidate and evidence')
 started=time.perf_counter();last=[started]
 def guard(phase=None):
  rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
  if time.perf_counter()-started>600 or rss>3000000000:raise RuntimeError('Inference10min3GB budget exceeded')
  if phase and time.perf_counter()-last[0]>25:log({'phase':phase,'peak_rss_bytes':rss});last[0]=time.perf_counter()
 def log(value):
  value['wall_seconds']=time.perf_counter()-started;print(json.dumps(value),flush=True)
  with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
 check_helper_parity()
 inputs=[B/'cache/context_features/test.npz',B/'cache/test_weekly/test_row_features.npy',B/'cache/deep_temporal/top5_count80_24features.txt',B/'cache/airing_candidate/event_test_expected_counts.npy',B/'candidates/weekly_ranker_v3.csv',B/'candidates/event_v3_rrf_v4.csv',B/'cache/interactions.sqlite',ROOT/'midterm/data/submission.csv',Path(__file__),args.selection_file.resolve()]
 if identity:
  result=json.loads((B/'evidence/final_identity/results.json').read_text());assert result['status']=='complete'
  assert result['code_sha256']==sha(B/'tools/run_final_identity.py')
  inputs += [B/'cache/final_identity/identity_ranker.txt',B/'cache/final_identity/categorical_encoders.json',B/'tools/run_final_identity.py']
 if replay:
  result=json.loads((B/'evidence/final_replay/results.json').read_text());assert result['status']=='complete'
  assert result['code_sha256']==sha(B/'tools/run_final_replay.py')
  pp=json.loads((B/'evidence/final_replay/protocol.json').read_text());assert pp['helper_sha256']==sha(B/'tools/final_replay_features.py')
  for dependency in ('airing_inference_features.py','run_airing_model.py','airing_v2_features.py'):
   dependency_path=B/'tools'/dependency
   assert sha(dependency_path)==pp['input_hashes'][str(dependency_path.relative_to(ROOT))], 'Frozen replay dependency changed: '+dependency
  inputs += [B/'cache/final_replay/replay_model.txt',B/'tools/run_final_replay.py',B/'tools/final_replay_features.py',B/'tools/airing_v2_features.py',B/'tools/airing_inference_features.py',B/'tools/run_airing_model.py',ROOT/'midterm/data/dataset11-30.csv']
 out.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
 protocol={'arm':args.arm,'status':'inference only; no submission','selection_evidence':str(args.selection_file),'history_cutoff':'2020-07-27 00:00:00','forecast_end':'2020-10-19 00:00:00','ranking':'fixed equal RRF60, numeric ID ties, unchanged common cold fallback','fits':0,'submissions':0,'budget':'10min3GB1CPU','test_labels_used':False,'sha256':{str(p):sha(p) for p in inputs}}
 atomic_json(out/'protocol.json',protocol)
 with np.load(B/'cache/context_features/test.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
 weekly=np.load(B/'cache/test_weekly/test_row_features.npy');assert d['X'].shape[1]==19 and weekly.shape==(len(d['X']),5)
 x=np.concatenate([d['X'],weekly],axis=1);assert np.isfinite(x).all()
 v3=lgb.Booster(model_file=str(B/'cache/deep_temporal/top5_count80_24features.txt'));assert v3.num_trees()==417 and v3.num_feature()==24
 v3scores=v3.predict(x,num_threads=1);_,v3pred=evaluate(d,v3scores);assert v3pred==readpred(B/'candidates/weekly_ranker_v3.csv')
 cached=B/'cache/airing_candidate/v3_test_scores.npy'
 if cached.exists():assert np.array_equal(v3scores,np.load(cached))
 oldevent=np.load(B/'cache/airing_candidate/event_test_expected_counts.npy');assert oldevent.shape==v3scores.shape and np.isfinite(oldevent).all()
 _,oldpred=evaluate(d,rrf(d,v3scores,oldevent));assert oldpred==readpred(B/'candidates/event_v3_rrf_v4.csv')
 ranker_scores=v3scores;encoding=None
 if identity:
  enc=json.loads((B/'cache/final_identity/categorical_encoders.json').read_text());assert enc['unknown_code']==-1
  usercodes=np.repeat([enc['users'].get(str(u),-1) for u in d['users']],np.diff(d['group_offsets']))
  itemcodes=np.array([enc['items'].get(str(d['item_ids'][i]),-1) for i in d['row_item_indices']])
  xx=np.column_stack([x,usercodes,itemcodes]).astype(np.float32);assert np.array_equal(xx[:,:24],x)
  model=lgb.Booster(model_file=str(B/'cache/final_identity/identity_ranker.txt'));assert model.num_feature()==26 and model.num_trees()==417
  ranker_scores=model.predict(xx,num_threads=1);encoding={'unknown_user_rows':int((usercodes<0).sum()),'unknown_item_rows':int((itemcodes<0).sum())};del xx,model
 del x,weekly,v3
 db=sqlite3.connect('file:'+str(B/'cache/interactions.sqlite')+'?mode=ro',uri=True);db.execute('PRAGMA cache_size=-64000');db.execute('PRAGMA temp_store=FILE')
 cut,end=tick(protocol['history_cutoff']),tick(protocol['forecast_end']);eligible={str(r[0]) for r in db.execute('SELECT DISTINCT item FROM events WHERE start<? AND end>?',(end,cut))}
 eventscores=oldevent;scored=0
 if replay:
  channels=sorted((r[0] for r in db.execute('SELECT DISTINCT channel FROM events')),key=int);chmap={c:j for j,c in enumerate(channels)}
  h=histories(d,cut,chmap,guard,log);context=FinalReplay(d['users'],cut,guard);groups=defaultdict(dict)
  for item,ch,a,b,dur,civil in db.execute('SELECT item,channel,start,end,duration,civil_start FROM events WHERE start<? AND end>?',(end,cut)):
   dt=datetime.fromisoformat(civil);t=(ch,dt.weekday(),dt.hour,((dur+299)//300)*300,(min(end,b)-max(cut,a))/dur,dt.hour*3600+dt.minute*60+dt.second,dur)
   groups[item][t]=groups[item].get(t,0)+1
  model=lgb.Booster(model_file=str(B/'cache/final_replay/replay_model.txt'));assert model.num_feature()==37 and model.num_trees()==300
  eventscores=np.zeros(len(d['X']));pending=[];destrows=[];mult=[];feasible=[]
  def flush():
   nonlocal scored
   if not pending:return
   xx=np.concatenate(pending);p=model.predict(xx,num_threads=1);p[~np.asarray(feasible,dtype=bool)]=0
   np.add.at(eventscores,np.asarray(destrows,dtype=np.int64),p*np.asarray(mult));scored+=len(p)
   pending.clear();destrows.clear();mult.clear();feasible.clear();guard('replay_forecast_batches')
  for j,u0 in enumerate(d['users']):
   u=str(u0);a,b=d['group_offsets'][j:j+2]
   context.prepare_user(u,(t for row in range(a,b) for t in groups[str(d['item_ids'][d['row_item_indices'][row]])]))
   for row in range(a,b):
    item=str(d['item_ids'][d['row_item_indices'][row]]);tuples=list(groups[item])
    original=eventfeatures(d['X'][row],u,[t[:5] for t in tuples],h,cut,chmap)
    pending.append(np.concatenate([original,context.features(u,item,tuples)],axis=1));destrows.extend([row]*len(tuples));mult.extend(groups[item].values());feasible.extend(t[4]>.8 for t in tuples)
    if len(destrows)>=65536:flush()
   if j%100==0:log({'phase':'replay_users','users_done':j});guard()
  flush()
 db.close();assert np.isfinite(ranker_scores).all() and np.isfinite(eventscores).all()
 fused=rrf(d,ranker_scores,eventscores);_,pred=evaluate(d,fused)
 with (ROOT/'midterm/data/submission.csv').open(newline='') as f:
  reader=csv.DictReader(f);fields=reader.fieldnames;users=[r['user_id'] for r in reader]
 assert fields==['user_id','tv_show_id'] and users==list(map(str,d['users'])) and len(users)==len(set(users))==1260 and set(pred)==set(users)
 rows=[]
 for j,u in enumerate(users):
  top=pred[u];a,b=d['group_offsets'][j:j+2]
  assert len(top)==len(set(top))==5 and '0' not in top and set(top)<=eligible
  assert set(top)<=set(map(str,d['item_ids'][d['row_item_indices'][a:b]]))
  if d['cold'][j]:assert top==list(map(str,d['item_ids'][d['fallback_item_indices'][:5]]))
  rows.append({'user_id':u,'tv_show_id':' '.join(top)})
 tmp=dest.with_suffix('.csv.tmp')
 with tmp.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
 with tmp.open(newline='') as f:assert list(csv.DictReader(f))==rows
 guard();tmp.replace(dest)
 for label,scores in [('ranker',ranker_scores),('event',eventscores),('rrf',fused)]:np.save(cache/f'{label}_test_scores.npy',scores)
 manifest={'status':'prepared, not submitted','arm':args.arm,'rows':1260,'fits':0,'submissions':0,'test_labels_used':False,'encoding':encoding,'cold_users':int(d['cold'].sum()),'scored_replay_tuples':scored,'changed_ranked_lists_vs_v4':sum(pred[u]!=oldpred[u] for u in users),'checks':['v3 exact cached scores if available and original CSV parity','original v4 CSV parity','1260 ordered string IDs','5 unique nonzero actual future eligible IDs','candidate membership','cold fallback','finite scores','CSV readback'],'sha256':{**protocol['sha256'],str(dest):sha(dest)},'wall_seconds':time.perf_counter()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)}
 atomic_json(out/'manifest.json',manifest);log(manifest)

if __name__=='__main__':
 with threadpool_limits(limits=1):main()
