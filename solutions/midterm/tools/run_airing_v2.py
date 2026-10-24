"""Two fixed event representations: conditional recurrence and exact weekly session replay."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import argparse,bisect,csv,hashlib,json,math,pickle,resource,sqlite3,sys,time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
import lightgbm as lgb
from scipy.special import xlogy
from threadpoolctl import threadpool_limits
from build_interactions import tick
from build_weekly_context import segments
from check_history_normalization import evaluate
from run_deep_temporal import Metric
from airing_v2_features import Context, RECURRENCE, REPLAY
from airing_inference_features import eventfeatures as original_eventfeatures
from export_airing_fusion import check_helper_parity
B=Path(__file__).resolve().parents[1];ROOT=B.parents[1]
WINDOWS=[('fit','2020-04-13 00:00:00','2020-05-18 00:00:00'),('eval','2020-05-18 00:00:00','2020-07-27 00:00:00'),('later','2020-06-22 00:00:00','2020-07-27 00:00:00')]
EXTRA=['airing_channel','airing_weekday','airing_hour_sin','airing_hour_cos','log_duration_bin_seconds','boundary_overlap_fraction','history_channel_time_fraction','history_joint_slot_propensity','session_duration_survival80','log_channel_sessions','channel_history_missing','historical_count80_per_airing_smoothed','historical_count30_per_airing_smoothed','user_count30_per_history_day']

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def atomic_json(p,x):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true')
 ap.add_argument('--recurrence-only',action='store_true');args=ap.parse_args();only=args.recurrence_only
 if not args.run:raise SystemExit('Prepared only; root stage approval required for --run')
 temporal_marker=B/'evidence/rolling_temporal/results.json'
 if not temporal_marker.exists() or json.loads(temporal_marker.read_text()).get('status')!='complete':raise SystemExit('Temporal stage must complete first')
 check_helper_parity()
 frozen_run=json.loads((B/'evidence/airing_model/results.json').read_text())
 assert frozen_run['status']=='complete' and frozen_run['code_sha256']==sha(B/'tools/run_airing_model.py')
 frozen_v4={r['window']:r['equal_rrf60_v3'] for r in frozen_run['results']}
 folder='airing_recurrence' if only else 'airing_v2'
 out=B/f'evidence/{folder}';cache=B/f'cache/{folder}'
 if (out/'results.json').exists() or (out/'protocol.json').exists():raise SystemExit('Preserve existing run; explicit recovery required')
 out.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
 began=time.perf_counter();lastlog=[began]
 def log(x):
  x['wall_seconds']=time.perf_counter()-began;print(json.dumps(x),flush=True)
  with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
 def guard(phase=None):
  rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
  if time.perf_counter()-began>(900 if only else 1800) or rss>4000000000:raise RuntimeError('30minute/4GB stage budget exceeded; outputs incomplete')
  if phase and time.perf_counter()-lastlog[0]>25:log({'phase':phase,'rss_bytes':rss});lastlog[0]=time.perf_counter()
 protocol={'model':'two fixed binary LightGBM models: recurrence39, recurrence_replay43; no secondary ranker','windows':WINDOWS,'features':'base33 plus recurrence6 then replay4: '+str(RECURRENCE+REPLAY),
  'parameters':{'objective':'binary','num_leaves':15,'learning_rate':.05,'min_data_in_leaf':100,'lambda_l2':10,'num_iterations':300,'max_bin':63,'num_threads':1,'seed':20261023},
  'universe':'each original v3 user-item candidate crossed with its supplied forecast airings; keep all recorded qualifying positives inside this universe',
  'target':'one MAX clipped stored overlap strictly>80% full airing duration; artificial split fragments not rejoined; absence means no recorded qualifying airing, not dislike',
  'negative_sampling':'independent Bernoulli q=.01 implemented binomial sample count then uniform without replacement among negatives; positive event indices excluded',
  'sample_weights':'positive1, sampled negative100; no class balancing, no is_unbalance, no scale_pos_weight',
  'duration':'Original33 use ceil(duration/300)*300; added replay uses exact physical duration and exact civil start second; labels exact duration',
  'historical_exposure_rates':'(count80+0.1)/(pastairings+1), (count30+0.1)/(pastairings+1), usercount30/historydays; pastairings=expm1(base4)*historydays',
  'context':'history only: user-channel raw session length distribution uncapped, clipped before cutoff; joint channel/weekday/hour seconds use proportional3600s session cap and propensity<=1',
  'collapse_key':['item','channel','weekday','civil_start_hour','duration_bin_seconds','exact_boundary_overlap_fraction','civil_second_of_day','exact_duration_seconds'],
  'exactness':'All varying event features use this tuple; sum probabilities multiplied by identical-tuple multiplicity equals full event scoring for chosen features.',
  'hard_constraint':'event probability forced0 when exact forecast-clipped/full-duration fraction<=.8; recorded strict80 completion impossible within window',
  'aggregation':'sum event probabilities = expected qualifying airing count; no independence assumption and no probability-any claim',
  'primary':'rank expected count on original candidates, common cold popularity fallback','diagnostic':'prespecified equal RRF60 of expectedcount and frozen v3 samepool scores',
  'calibration':'inverse-inclusion weights correct sampled objective; weighted training logloss is descriptive only, not heldout calibration',
  'fits':2,'submissions':0,'budget':'30min4GB1CPU; preprocessing guard before fitting',
  'recurrence':'Complete historical EPG airings only; user-item-channel/hour counts30/80 divided matching schedule exposures; shrinkage5 airings to user-item/channel hierarchy',
  'replay':'Exact historical civil weekday/start-second shifted intervals of actual forecast duration; one Boolean per fully observed historical date; MAX single original raw session, no union or cap; nonexistent DST hour excluded',
  'replay_features':REPLAY,'recurrence_features':RECURRENCE,'frozen_v4_baselines':frozen_v4,
  'recent_replay_window':'28 civil calendar days before cutoff, then mapped to normalized tick','helper_sha256':sha(B/'tools/airing_v2_features.py'),'source_sha256':sha(Path(__file__)),
  'input_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'midterm/data/dataset11-30.csv',B/'cache/interactions.sqlite',B/'tools/run_airing_model.py',B/'tools/airing_inference_features.py',B/'tools/airing_v2_features.py',B/'cache/deep_temporal/top5_count80_24features.txt',B/'evidence/airing_model/results.json',*[B/f'cache/context_features/{n}.npz' for n,_,_ in WINDOWS],*[B/f'cache/deep_temporal/top5_count80_24features_{n}_scores.npy' for n in ('eval','later')]]}}
 if only:
  protocol.update(model='one fixed recurrence39 binary model',features='original33 plus recurrence6',fits=1,budget='15min4GB1CPU',replay='not computed',replay_features=[],primary='fixed equal RRF60 recurrence+v3 compared with frozen v4',collapse_key=protocol['collapse_key'][:-2])
 atomic_json(out/'protocol.json',protocol)
 db=sqlite3.connect('file:'+str(B/'cache/interactions.sqlite')+'?mode=ro',uri=True);db.execute('PRAGMA temp_store=FILE');db.execute('PRAGMA cache_size=-64000')
 channels=sorted((r[0] for r in db.execute('SELECT DISTINCT channel FROM events')),key=int);chmap={c:j for j,c in enumerate(channels)}
 def load(name):
  with np.load(B/f'cache/context_features/{name}.npz',allow_pickle=False) as z:return {k:z[k] for k in z.files}
 def histories(d,cut):
  wanted=set(map(str,d['users']));lens=defaultdict(list);ulens=defaultdict(list);ct=defaultdict(float);slotsec=defaultdict(float)
  with (ROOT/'midterm/data/dataset11-30.csv').open(encoding='utf-8-sig',newline='') as f:
   for n,r in enumerate(csv.DictReader(f)):
    if n%100000==0:guard('historical_context')
    u=r['user_id'];ch=r['vsetv_id']
    if u not in wanted or ch not in chmap:continue
    a=max(0,tick(r['start_time']))
    if a>=cut:continue
    b=min(cut,tick(r['stop_time']))
    if b<=a:continue
    duration=b-a;lens[(u,ch)].append(duration);ulens[u].append(duration);weight=min(1.,3600/duration);ct[(u,ch)]+=duration*weight
    for slot,day,left,right in segments(a,b):slotsec[(u,ch,slot)]+=(right-left)*weight
  for v in lens.values():v.sort()
  for v in ulens.values():v.sort()
  denom=np.zeros(168)
  for slot,day,a,b in segments(0,cut):denom[slot]+=b-a
  for key in slotsec:slotsec[key]=min(1.,slotsec[key]/denom[key[2]])
  guard();log({'phase':'history_ready','users':len(wanted),'channel_groups':len(lens),'slots':len(slotsec)})
  return lens,ulens,ct,slotsec
 def schedule(cut,end):
  events=defaultdict(list);groups=defaultdict(dict)
  for eid,item,ch,a,b,dur,civil in db.execute('SELECT event_id,item,channel,start,end,duration,civil_start FROM events WHERE start<? AND end>?',(end,cut)):
   dt=datetime.fromisoformat(civil);t=(ch,dt.weekday(),dt.hour,((dur+299)//300)*300,(min(end,b)-max(cut,a))/dur,dt.hour*3600+dt.minute*60+dt.second,dur)
   if only:t=t[:5]
   events[item].append((eid,t));groups[item][t]=groups[item].get(t,0)+1
  return events,groups
 def eventfeatures(base,u,tuples,h,cut,item):
  old=original_eventfeatures(base,u,[t[:5] for t in tuples],h,cut,chmap)
  extra=np.asarray([context.recurrence(u,item,t[0],t[2]) for t in tuples],dtype=np.float32) if only else context.features(u,item,tuples)
  return np.concatenate([old,extra],axis=1)
 # Exact finite workload estimate before either model is fit; no user-event matrix.
 sizes=[]
 for name,cut_s,end_s in WINDOWS:
  dd=load(name);ee,gg=schedule(tick(cut_s),tick(end_s));freq=np.bincount(dd['row_item_indices'],minlength=len(dd['item_ids']))
  sizes.append({'window':name,'full_airing_pairs':sum(int(freq[k])*len(ee[str(i)]) for k,i in enumerate(dd['item_ids'])),'collapsed_feature_rows':sum(int(freq[k])*len(gg[str(i)]) for k,i in enumerate(dd['item_ids']))})
  del dd,ee,gg,freq
 atomic_json(out/'workload.json',sizes);log({'phase':'prefit_workload','windows':sizes});guard()
 fit=load('fit');cut,end=map(tick,WINDOWS[0][1:]);events,groups=schedule(cut,end);h=histories(fit,cut);context=Context(db,fit['users'],cut,guard,with_replay=not only)
 event_positions={i:{eid:k for k,(eid,t) in enumerate(ev)} for i,ev in events.items()}
 positive=defaultdict(lambda:defaultdict(list));all_positive=0
 q='''SELECT o.user,e.item,o.event_id,e.duration,MAX(MAX(0,MIN(o.end,?)-MAX(o.start,?))) FROM overlaps o JOIN events e USING(event_id) WHERE o.start<? AND o.end>? GROUP BY o.user,o.event_id'''
 for u,i,eid,dur,s in db.execute(q,(end,cut,end,cut)):
  if s>.8*dur:positive[u][i].append(eid);all_positive+=1
 rng=np.random.default_rng(20261023);samples=[];labels=[];rowowners=[];pairrows=[];exposure=0;kept_positive=0
 for j,u0 in enumerate(fit['users']):
  u=str(u0);a,b=fit['group_offsets'][j:j+2]
  for row in range(a,b):
   i=str(fit['item_ids'][fit['row_item_indices'][row]]);n=len(events[i]);exposure+=n
   pos=np.array(sorted(event_positions[i][eid] for eid in positive.get(u,{}).get(i,[]) if eid in event_positions[i]),dtype=np.int64)
   countneg=n-len(pos);k=rng.binomial(countneg,.01)
   neg=rng.choice(countneg,size=k,replace=False) if k else np.empty(0,dtype=np.int64)
   neg=neg+np.searchsorted(pos-np.arange(len(pos)),neg,side='right')
   selected=np.concatenate([pos,neg]);lab=np.concatenate([np.ones(len(pos),dtype=np.uint8),np.zeros(k,dtype=np.uint8)])
   assert not np.intersect1d(pos,neg).size
   samples.extend((i,int(ix)) for ix in selected);labels.extend(lab);rowowners.extend([j]*len(selected));pairrows.extend([row]*len(selected));kept_positive+=len(pos)
  if j%100==0:guard('sample_training_exposures')
 y=np.asarray(labels,dtype=np.uint8);weights=np.where(y,1.,100.).astype(np.float32)
 x=np.empty((len(y),19+len(EXTRA)+(6 if only else 10)),dtype=np.float32)
 lo=0
 while lo<len(y):
  hi=lo+1
  while hi<len(y) and pairrows[hi]==pairrows[lo]:hi+=1
  row=pairrows[lo];u=str(fit['users'][rowowners[lo]]);tuples=[events[i][ix][1] for i,ix in samples[lo:hi]]
  x[lo:hi]=eventfeatures(fit['X'][row],u,tuples,h,cut,str(fit['item_ids'][fit['row_item_indices'][row]]));lo=hi
  if lo%10000<200:guard('assemble_training_features')
 assert np.isfinite(x).all()
 fit_user_positive=sum(len(v) for u,items in positive.items() if u in set(map(str,fit['users'])) for v in items.values())
 sample_report={'full_exposure_pairs':exposure,'all_window_recorded_positive_airings':all_positive,'pool_positive_airings':kept_positive,'sampled_negatives':int((y==0).sum()),'sampled_rows':len(y),'effective_weight_sum':float(weights.sum()),'weighted_positive_rate':float(np.dot(weights,y)/weights.sum()),'global_window_positive_coverage':kept_positive/all_positive,'fit_user_window_positive_airings':fit_user_positive,'fit_user_pool_positive_coverage':kept_positive/fit_user_positive}
 atomic_json(out/'sampling.json',sample_report);log({'phase':'training_sample_ready',**sample_report})
 del samples,labels,rowowners,pairrows,positive,events,groups,event_positions,h,context
 params={k:v for k,v in protocol['parameters'].items() if k!='num_iterations'};params.update(verbosity=-1,deterministic=True,force_col_wise=True)
 models={};fit_reports={}
 def progress(env):
  if env.iteration%20==0:guard('training')
 # Build once and fit only the two prespecified feature sets, identical sample and weights.
 for variant,nfeatures in ([('recurrence',39)] if only else [('recurrence',39),('recurrence_replay',43)]):
  guard();dataset=lgb.Dataset(x[:,:nfeatures],label=y,weight=weights,categorical_feature=[19,20],free_raw_data=False)
  started=time.perf_counter();model=lgb.train(params,dataset,num_boost_round=300,callbacks=[progress]);fit_seconds=time.perf_counter()-started
  p=np.clip(model.predict(x[:,:nfeatures],num_threads=1),1e-9,1-1e-9)
  loss=float(np.sum(weights*(-xlogy(y,p)-xlogy(1-y,1-p)))/weights.sum())
  model.save_model(str(cache/f'{variant}_model.txt'));models[variant]=model
  fit_reports[variant]={'fit_seconds':fit_seconds,'weighted_training_logloss':loss,'features':nfeatures}
  atomic_json(out/'fit.json',fit_reports);log({'phase':'model_fit_complete','variant':variant,**fit_reports[variant]});del dataset,p
 del x,y,weights,fit
 results=[]
 for name,cut_s,end_s in WINDOWS[1:]:
  d=load(name);cut,end=tick(cut_s),tick(end_s);ev,groups=schedule(cut,end);del ev;h=histories(d,cut);context=Context(db,d['users'],cut,guard,with_replay=not only)
  expected={v:np.zeros(len(d['y']),dtype=np.float64) for v in models};pending=[];dest=[];multiplicity=[];feasible=[];scored=0
  def flush():
   nonlocal scored
   if not pending:return
   matrix=np.concatenate(pending)
   for variant,model in models.items():
    prob=model.predict(matrix[:,:model.num_feature()],num_threads=1);prob[~np.asarray(feasible,dtype=bool)]=0
    np.add.at(expected[variant],np.asarray(dest,dtype=np.int64),prob*np.asarray(multiplicity))
   scored+=len(prob)
   pending.clear();dest.clear();multiplicity.clear();feasible.clear();guard('forecast_airing_batches')
  for j,u0 in enumerate(d['users']):
   u=str(u0);a,b=d['group_offsets'][j:j+2]
   for row in range(a,b):
    i=str(d['item_ids'][d['row_item_indices'][row]]);tuples=list(groups[i]);pending.append(eventfeatures(d['X'][row],u,tuples,h,cut,i));dest.extend([row]*len(tuples));multiplicity.extend(groups[i].values());feasible.extend(t[4]>.8 for t in tuples)
    if len(dest)>=65536:flush()
   if j%100==0:guard('forecast_airing_assembly')
  flush()
  score_v3=np.load(B/f'cache/deep_temporal/top5_count80_24features_{name}_scores.npy')
  truths={'all80':json.loads((B/f'evidence/context_features/{name}_truth.json').read_text()),'top5_count80':json.loads((B/f'evidence/top5_target_audit/{name}_truth.json').read_text())['top5_count80']}
  assert all(set(t)==set(map(str,d['users'])) for t in truths.values())
  metrics={target:Metric(d,value) for target,value in truths.items()}
  report={'window':name,'scored_collapsed_rows':scored,'frozen_v4_equal_rrf60':frozen_v4[name],'variants':{}}
  for variant,scores in expected.items():
   assert np.isfinite(scores).all() and score_v3.shape==scores.shape
   np.save(cache/f'{name}_{variant}_expected_counts.npy',scores)
   fused=np.zeros(len(scores))
   for j in range(len(d['users'])):
    a,b=d['group_offsets'][j:j+2];ids=d['item_ids'][d['row_item_indices'][a:b]]
    for s in (scores,score_v3):
     order=sorted(range(b-a),key=lambda k:(-s[a+k],int(ids[k])));ranks=np.empty(b-a);ranks[order]=np.arange(1,b-a+1);fused[a:b]+=.5/(60+ranks)
   row={}
   for label,s in [('expected_count',scores),('equal_rrf60_v3',fused)]:
    row[label]={target:metric.rank(s)[0] for target,metric in metrics.items()};atomic_json(out/f'{name}_{variant}_{label}_predictions.json',metrics['all80'].predictions(s))
   row['equal_rrf60_delta_vs_frozen_v4']={target:row['equal_rrf60_v3'][target]-frozen_v4[name][target] for target in truths}
   report['variants'][variant]=row
  results.append(report);log(report);del d,h,context,groups,expected,score_v3,fused
 db.close();atomic_json(out/'results.json',{'status':'complete','results':results,'fit_reports':fit_reports,'sampling':sample_report,'wall_seconds':time.perf_counter()-began,'fits':len(models),'submissions':0,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),'code_sha256':sha(Path(__file__))})
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
