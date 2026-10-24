"""Evaluate frozen predictions under plausible preferred-top5 target proxies."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import json,sqlite3,time
from collections import defaultdict
import numpy as np
from check_history_normalization import B,evaluate
from build_interactions import tick
from run_round1 import apk

def main():
 out=B/'evidence/top5_target_audit';out.mkdir(exist_ok=True)
 if (out/'results.json').exists():raise SystemExit('Preserve audit')
 protocol={'motivation':'User official excerpt says5mostpreferred; scorer truth size/preference formula remain undisclosed','proxies':['top5_count80','top5_seconds80','top5_count30','top5_seconds30'],'ties':'ascending numeric programme ID after descending preference proxy','count':'qualified distinct airing events from maxsingle clipped fragment','seconds':'sum maxsinglefragment seconds only for qualified events','thresholds':'strictly >80% or30% full airingduration','no_fitting':True,'warning':'Kaggle score closeness cannot identify true target; same two outcomes cannot validate proxy selection'}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2));start=time.perf_counter();db=sqlite3.connect('file:'+str(B/'cache/interactions.sqlite')+'?mode=ro',uri=True);results={};m=np.load(B/'cache/simple_reranker.npz')
 for name,cutdate in [('eval','2020-05-18'),('later','2020-06-22')]:
  with np.load(B/f'cache/context_features/{name}.npz') as f:d={k:f[k] for k in f.files}
  users=[str(u) for u in d['users']];us=set(users);cut=tick(cutdate+' 00:00:00');end=tick('2020-07-27 00:00:00');summaries={p:defaultdict(list) for p in protocol['proxies']}
  q='''SELECT v.user,e.item,SUM(CASE WHEN v.s>.8*e.duration THEN 1 ELSE 0 END),SUM(CASE WHEN v.s>.8*e.duration THEN v.s ELSE 0 END),SUM(CASE WHEN v.s>.3*e.duration THEN 1 ELSE 0 END),SUM(CASE WHEN v.s>.3*e.duration THEN v.s ELSE 0 END) FROM (SELECT user,event_id,MAX(MAX(0,MIN(end,?)-MAX(start,?))) AS s FROM overlaps WHERE start<? AND end>? GROUP BY user,event_id) v JOIN events e USING(event_id) GROUP BY v.user,e.item'''
  for u,i,*vals in db.execute(q,(end,cut,end,cut)):
   if u in us:
    for p,v in zip(protocol['proxies'],vals):
     if v>0:summaries[p][u].append((v,i))
  truth={p:{u:[i for v,i in sorted(summaries[p][u],key=lambda x:(-x[0],int(x[1])))[:5]] for u in users} for p in protocol['proxies']}
  (out/f'{name}_truth.json').write_text(json.dumps(truth))
  s1=((d['X'][:,:9]-m['mean'])/m['scale'])@m['coefficients'][:-1]+m['coefficients'][-1];s2=np.load(B/f'cache/training_filter/filtered_equal_rows_{name}_scores.npy');fused=np.zeros(len(s1))
  for n in range(len(users)):
   a,b=d['group_offsets'][n:n+2];idx=d['row_item_indices'][a:b]
   for s in [s1,s2]:
    order=sorted(range(b-a),key=lambda j:(-s[a+j],int(d['item_ids'][idx[j]])));r=np.empty(b-a);r[order]=np.arange(1,b-a+1);fused[a:b]+=.5/(60+r)
  methods={}
  for method,s in [('v1_logistic',s1),('filtered_boost',s2),('v2_equal_fusion',fused)]:
   _,pred=evaluate(d,s);methods[method]=pred
  results[name]={method:{p:float(np.mean([apk(t[u],pred[u]) for u in users])) for p,t in truth.items()} for method,pred in methods.items()}
  print(json.dumps({'window':name,'results':results[name]}),flush=True)
 db.close();(out/'results.json').write_text(json.dumps({'results':results,'wall_seconds':time.perf_counter()-start},indent=2))
if __name__=='__main__':main()
