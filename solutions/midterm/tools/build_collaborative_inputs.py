"""Shared sparse historical interactions for collaborative experiments."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import json,sqlite3,time
from pathlib import Path
import numpy as np
from scipy import sparse
from build_interactions import tick
B=Path(__file__).resolve().parents[1]

def main():
 out=B/'cache/collaborative';out.mkdir(exist_ok=True);e=B/'evidence/collaborative_inputs';e.mkdir(exist_ok=True)
 if (e/'results.json').exists():raise SystemExit('Already built')
 protocol={'threshold':'MAX single raw fragment clipped beforecut >30% full airing','windows':[['fit','2020-04-13','2020-05-18'],['eval','2020-05-18','2020-07-27'],['later','2020-06-22','2020-07-27']],'no_future_views':True,'origin':'2020-03-09','repeats':'allowed','missing_session_identity':'stored old-cut fragments not rejoined'}
 (e/'protocol.json').write_text(json.dumps(protocol,indent=2));db=sqlite3.connect('file:'+str(B/'cache/interactions.sqlite')+'?mode=ro',uri=True);results=[]
 for name,date,enddate in protocol['windows']:
  start=time.perf_counter();cut=tick(date+' 00:00:00');end=tick(enddate+' 00:00:00');agg={}
  q='''SELECT o.user,e.item,COUNT(*) FROM (SELECT user,event_id FROM overlaps WHERE start<? AND end>0 GROUP BY user,event_id HAVING MAX(MAX(0,MIN(end,?)-MAX(start,0)))>0.3*(SELECT duration FROM events WHERE events.event_id=overlaps.event_id)) o JOIN events e USING(event_id) GROUP BY o.user,e.item'''
  rows=list(db.execute(q,(cut,cut)))
  with np.load(B/f'cache/context_features/{name}.npz') as f:target=f['users'];candidate=f['item_ids']
  users=sorted({u for u,i,c in rows}|set(target),key=int);items=sorted({i for u,i,c in rows}|set(candidate),key=int);um={u:j for j,u in enumerate(users)};im={i:j for j,i in enumerate(items)}
  mat=sparse.csr_matrix((np.array([c for u,i,c in rows],dtype=np.float32),([um[u] for u,i,c in rows],[im[i] for u,i,c in rows])),shape=(len(users),len(items)))
  sparse.save_npz(out/f'{name}_counts.npz',mat);np.savez_compressed(out/f'{name}_catalog.npz',users=np.array(users),items=np.array(items),target_users=target,target_user_indices=np.array([um[str(u)] for u in target]),candidate_ids=candidate,candidate_item_indices=np.array([im[str(i)] for i in candidate]))
  result={'window':name,'shape':mat.shape,'nnz':mat.nnz,'wall_seconds':time.perf_counter()-start};results.append(result);print(json.dumps(result),flush=True)
 db.close();(e/'results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
