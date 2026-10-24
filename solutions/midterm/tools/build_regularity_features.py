"""Historical item regularity only; fixed aligned features, no fitting."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import json,sqlite3,time,hashlib,resource,sys,math
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timedelta
import numpy as np
from build_interactions import tick
B=Path(__file__).resolve().parents[1]
WINDOWS=[('inner_fit','2020-04-13 00:00:00','deep_context'),('inner_stop','2020-05-04 00:00:00','deep_context'),('fit','2020-04-13 00:00:00','context_features'),('eval','2020-05-18 00:00:00','context_features'),('later','2020-06-22 00:00:00','context_features')]
NAMES=['active_history_weeks_fraction30','active_last4weeks_fraction30','recent14_previous14_count30_ratio_smoothed','count30_per_history_day','item_share_user_count30','completion80_given30_smoothed','completion30_given10_smoothed','mean_maxsingle_completion_gt10','recent28_completion80_given30_smoothed','weeks_since_first30_normalized']

def main():
    started=time.perf_counter();out=B/'cache/regularity_features';ev=B/'evidence/regularity_features'
    if (ev/'results.json').exists():raise SystemExit('Preserve completed regularity features')
    out.mkdir(parents=True,exist_ok=True);ev.mkdir(parents=True,exist_ok=True)
    protocol={'windows':WINDOWS,'features':NAMES,'fits':0,'submissions':0,'alignment':'base19 context group/user/item row order',
      'events':'MAX clipped stored raw overlap per user-airing; denominator full airing duration; no fragment union',
      'timing':'event last endpoint clipped at cutoff assigns week and recent-window membership; point at cutoff belongs to preceding instant',
      'opportunities':'calendar weeks since March9, including partial week; last4weeks uses last min(28,historydays) calendar days',
      'trend':'(recent14 count30+1)/(preceding14 count30+1); no clipping',
      'rates':'historyday denominator is elapsed civil calendar days, not number of active days',
      'ratios':'(count80+1)/(count30+2), (count30+1)/(count10+2), (recent28count80+1)/(recent28count30+2)',
      'completion':'mean maxsingle overlap/full duration for events >10%; thresholds strictly greater',
      'first_age':'(cutoff-first qualifying max-overlap endpoint)/(7days), divided by historical opportunityweeks; no history=>0',
      'future_inputs':'none; row candidates already supplied, no forecast behaviour queried',
      'groups':{'regularity':[0,1,9],'trend':[2],'intensity':[3,4],'completion':[5,6,7,8]},'budget':'5min1GB'}
    (ev/'protocol.json').write_text(json.dumps(protocol,indent=2))
    db=sqlite3.connect('file:'+str(B/'cache/interactions.sqlite')+'?mode=ro',uri=True);db.execute('PRAGMA temp_store=FILE');db.execute('PRAGMA cache_size=-64000')
    query='''SELECT o.user,e.item,e.duration,MAX(MAX(0,MIN(o.end,?)-MAX(o.start,0))),MAX(MIN(o.end,?)) FROM overlaps o JOIN events e USING(event_id) WHERE o.start<? AND o.end>0 GROUP BY o.user,o.event_id'''
    computed={};results=[]
    for name,cut_s,source in WINDOWS:
        began=time.perf_counter();cut=tick(cut_s);dt=datetime.fromisoformat(cut_s);historydays=(dt-datetime(2020,3,9)).days;weeks=math.ceil(historydays/7)
        if cut not in computed:
            cutoff14=tick(str(dt-timedelta(days=14)));cutoff28=tick(str(dt-timedelta(days=28)))
            hist=defaultdict(dict);totals=Counter()
            for n,(u,i,d,s,last) in enumerate(db.execute(query,(cut,cut,cut))):
                if n%100000==0:
                    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
                    if time.perf_counter()-started>300 or rss>1_000_000_000:raise RuntimeError('Preparation budget exceeded; no fit')
                if s<=.1*d:continue
                # c10,c30,c80,sumcompletion,weeksbits,last4bits,recent14,prev14,recent28c30,recent28c80,first
                v=hist[u].setdefault(i,[0,0,0,0.,0,0,0,0,0,0,cut]);v[0]+=1;v[3]+=s/d
                if s<=.3*d:continue
                v[1]+=1;v[2]+=s>.8*d;totals[u]+=1;v[10]=min(v[10],last)
                # Tick's one missing DST hour is restored only for calendar bin assignment.
                civil_last=last+(3600 if last>=tick('2020-03-29 04:00:00') else 0)
                day=max(0,(civil_last-1)//86400);v[4]|=1<<(day//7)
                if last>cutoff28:
                    v[5]|=1<<((day-max(0,historydays-28))//7);v[8]+=1;v[9]+=s>.8*d
                    if last>cutoff14:v[6]+=1
                    else:v[7]+=1
            computed[cut]=(hist,totals)
        hist,totals=computed[cut]
        with np.load(B/f'cache/{source}/{name}.npz',allow_pickle=False) as z:d={k:z[k] for k in ('users','group_offsets','item_ids','row_item_indices','y')}
        features=np.zeros((len(d['y']),10),dtype=np.float32)
        # Smoothed completion fractions have an explicit Beta(1,1) prior for missing histories.
        features[:,[5,6,8]]=.5;features[:,2]=1
        for n,u in enumerate(d['users']):
            a,b=d['group_offsets'][n:n+2];h=hist.get(str(u),{})
            for row in range(a,b):
                v=h.get(str(d['item_ids'][d['row_item_indices'][row]]))
                if v is None:continue
                c10,c30,c80,completion,wbits,lbits,r14,p14,r30,r80,first=v
                features[row]=[wbits.bit_count()/weeks,lbits.bit_count()/min(4,weeks),(r14+1)/(p14+1),c30/historydays,
                    c30/totals[str(u)] if totals[str(u)] else 0,(c80+1)/(c30+2),(c30+1)/(c10+2),completion/c10,
                    (r80+1)/(r30+2),max(0,(cut-first)/(7*86400*weeks)) if c30 else 0]
        assert np.isfinite(features).all() and np.all(features>=0)
        np.save(out/f'{name}_row_features.npy',features)
        summary={'window':name,'rows':len(features),'features':NAMES,'history_days':historydays,'opportunity_weeks':weeks,'wall_seconds':time.perf_counter()-began,
          'source_sha256':hashlib.sha256((B/f'cache/{source}/{name}.npz').read_bytes()).hexdigest(),'sha256':hashlib.sha256((out/f'{name}_row_features.npy').read_bytes()).hexdigest()}
        results.append(summary);print(json.dumps(summary),flush=True)
        # Retain Apr13 only until fit; discard other windows promptly.
        if name in ('fit','eval','later'):computed.pop(cut,None)
    db.close();(ev/'results.json').write_text(json.dumps({'results':results,'wall_seconds':time.perf_counter()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
if __name__=='__main__':main()
