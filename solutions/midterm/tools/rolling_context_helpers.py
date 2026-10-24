"""Local copy of base19 builder with explicit rolling origin; originals unchanged."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
import math
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from build_interactions import tick

BASE = Path(__file__).resolve().parents[1]
FEATURES = ['log_count30','log_count80','log_decay60_count30',
 'log_future_airings_per_day','log_past_airings_per_day','log_personal_schedule_score',
 'log_pop_viewers','log_user_count','days_since_last_watch30',
 'log_count10','log_count50','log_decay7_count30','log_decay28_count30',
 'log_distinct_watched_airing_days','log_user_distinct_items',
 'channel_affinity','hour_affinity','channel_hour_product',
 'log_exposure_ratio_smoothed10']
WINDOWS = [('fit','2020-04-13 00:00:00','2020-05-18 00:00:00',True),
 ('eval','2020-05-18 00:00:00','2020-07-27 00:00:00',False),
 ('later','2020-06-22 00:00:00','2020-07-27 00:00:00',False)]


def build_window(db, name, cutoff, finish, train_users, out, evidence, collect_truth=True, extra_candidates=None, history_start=None):
    started = time.perf_counter()
    origin, cut, end = map(tick, [history_start or '2020-03-09 00:00:00',cutoff,finish])
    history = defaultdict(dict)
    channel_counts, hour_counts = defaultdict(Counter), defaultdict(Counter)
    query = '''SELECT o.user,e.item,e.duration,e.channel,e.civil_start,
       MAX(MAX(0,MIN(o.end,?)-MAX(o.start,?))),MAX(MIN(o.end,?))
       FROM overlaps o JOIN events e USING(event_id)
       WHERE o.start<? AND o.end>? GROUP BY o.user,o.event_id'''
    for u,i,d,ch,civil,s,last in db.execute(query,(cut,origin,cut,cut,origin)):
        if s<=.1*d:
            continue
        # count10,count30,count50,count80,decay7,decay28,decay60,last,days
        v=history[u].setdefault(i,[0,0,0,0,0.,0.,0.,0,set()])
        v[0]+=1
        if s<=.3*d:
            continue
        v[1]+=1; v[2]+=s>.5*d; v[3]+=s>.8*d
        age=max(0,cut-last)/86400
        for j,days in enumerate((7,28,60),4):
            v[j]+=math.exp(-math.log(2)*age/days)
        v[7]=max(v[7],last);v[8].add(civil[:10])
        channel_counts[u][ch]+=1;hour_counts[u][int(civil[11:13])]+=1
    truth=defaultdict(set)
    if collect_truth:
        for u,i,d,ch,civil,s,last in db.execute(query,(end,cut,end,end,cut)):
            if s>.8*d:
                truth[u].add(i)
    past=dict(db.execute('SELECT item,COUNT(*) FROM events WHERE start<? AND end>? GROUP BY item',(cut,origin)))
    future=Counter(); item_channels=defaultdict(Counter);item_hours=defaultdict(Counter)
    for i,ch,civil in db.execute('SELECT item,channel,civil_start FROM events WHERE start<? AND end>?',(end,cut)):
        future[i]+=1;item_channels[i][ch]+=1;item_hours[i][int(civil[11:13])]+=1
    item_ids=sorted(future,key=int);item_index={i:j for j,i in enumerate(item_ids)}
    channels=sorted({c for counts in item_channels.values() for c in counts},key=int)
    channel_index={c:j for j,c in enumerate(channels)}
    item_ch=np.zeros((len(item_ids),len(channels)),dtype=np.float32)
    item_hr=np.zeros((len(item_ids),24),dtype=np.float32)
    for i in item_ids:
        j=item_index[i]
        for c,n in item_channels[i].items():item_ch[j,channel_index[c]]=n/future[i]
        for h,n in item_hours[i].items():item_hr[j,h]=n/future[i]
    pop=Counter()
    for h in history.values():pop.update(i for i,v in h.items() if v[1])
    fallback=sorted(item_ids,key=lambda i:(-pop[i],int(i)));glob=fallback[:30]
    target=json.loads((BASE/'evidence/round1/catalog.json').read_text())['users']
    users=sorted((u for u,h in history.items() if any(v[1] for v in h.values())),key=int) if train_users else target
    feature_rows=[];labels=[];row_items=[];offsets=[0];cold=[];truth_counts=[];oracle=[]
    for batch in range(0,len(users),128):
        names=users[batch:batch+128]
        uc=np.zeros((len(names),len(channels)),dtype=np.float32);uh=np.zeros((len(names),24),dtype=np.float32)
        for j,u in enumerate(names):
            total=sum(channel_counts[u].values())
            if total:
                for c,n in channel_counts[u].items():
                    if c in channel_index:uc[j,channel_index[c]]=n/total
                for h,n in hour_counts[u].items():uh[j,h]=n/total
        cs=uc@item_ch.T;hs=uh@item_hr.T;context=cs*hs
        for j,u in enumerate(names):
            hist=history.get(u,{})
            known={i:v for i,v in hist.items() if v[1]}
            total=sum(v[1] for v in known.values())
            def base(i):return known[i][1]*math.sqrt((future[i]/(end-cut))/(past.get(i,1)/(cut-origin)))
            personal=sorted((i for i in known if i in future),key=lambda i:(-base(i),int(i)))[:100]
            ctx=[item_ids[k] for k in np.argsort(-context[j],kind='stable')[:30] if context[j,k]>0]
            extra=[i for i in (extra_candidates or {}).get(u,[]) if i in future]
            cand=list(dict.fromkeys(personal+glob+ctx+extra))
            for i in cand:
                v=hist.get(i,[0,0,0,0,0.,0.,0.,0,set()]); k=item_index[i]
                c,s,r,last=v[1],v[3],v[6],v[7]
                fp=future[i]*86400/(end-cut);pp=past.get(i,0)*86400/(cut-origin)
                score=c*math.sqrt(fp/pp) if pp else 0
                feature_rows.append([math.log1p(c),math.log1p(s),math.log1p(r),math.log1p(fp),math.log1p(pp),
                 math.log1p(score),math.log1p(pop[i]),math.log1p(total),min((cut-last)/86400,365) if last else 365,
                 math.log1p(v[0]),math.log1p(v[2]),math.log1p(v[4]),math.log1p(v[5]),math.log1p(len(v[8])),
                 math.log1p(len(known)),float(cs[j,k]),float(hs[j,k]),float(context[j,k]),
                 math.log1p(((future[i]+10)/(end-cut))/((past.get(i,0)+10)/(cut-origin)))])
                labels.append(i in truth[u]);row_items.append(k)
            offsets.append(len(labels));cold.append(not bool(known));truth_counts.append(len(truth[u]))
            oracle.append(min(5,len(set(cand)&truth[u]))/min(5,len(truth[u])) if truth[u] else 0)
        print(json.dumps({'window':name,'users_prepared':min(batch+128,len(users))}),flush=True) if batch%1024==0 else None
    x=np.asarray(feature_rows,dtype=np.float32); y=np.asarray(labels,dtype=np.uint8)
    assert np.isfinite(x).all() and x.shape==(len(y),len(FEATURES))
    if name=='eval':
        ref=json.loads((BASE/'evidence/strict_labels/truth.json').read_text())
        assert all(set(ref[u])==truth[u] for u in users)
    np.savez_compressed(out/f'{name}.npz', X=x,y=y,group_offsets=np.asarray(offsets,dtype=np.int64),
        users=np.asarray(users),item_ids=np.asarray(item_ids),row_item_indices=np.asarray(row_items,dtype=np.int32),
        cold=np.asarray(cold,dtype=bool),truth_counts=np.asarray(truth_counts,dtype=np.int32),
        fallback_item_indices=np.asarray([item_index[i] for i in fallback],dtype=np.int32),
        feature_names=np.asarray(FEATURES))
    if collect_truth:
        (evidence/f'{name}_truth.json').write_text(json.dumps({u:sorted(truth[u],key=int) for u in users}))
    result={'window':name,'users':len(users),'rows':len(y),'features':len(FEATURES),'positive_rate':float(y.mean()) if collect_truth else None,
        'labels_available':collect_truth,'cold_users':sum(cold),'candidate_oracle_map5':float(np.mean(oracle)) if collect_truth else None,'wall_seconds':time.perf_counter()-started,
        'npz_sha256':hashlib.sha256((out/f'{name}.npz').read_bytes()).hexdigest()}
    print(json.dumps(result),flush=True)
    return result



def weekly_window(db, data, history_start, cutoff, finish, raw, guard):
    """Same weekly5 as v3, with BOTH actual sessions and opportunities clipped."""
    import csv
    from datetime import datetime,timedelta
    from scipy import sparse
    from build_weekly_context import segments
    origin,cut,end=map(tick,[history_start,cutoff,finish])
    recent_start=max(origin,tick(str(datetime.fromisoformat(cutoff)-timedelta(days=28))))
    channels=[r[0] for r in db.execute('SELECT DISTINCT channel FROM events')]
    chidx={c:j for j,c in enumerate(channels)};nslots=len(channels)*168
    users=data['users'];uidx={str(u):j for j,u in enumerate(users)}
    secs={};recent={};dates={};used=0
    with raw.open(encoding='utf-8-sig',newline='') as stream:
        for row,r in enumerate(csv.DictReader(stream)):
            if row%200000==0:guard()
            u=uidx.get(r['user_id']);channel=chidx.get(r['vsetv_id'])
            if u is None or channel is None:continue
            a=max(origin,tick(r['start_time']))
            if a>=cut:continue
            b=min(cut,tick(r['stop_time']))
            if b<=a:continue
            used+=1;weight=min(1.,3600/(b-a));base=u*nslots+channel*168
            for slot,day,left,right in segments(a,b):
                key=base+slot;secs[key]=secs.get(key,0.)+(right-left)*weight
                dates[key]=dates.get(key,0)|(1<<day)
                if right>recent_start:recent[key]=recent.get(key,0.)+(right-max(left,recent_start))*weight
    denom=np.zeros(168);denrecent=np.zeros(168)
    for slot,day,a,b in segments(origin,cut):denom[slot]+=b-a
    for slot,day,a,b in segments(recent_start,cut):denrecent[slot]+=b-a
    keys=np.fromiter(secs,dtype=np.int64,count=len(secs));slot=keys%168
    all_values=np.minimum(1,np.fromiter(secs.values(),dtype=np.float64)/denom[slot])
    recent_values=np.minimum(1,np.array([recent.get(int(k),0.) for k in keys])/denrecent[slot])
    active_values=np.minimum(1,np.array([dates[int(k)].bit_count() for k in keys])/(denom[slot]/3600))
    coords=(keys//nslots,keys%nslots);shape=(len(users),nslots)
    matrices=[sparse.csr_matrix((v.astype(np.float32),coords),shape=shape) for v in (all_values,recent_values,active_values)]
    del secs,recent,dates,keys,all_values,recent_values,active_values
    item_ids=data['item_ids'];ii={str(i):j for j,i in enumerate(item_ids)}
    duration_weights={};fraction_weights={};airings=np.zeros(len(item_ids));totaldur=np.zeros(len(item_ids))
    for i,ch,a,b,duration in db.execute('SELECT item,channel,start,end,duration FROM events WHERE start<? AND end>?',(end,cut)):
        if i not in ii:continue
        j=ii[i];airings[j]+=1;left=max(a,cut);right=min(b,end);totaldur[j]+=right-left
        for slot,day,aa,bb in segments(left,right):
            key=(chidx[ch]*168+slot,j);seconds=bb-aa
            duration_weights[key]=duration_weights.get(key,0)+seconds
            fraction_weights[key]=fraction_weights.get(key,0.)+seconds/duration
    coords=tuple(np.array(v,dtype=np.int32) for v in zip(*duration_weights));epgshape=(nslots,len(item_ids))
    durations=sparse.csr_matrix((np.array(list(duration_weights.values()),dtype=np.float32),coords),shape=epgshape)
    fractions=sparse.csr_matrix((np.array([fraction_weights[k] for k in duration_weights],dtype=np.float32),coords),shape=epgshape)
    expected=(matrices[0]@durations).tocsr();rec_expected=(matrices[1]@durations).tocsr()
    mean_fraction=(matrices[0]@fractions).tocsr();activity=(matrices[2]@durations).tocsr()
    features=np.zeros((len(data['y']),5),dtype=np.float32)
    for j in range(len(users)):
        a,b=data['group_offsets'][j:j+2];ix=data['row_item_indices'][a:b]
        value=expected[j,ix].toarray().ravel();rv=rec_expected[j,ix].toarray().ravel()
        features[a:b,0]=value/(60*((end-cut)/86400))
        features[a:b,1]=mean_fraction[j,ix].toarray().ravel()/airings[ix]
        features[a:b,2]=rv/(60*((end-cut)/86400))
        features[a:b,3]=activity[j,ix].toarray().ravel()/totaldur[ix]
        features[a:b,4]=np.minimum(10,np.divide(rv,value,out=np.zeros_like(rv),where=value>0))
    assert np.isfinite(features).all() and np.all(features>=0)
    return features,used
