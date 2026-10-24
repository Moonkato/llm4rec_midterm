"""Bounded temporal feature preparation only: no model fitting or submission."""
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


def build_window(db, name, cutoff, finish, train_users, out, evidence, collect_truth=True, extra_candidates=None):
    started = time.perf_counter()
    origin, cut, end = map(tick, ['2020-03-09 00:00:00',cutoff,finish])
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


def main():
    out=BASE/'cache/context_features';evidence=BASE/'evidence/context_features'
    if (evidence/'results.json').exists():raise SystemExit('Preserve completed features.')
    out.mkdir(parents=True,exist_ok=True);evidence.mkdir(parents=True,exist_ok=True)
    protocol={'windows':WINDOWS,'features':FEATURES,'training_runs':0,'submissions':0,
       'labels':'maximum single stored clipped overlap fragment >80% full airing duration',
       'fragment_caveat':'old boundary fragments are not rejoined without original session identity',
       'history':'all actual labelled interactions before each cutoff; MAX per user-airing across stored splits',
       'candidate_pool':'union top100 personal schedule frequency, top30 train popularity, top30 positive channel-hour score',
       'context':'user >30% airing count share by channel/hour multiplied by forecast item airing share; context rank is channel affinity * hour affinity',
       'days_feature':'distinct dates of watched airing starts, not original session dates',
       'future_allowed':'only item IDs, channel, airing time, duration; no future behaviour as features and no future metadata',
       'fallback':'cold users use historical >30% unique viewer popularity, numeric ID ties',
       'format':'X float32; y binary; group_offsets contiguous user groups; row_item_indices index item_ids; Unicode identifiers preserve precision',
       'validation':'forecast truth stored separately for evaluation; fit window labels alone intended for training',
       'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (evidence/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    db=sqlite3.connect('file:'+str(BASE/'cache/interactions.sqlite')+'?mode=ro',uri=True)
    db.execute('PRAGMA temp_store=FILE');db.execute('PRAGMA cache_size=-64000')
    results=[build_window(db,*w,out,evidence) for w in WINDOWS]
    db.close();(evidence/'results.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
