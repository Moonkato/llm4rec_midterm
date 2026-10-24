"""Independent deterministic raw-log join sample; never modifies model caches."""
import bisect,csv,hashlib,json,random,sqlite3,time
from collections import Counter,defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy import sparse
B=Path(__file__).resolve().parents[1];ROOT=B.parents[1]
def stamp(s):
    assert not s.startswith('2020-03-29 03:')
    return int((datetime.fromisoformat(s)-datetime(2020,3,9)).total_seconds())-(3600 if s>='2020-03-29 04:00:00' else 0)
def main():
    began=time.perf_counter();out=B/'evidence/deep_join_audit.json'
    if out.exists():raise SystemExit('Preserve completed audit')
    with np.load(B/'cache/collaborative/eval_catalog.npz') as z:cat={k:z[k] for k in z.files}
    m=sparse.load_npz(B/'cache/collaborative/eval_counts.npz');activity=np.asarray(m[cat['target_user_indices']].sum(axis=1)).ravel()
    ordered=sorted(range(len(activity)),key=lambda j:(activity[j],int(cat['target_users'][j])))
    selected=[str(cat['target_users'][ordered[round(q*(len(ordered)-1)/9)]]) for q in range(10)];allowed=set(selected)
    protocol={'selection':'10 evenly spaced activity order statistics incl light/heavy; activity uses historical >30% counts strictly beforeMay18; numeric user ID ties','users':selected,'seed':20261023,'schedule_sample':100,'no_model_fitting':True,'no_test_views':True}
    (B/'evidence/deep_join_audit_protocol.json').write_text(json.dumps(protocol,indent=2))
    db=sqlite3.connect('file:'+str(B/'cache/interactions.sqlite')+'?mode=ro',uri=True)
    origin=stamp('2020-03-09 00:00:00');old=stamp('2020-05-18 00:00:00');later=stamp('2020-06-22 00:00:00');end=stamp('2020-07-27 00:00:00')
    windows={'history_may18':(origin,old),'forecast_may18':(old,end),'history_june22':(origin,later),'forecast_june22':(later,end)}
    events={r[0]:r for r in db.execute('SELECT event_id,channel,item,start,end,duration,civil_start FROM events WHERE start<?',(end,))}
    schedule_sample=random.Random(20261023).sample(sorted(events),100)
    wanted={(events[e][1],events[e][2],events[e][6]):e for e in schedule_sample};matched=set();schedule_errors=[]
    for path in sorted((ROOT/'midterm/data').glob('export*')):
        if '31-42' in path.name:continue
        with path.open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                s=row['start_time'];civil=s[6:10]+'-'+s[3:5]+'-'+s[:2]+' '+s[11:]
                key=(row['channel_id'],row['tv_show_id'],civil)
                if key in wanted:
                    eid=wanted[key];e=events[eid];matched.add(eid)
                    if stamp(civil)!=e[3] or int(row['duration'])!=e[5] or e[4]!=stamp(civil)+int(row['duration']):schedule_errors.append({'event_id':eid,'row':row})
    print(json.dumps({'phase':'schedule_checked','matched':len(matched)}),flush=True)
    bychannel=defaultdict(list)
    for eid,e in events.items():bychannel[e[1]].append((e[3],e[4],eid))
    lookup={}
    for ch,seq in bychannel.items():
        seq.sort();lookup[ch]=([e[0] for e in seq],seq,max(b-a for a,b,_ in seq))
    sessions=defaultdict(list);raw={n:{} for n in windows};rows=0;duration_bad=0
    with (ROOT/'midterm/data/dataset11-30.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['user_id'] not in allowed:continue
            u=r['user_id'];a=stamp(r['start_time']);b=stamp(r['stop_time']);ch=r['vsetv_id'];rows+=1
            duration_bad+=b-a!=int(r['duraton']);sessions[u].append((a,b,ch))
            if ch not in lookup:continue
            starts,seq,longest=lookup[ch];lo=bisect.bisect_left(starts,a-longest);hi=bisect.bisect_left(starts,b)
            for ea,eb,eid in seq[lo:hi]:
                left,right=max(a,ea,origin),min(b,eb,end)
                if left>=right:continue
                key=(u,eid)
                for name,(wa,wb) in windows.items():
                    seconds=max(0,min(right,wb)-max(left,wa))
                    if seconds>raw[name].get(key,0):raw[name][key]=seconds
    cached={n:{} for n in windows};marks=','.join('?' for _ in selected)
    for u,eid,a,b in db.execute(f'SELECT user,event_id,start,end FROM overlaps WHERE user IN ({marks})',selected):
        for name,(wa,wb) in windows.items():
            seconds=max(0,min(b,wb)-max(a,wa));key=(u,eid)
            if seconds>cached[name].get(key,0):cached[name][key]=seconds
    db.close();comparison={}
    for name in windows:
        differing=[k for k in raw[name].keys()|cached[name].keys() if raw[name].get(k,0)!=cached[name].get(k,0)]
        threshold_results={}
        for numerator,denominator in [(3,10),(4,5)]:
            rs={k for k,s in raw[name].items() if denominator*s>numerator*events[k[1]][5]};cs={k for k,s in cached[name].items() if denominator*s>numerator*events[k[1]][5]}
            ritems={(u,events[eid][2]) for u,eid in rs};citems={(u,events[eid][2]) for u,eid in cs}
            threshold_results[str(numerator/denominator)]={'raw_positive_airings':len(rs),'cached_positive_airings':len(cs),'raw_only_airings':len(rs-cs),'cache_only_airings':len(cs-rs),'raw_only_user_items':len(ritems-citems),'cache_only_user_items':len(citems-ritems)}
        examples=[]
        for u,eid in sorted(differing)[:15]:
            e=events[eid];examples.append({'user':u,'event_id':eid,'item':e[2],'channel':e[1],'civil_start':e[6],'duration':e[5],'raw_max_overlap':raw[name].get((u,eid),0),'cached_max_fragment':cached[name].get((u,eid),0),'event_crosses_original_may18_boundary':e[3]<old<e[4]})
        comparison[name]={'different_max_overlap_groups':len(differing),'thresholds':threshold_results,'examples':examples}
    duplicates=Counter();conflicts=Counter();conflict_examples=[]
    for u,seq in sessions.items():
        counts=Counter(seq);duplicates['exact_duplicate_extra_rows']+=sum(n-1 for n in counts.values());active=[]
        for a,b,ch in sorted(set(seq)):
            active=[x for x in active if x[1]>a]
            for oa,ob,och in active:
                if och!=ch:
                    conflicts['different_channel_overlapping_session_pairs']+=1
                    if len(conflict_examples)<10:conflict_examples.append({'user':u,'channel_a':och,'channel_b':ch,'overlap_seconds':min(ob,b)-a})
                else:conflicts['same_channel_nonidentical_overlapping_session_pairs']+=1
            active.append((a,b,ch))
    report={'status':'complete','protocol':protocol,'selected_user_train_counts':{str(cat['target_users'][j]):float(activity[j]) for j in range(len(activity)) if str(cat['target_users'][j]) in allowed},'raw_selected_rows':rows,'duration_mismatches':duration_bad,'schedule_check':{'requested':100,'matched':len(matched),'missing_ids':sorted(set(schedule_sample)-matched),'mismatches':schedule_errors},'comparison':comparison,'sessions':{**duplicates,**conflicts,'examples':conflict_examples},'interpretation':'MAX cached fragments can underestimate a raw session crossing originalMay18 split in later historical windows; disconnected sessions are never unioned. Multi-channel overlaps may be household/device concurrency, not automatically corrupt data. Sample audit cannot establish global completeness or geographic timezone.','wall_seconds':time.perf_counter()-began,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'fits':0,'submissions':0}
    out.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['protocol','sessions','selected_user_train_counts']}),flush=True)
if __name__=='__main__':main()
