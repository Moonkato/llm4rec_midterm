"""Deterministic bounded preprocessing; standard library only; no model fitting."""
import bisect
import csv
import json
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT/'midterm/data'
CACHE = ROOT/'solutions/midterm/cache'
EVIDENCE = ROOT/'solutions/midterm/evidence'
EPOCH = datetime(2020,3,9)
SHIFT = '2020-03-29 04:00:00'

def tick(s):
    if s.startswith('2020-03-29 03:'):
        raise ValueError('Observed timestamp inside proposed missing civil hour; inspect clock hypothesis.')
    return int((datetime.fromisoformat(s)-EPOCH).total_seconds())-(3600 if s>=SHIFT else 0)

def iso_schedule(s):
    return s[6:10]+'-'+s[3:5]+'-'+s[:2]+' '+s[11:]

def main():
    began = time.perf_counter(); CACHE.mkdir(parents=True,exist_ok=True); EVIDENCE.mkdir(parents=True,exist_ok=True)
    target = CACHE/'interactions.sqlite'
    if target.exists():
        raise SystemExit('Cache exists; preserve it and choose an explicit rebuild path.')
    db = sqlite3.connect(target)
    db.executescript('''
      PRAGMA journal_mode=OFF;
      PRAGMA synchronous=OFF;
      PRAGMA temp_store=FILE;
      PRAGMA cache_size=-64000;
      CREATE TABLE events (event_id INTEGER PRIMARY KEY, channel TEXT, item TEXT, start INTEGER, end INTEGER, duration INTEGER, civil_start TEXT);
      CREATE TABLE overlaps (user TEXT, event_id INTEGER, split INTEGER, start INTEGER, end INTEGER);
      CREATE TABLE watches (user TEXT, event_id INTEGER, split INTEGER, seconds INTEGER, last_seen INTEGER, fragments INTEGER);
      CREATE TABLE metadata (scope TEXT, item TEXT, attributes TEXT, PRIMARY KEY(scope,item));
    ''')
    cut = tick('2020-05-18 00:00:00'); end = tick('2020-07-27 00:00:00')
    origin = tick('2020-03-09 00:00:00')
    period_end = tick('2020-10-19 00:00:00')
    stats = Counter(); events = []; labelled = defaultdict(list); coverage = defaultdict(list); seen = set(); metadata = {}
    for p in sorted(DATA.glob('export*.csv')):
        future_file = '31-42' in p.name
        with p.open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                stats['schedule_rows']+=1
                civil = iso_schedule(r['start_time']); a = tick(civil); duration = int(r['duration']); b = a+duration
                if duration<=0: stats['nonpositive_schedule_duration_dropped']+=1; continue
                c = r['channel_id']; item = r['tv_show_id']
                spans = coverage[c]
                if spans and a<=spans[-1][1]: spans[-1]=(min(spans[-1][0],a),max(spans[-1][1],b))
                else: spans.append((a,b))
                if item=='0': stats['zero_id_schedule_rows']+=1; continue
                key = (c,a,b,item)
                if key in seen: stats['duplicate_labelled_events_dropped']+=1; continue
                seen.add(key)
                eid=len(events); events.append((eid,c,item,a,b,duration,civil))
                if a<end and not future_file: labelled[c].append((a,b,eid))
                if not future_file:
                    for scope,boundary in [('train',cut),('history',end)]:
                        if a<boundary and (scope,item) not in metadata:
                            attrs={k:r[k] for k in ['tv_show_title','tv_show_category','tv_show_genre_1','tv_show_genre_2','tv_show_genre_3','year_of_production','director','actors']}
                            metadata[(scope,item)]=json.dumps(attrs,ensure_ascii=False)
        print(json.dumps({'phase':'schedule_loaded','file':p.name,'labelled_events':len(events),'elapsed_seconds':round(time.perf_counter()-began,1)}),flush=True)
    db.executemany('INSERT INTO events VALUES (?,?,?,?,?,?,?)',events)
    db.executemany('INSERT INTO metadata VALUES (?,?,?)',[(scope,item,attrs) for (scope,item),attrs in metadata.items()]); db.commit()
    # Conservative removal of historical events with overlapping, different labelled items.
    bad=set(); conflict_examples=[]
    for c, seq in labelled.items():
        seq.sort(); active=[]
        for a,b,eid in seq:
            active=[e for e in active if e[1]>a]
            for other in active:
                if events[other[2]][2]!=events[eid][2]:
                    bad.update((eid,other[2])); stats['conflicting_labelled_event_pairs']+=1
                    if len(conflict_examples)<10: conflict_examples.append([other[2],eid])
            active.append((a,b,eid))
    lookup={}
    for c,seq in labelled.items():
        seq=[e for e in seq if e[2] not in bad]; prefix=[]; mx=-10**12
        for a,b,eid in seq: mx=max(mx,b); prefix.append(mx)
        lookup[c]=([e[0] for e in seq],prefix,seq)
    stats['conflicting_events_dropped_from_interaction_join']=len(bad)
    for c,seq in coverage.items():
        merged=[]
        for a,b in sorted(seq):
            if merged and a<=merged[-1][1]: merged[-1]=(merged[-1][0],max(merged[-1][1],b))
            else: merged.append((a,b))
        coverage[c]=([e[0] for e in merged],merged)
    batch=[]
    with (DATA/'dataset11-30.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            stats['view_rows']+=1; a=tick(r['start_time']); b=tick(r['stop_time']); c=r['vsetv_id']; uid=r['user_id']
            if b-a!=int(r['duraton']):
                stats['normalized_duration_mismatches']+=1
                raise ValueError('Remaining duration discrepancy; halt before producing finalized tables.')
            a=max(a,origin); b=min(b,end)
            if a>=b: stats['views_outside_history_dropped']+=1; continue
            stats['view_seconds_before_deduplicating_user_sessions']+=b-a
            covered=0
            if c in coverage:
                starts,spans=coverage[c]; idx=max(0,bisect.bisect_right(starts,a)-1)
                while idx<len(spans) and spans[idx][0]<b:
                    covered+=max(0,min(b,spans[idx][1])-max(a,spans[idx][0])); idx+=1
            stats['view_seconds_with_any_schedule_before_session_dedup']+=covered
            hit=False
            if c in lookup:
                starts,prefix,seq=lookup[c]
                lo=bisect.bisect_right(prefix,a); hi=bisect.bisect_left(starts,b)
                for ea,eb,eid in seq[lo:hi]:
                    left=max(a,ea); right=min(b,eb)
                    if left>=right: continue
                    hit=True
                    if left<cut: batch.append((uid,eid,0,left,min(right,cut)))
                    if right>cut: batch.append((uid,eid,1,max(left,cut),right))
            stats['view_rows_with_labelled_overlap']+=hit
            if len(batch)>=20000:
                db.executemany('INSERT INTO overlaps VALUES (?,?,?,?,?)',batch); stats['raw_overlap_fragments']+=len(batch); batch.clear()
            if stats['view_rows']%1000000==0:
                db.commit(); print(json.dumps({'phase':'joining','rows':stats['view_rows'],'elapsed_seconds':round(time.perf_counter()-began,1)}),flush=True)
    if batch: db.executemany('INSERT INTO overlaps VALUES (?,?,?,?,?)',batch); stats['raw_overlap_fragments']+=len(batch)
    db.commit(); print(json.dumps({'phase':'merge_user_airing_fragments','raw_fragments':stats['raw_overlap_fragments']}),flush=True)
    # Sort on disk and merge overlaps; duplicate sessions never double-count watched seconds.
    read = db.execute('SELECT user,event_id,split,start,end FROM overlaps ORDER BY user,event_id,split,start,end')
    key=None; low=high=total=fragments=0; batch=[]; seconds_hist=Counter()
    def emit():
        seconds=total+high-low
        batch.append((*key,seconds,high,fragments)); seconds_hist[seconds]+=1
    for uid,eid,split,a,b in read:
        new=(uid,eid,split)
        if new!=key:
            if key is not None: emit()
            key=new; low=a; high=b; total=0; fragments=1
        else:
            fragments+=1
            if a<=high: high=max(high,b)
            else: total+=high-low; low=a; high=b
        if len(batch)>=20000:
            db.executemany('INSERT INTO watches VALUES (?,?,?,?,?,?)',batch); stats['user_airing_split_records']+=len(batch); batch.clear()
    if key is not None: emit()
    if batch: db.executemany('INSERT INTO watches VALUES (?,?,?,?,?,?)',batch); stats['user_airing_split_records']+=len(batch)
    db.commit()
    template={r['user_id'] for r in csv.DictReader((DATA/'submission.csv').open())}
    thresholds=[60,180,300]; cohorts={t:[defaultdict(set),defaultdict(set)] for t in thresholds}
    splits=Counter()
    for uid,item,split,seconds in db.execute('SELECT w.user,e.item,w.split,w.seconds FROM watches w JOIN events e USING(event_id)'):
        splits[split]+=1
        if uid in template:
            for t in thresholds:
                if seconds>=t: cohorts[t][split][uid].add(item)
    coverage_summary={}
    for t,(train,val) in cohorts.items():
        ntruth=sum(len(v) for v in val.values()); repeated=sum(len(v & train.get(u,set())) for u,v in val.items())
        coverage_summary[str(t)]={'template_train_users_with_labels':len(train),'template_validation_users_with_labels':len(val),
            'template_train_user_item_pairs':sum(map(len,train.values())),'template_validation_user_item_pairs':ntruth,
            'validation_user_item_pairs_seen_in_train':repeated,'repeat_fraction':repeated/ntruth if ntruth else None}
    db.execute('CREATE INDEX watches_split ON watches(split)'); db.commit(); db.close()
    result={'status':'complete','clock_rule':{'civil_gap':'2020-03-29 03:00:00 through 03:59:59',
        'offset_after_04':-3600,'schedule_end':'normalized start + duration','geographic_timezone':'unverified'},
        'train_end_civil':'2020-05-18 00:00:00','history_end_civil':'2020-07-27 00:00:00',
        'thresholds_are_coverage_sensitivity_only':True,'counts':dict(stats),'split_records':dict(splits),
        'threshold_coverage':coverage_summary,'conflicting_event_examples':conflict_examples,
        'watch_seconds_histogram':dict(sorted(seconds_hist.items())),
        'wall_seconds':round(time.perf_counter()-began,2),'sqlite_bytes':target.stat().st_size}
    (EVIDENCE/'interaction_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='watch_seconds_histogram'}),flush=True)

if __name__=='__main__': main()
