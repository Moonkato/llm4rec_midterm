"""Exact hour-overlap weekly context from actual sessions; no model training."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import csv,json,time,sqlite3,hashlib,resource,sys,argparse
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
from scipy import sparse
from build_interactions import tick
B=Path(__file__).resolve().parents[1];ROOT=B.parents[1]
WINDOWS=[('fit','2020-04-13 00:00:00','2020-05-18 00:00:00'),('eval','2020-05-18 00:00:00','2020-07-27 00:00:00'),('later','2020-06-22 00:00:00','2020-07-27 00:00:00')]
NAMES=['weekly_expected_minutes_per_day','weekly_mean_airing_fraction','weekly_recent28_expected_minutes_per_day','weekly_active_date_fraction','weekly_recent_all_ratio_clip10']
SHIFT=tick('2020-03-29 04:00:00')//3600

def segments(a,b):
    while a<b:
        h=a//3600;stop=min(b,(h+1)*3600);civil=h+(h>=SHIFT)
        yield civil%168,civil//24,a,stop
        a=stop

def main(deep=False, test=False, retrieval=False):
    assert not (deep and test)
    windows=[('inner_fit','2020-04-13 00:00:00','2020-05-04 00:00:00'),('inner_stop','2020-05-04 00:00:00','2020-05-18 00:00:00')] if deep else WINDOWS
    if deep and not (B/'evidence/deep_context/results.json').exists():raise SystemExit('Wait for completed inner context arrays')
    started=time.perf_counter();out=B/('cache/weekly_candidates' if retrieval else 'cache/deep_weekly' if deep else 'cache/weekly_context');ev=B/('evidence/weekly_candidates' if retrieval else 'evidence/deep_weekly' if deep else 'evidence/weekly_context')
    if test:
        windows=[('test','2020-07-27 00:00:00','2020-10-19 00:00:00')]
        out=B/'cache/test_weekly';ev=B/'evidence/test_weekly'
    if (ev/'results.json').exists():raise SystemExit('Preserve completed context')
    out.mkdir(parents=True,exist_ok=True);ev.mkdir(parents=True,exist_ok=True)
    protocol={'windows':windows,'features':NAMES,'fits':0,'submissions':0,'candidate_pool':'base19context rows; original3 mapped from collaborative rows' if deep else 'unchanged collaborative_features',
      'history':'actual raw sessions before each cutoff, including unlabelled airtime',
      'cap':'Each history-clipped session contributes min(duration,3600) seconds proportionally across its exact hour overlaps; cap is not deduplication.',
      'duplicates':'Logged sessions counted separately; per-slot estimated propensity clipped to1.',
      'clock':'same tick missing2020-03-29 03hour correction; normalized hour inverse maps to civil weekday/hour',
      'rates':'slot seconds / actual historical opportunity seconds; recent28 calendar days same denominator; active-date bits per actual clipped overlap day divided by slot opportunities',
      'schedule':'only future IDs/channel/time/duration; integrate forecast airing slot durations and full-event fractions',
      'unknown_channels':'No candidate has these channels; their actual slots cannot contribute to item exposure. Opportunity denominators are temporal, not share-of-channel mass.',
      'ratio':'recent expected/all expected; denominator0 =>0, cap10','budget':'5min2GB, sequential windows','no_model_training':True}
    if retrieval:
        protocol.update(retrieval='union top30 positive expected minutes/day and top30 positive recent28 minutes/day, numeric item ID ties, across all future eligible items',selection_labels='none; only historical sessions and future schedule')
    (ev/'protocol.json').write_text(json.dumps(protocol,indent=2))
    if test:
        protocol['candidate_pool']='unchanged base19 full-history test candidates'
        protocol['test_labels_used']=False
        (ev/'protocol.json').write_text(json.dumps(protocol,indent=2))
    def guard():
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        if time.perf_counter()-started>300 or rss>2_000_000_000:raise RuntimeError('Preprocessing budget exceeded; no models trained')
    db=sqlite3.connect('file:'+str(B/'cache/interactions.sqlite')+'?mode=ro',uri=True)
    channels=[r[0] for r in db.execute('SELECT DISTINCT channel FROM events')];chidx={c:j for j,c in enumerate(channels)};nslots=len(channels)*168
    results=[];raw=ROOT/'midterm/data/dataset11-30.csv'
    if deep:
        for name,_,_ in WINDOWS:
            with np.load(B/f'cache/context_features/{name}.npz',allow_pickle=False) as z:base={k:z[k] for k in ('users','group_offsets','item_ids','row_item_indices','y')}
            with np.load(B/f'cache/collaborative_features/{name}.npz',allow_pickle=False) as z:expanded={k:z[k] for k in ('users','group_offsets','item_ids','row_item_indices')}
            old=np.load(B/f'cache/weekly_context/{name}_row_features.npy',mmap_mode='r');mapped=np.empty((len(base['y']),5),dtype=np.float32)
            assert np.array_equal(base['users'],expanded['users'])
            for n in range(len(base['users'])):
                a,b=base['group_offsets'][n:n+2];aa,bb=expanded['group_offsets'][n:n+2]
                lookup={str(expanded['item_ids'][expanded['row_item_indices'][j]]):j for j in range(aa,bb)}
                mapped[a:b]=old[[lookup[str(base['item_ids'][i])] for i in base['row_item_indices'][a:b]]]
            np.save(out/f'{name}_row_features.npy',mapped)
            results.append({'window':name,'rows':len(mapped),'features':NAMES,'method':'exact user/item remapping of existing computed features','sha256':hashlib.sha256((out/f'{name}_row_features.npy').read_bytes()).hexdigest()})
            del base,expanded,old,mapped
    for name,cut_s,end_s in windows:
        began=time.perf_counter();cut,end=tick(cut_s),tick(end_s)
        recent_start=tick(str(datetime.fromisoformat(cut_s)-timedelta(days=28)))
        source=B/('cache/deep_context' if deep else 'cache/collaborative_features')/f'{name}.npz'
        if test:source=B/'cache/context_features/test.npz'
        with np.load(source,allow_pickle=False) as z:d={k:z[k] for k in z.files if k!='X'}
        users=d['users'];uidx={str(u):j for j,u in enumerate(users)}
        secs={};recent={};dates={};rows=0;used=0
        with raw.open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                rows+=1
                if rows%200000==0:guard()
                u=uidx.get(r['user_id']);channel=chidx.get(r['vsetv_id'])
                if u is None or channel is None:continue
                a=max(0,tick(r['start_time']))
                if a>=cut:continue
                b=min(cut,tick(r['stop_time']))
                if b<=a:continue
                used+=1;weight=min(1.,3600/(b-a));base=u*nslots+channel*168
                for slot,day,left,right in segments(a,b):
                    key=base+slot;secs[key]=secs.get(key,0.)+(right-left)*weight
                    dates[key]=dates.get(key,0)|(1<<day)
                    if right>recent_start:recent[key]=recent.get(key,0.)+(right-max(left,recent_start))*weight
        denom=np.zeros(168);denrecent=np.zeros(168)
        for slot,day,a,b in segments(0,cut):denom[slot]+=b-a
        for slot,day,a,b in segments(recent_start,cut):denrecent[slot]+=b-a
        keys=np.fromiter(secs,dtype=np.int64,count=len(secs));slot=keys%168
        all_values=np.minimum(1,np.fromiter(secs.values(),dtype=np.float64)/denom[slot])
        recent_values=np.minimum(1,np.array([recent.get(int(k),0.) for k in keys])/denrecent[slot])
        active_values=np.minimum(1,np.array([dates[int(k)].bit_count() for k in keys])/(denom[slot]/3600))
        coords=(keys//nslots,keys%nslots);shape=(len(users),nslots)
        matrices=[sparse.csr_matrix((v.astype(np.float32),coords),shape=shape) for v in (all_values,recent_values,active_values)]
        del secs,recent,dates,keys,all_values,recent_values,active_values
        item_ids=d['item_ids'];ii={str(i):j for j,i in enumerate(item_ids)}
        duration_weights={};fraction_weights={};airings=np.zeros(len(item_ids));totaldur=np.zeros(len(item_ids))
        for i,ch,a,b,duration in db.execute('SELECT item,channel,start,end,duration FROM events WHERE start<? AND end>?',(end,cut)):
            if i not in ii:continue
            j=ii[i];airings[j]+=1;left=max(a,cut);right=min(b,end);totaldur[j]+=right-left
            for slot,day,aa,bb in segments(left,right):
                key=(chidx[ch]*168+slot,j);seconds=bb-aa
                duration_weights[key]=duration_weights.get(key,0)+seconds
                fraction_weights[key]=fraction_weights.get(key,0.)+seconds/duration
        coords=tuple(np.array(v,dtype=np.int32) for v in zip(*duration_weights))
        epgshape=(nslots,len(item_ids))
        durations=sparse.csr_matrix((np.array(list(duration_weights.values()),dtype=np.float32),coords),shape=epgshape)
        fractions=sparse.csr_matrix((np.array([fraction_weights[k] for k in duration_weights],dtype=np.float32),coords),shape=epgshape)
        del duration_weights,fraction_weights
        expected=(matrices[0]@durations).tocsr();rec_expected=(matrices[1]@durations).tocsr()
        mean_fraction=(matrices[0]@fractions).tocsr();activity=(matrices[2]@durations).tocsr()
        if retrieval:
            extra={}
            for j,u in enumerate(users):
                chosen=[]
                for matrix in (expected,rec_expected):
                    row=matrix.getrow(j)
                    ranking=sorted(((int(i),float(v)) for i,v in zip(row.indices,row.data) if v>0),key=lambda iv:(-iv[1],int(item_ids[iv[0]])))[:30]
                    chosen.extend(str(item_ids[i]) for i,_ in ranking)
                extra[str(u)]=list(dict.fromkeys(chosen))
            (ev/f'{name}_extra_candidates.json').write_text(json.dumps(extra))
            for suffix,matrix in [('expected',expected),('recent_expected',rec_expected),('mean_fraction',mean_fraction),('activity',activity)]:
                sparse.save_npz(out/f'{name}_{suffix}.npz',matrix)
            np.savez(out/f'{name}_catalog.npz',users=users,item_ids=item_ids,airings=airings,totaldur=totaldur,forecast_days=(end-cut)/86400)
        features=np.zeros((len(d['y']),5),dtype=np.float32)
        for j in range(len(users)):
            a,b=d['group_offsets'][j:j+2];ix=d['row_item_indices'][a:b]
            value=expected[j,ix].toarray().ravel();rv=rec_expected[j,ix].toarray().ravel()
            features[a:b,0]=value/(60*((end-cut)/86400))
            features[a:b,1]=mean_fraction[j,ix].toarray().ravel()/airings[ix]
            features[a:b,2]=rv/(60*((end-cut)/86400))
            features[a:b,3]=activity[j,ix].toarray().ravel()/totaldur[ix]
            features[a:b,4]=np.minimum(10,np.divide(rv,value,out=np.zeros_like(rv),where=value>0))
        assert np.isfinite(features).all() and np.all(features>=0)
        np.save(out/f'{name}_row_features.npy',features)
        row={'window':name,'rows':len(features),'features':NAMES,'used_sessions':used,'users':len(users),
             'slot_nonzeros':matrices[0].nnz,'nonzero_feature_rows':int(np.any(features>0,axis=1).sum()),
             'wall_seconds':time.perf_counter()-began,'sha256':hashlib.sha256((out/f'{name}_row_features.npy').read_bytes()).hexdigest()}
        results.append(row);print(json.dumps(row),flush=True);guard()
        del matrices,expected,rec_expected,mean_fraction,activity,durations,fractions,features
    db.close();(ev/'results.json').write_text(json.dumps({'results':results,'wall_seconds':time.perf_counter()-started,
      'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group();group.add_argument('--deep',action='store_true');group.add_argument('--test',action='store_true');group.add_argument('--retrieval',action='store_true')
    args=parser.parse_args();main(deep=args.deep,test=args.test,retrieval=args.retrieval)
