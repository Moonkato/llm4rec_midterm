"""One bounded experiment adding actual raw-session channel/hour context."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import csv
import hashlib
import json
import pickle
import resource
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from build_interactions import tick
from check_history_normalization import evaluate

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parents[1]
WINDOWS=[('fit','2020-04-13 00:00:00','2020-05-18 00:00:00'),
 ('eval','2020-05-18 00:00:00','2020-07-27 00:00:00'),
 ('later','2020-06-22 00:00:00','2020-07-27 00:00:00')]


def main():
    began=time.perf_counter();out=BASE/'evidence/raw_context';cache=BASE/'cache/raw_context'
    if (out/'results.json').exists():raise SystemExit('Preserve completed experiment')
    out.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    params=json.loads((BASE/'evidence/training_filter/protocol.json').read_text())['parameters']
    protocol={'windows':WINDOWS,'rationale':'Use actual historical channel/time context including unlabelled airtime omitted from item-interaction features.',
      'source':'midterm/data/dataset11-30.csv','session_weight':'min(max(0,min(stop,cut)-max(start,March9)),3600) seconds per logged session',
      'hour':'civil start_time hour; session weight assigned to its start hour, not spread across hours',
      'all_channels':'unknown or unavailable channels retain probability mass in denominator',
      'duplicates':'raw logged sessions are weighted individually; overlapping/duplicate sessions are not unioned in this ablation',
      'features_added':['raw_channel_affinity','raw_hour_affinity','raw_channel_hour_product'],
      'affinity':'user capped-duration shares dot candidate forecast airing channel/hour shares',
      'future_inputs':'only programme IDs, channel, and airing timestamps; no future views/attributes',
      'fit_filter':'early fit candidate groups containing at least one positive and one negative; equal row weight',
      'parameters':params,'candidate_pool':'existing context_features unchanged','fallback':'existing common cold popularity fallback unchanged',
      'comparison':'filtered_equal_rows 19feature reference, main .5892727072 later .4918811728',
      'selection':'single prespecified variant, exploratory previously used overlapping windows',
      'fits':1,'submissions':0,'timeout_seconds_before_fit':300}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    origin=tick('2020-03-09 00:00:00');cuts=[tick(w[1]) for w in WINDOWS]
    channels=[defaultdict(Counter) for _ in WINDOWS];hours=[defaultdict(Counter) for _ in WINDOWS];totals=[Counter() for _ in WINDOWS]
    raw=ROOT/'midterm/data/dataset11-30.csv';rows=0;weighted=[0]*3
    with raw.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            rows+=1;a=max(tick(r['start_time']),origin)
            if a>=cuts[-1]:continue
            b=tick(r['stop_time']);u=r['user_id'];ch=r['vsetv_id'];hour=int(r['start_time'][11:13])
            for j,cut in enumerate(cuts):
                weight=min(max(0,min(b,cut)-a),3600)
                if weight:
                    channels[j][u][ch]+=weight;hours[j][u][hour]+=weight;totals[j][u]+=weight;weighted[j]+=1
            if rows%1000000==0:
                print(json.dumps({'phase':'raw_stream','rows':rows,'seconds':time.perf_counter()-began}),flush=True)
            if rows%100000==0 and time.perf_counter()-began>300:raise SystemExit('Preprocessing exceeded5min; no fitting performed')
    db=sqlite3.connect('file:'+str(BASE/'cache/interactions.sqlite')+'?mode=ro',uri=True)
    datasets={};summaries=[]
    for j,(name,cut_s,end_s) in enumerate(WINDOWS):
        with np.load(BASE/f'cache/context_features/{name}.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
        item_ch=defaultdict(Counter);item_hr=defaultdict(Counter);exposure=Counter()
        for item,ch,civil in db.execute('SELECT item,channel,civil_start FROM events WHERE start<? AND end>?',(tick(end_s),tick(cut_s))):
            item_ch[item][ch]+=1;item_hr[item][int(civil[11:13])]+=1;exposure[item]+=1
        extra=np.zeros((len(d['y']),3),dtype=np.float32)
        for n,uid in enumerate(d['users']):
            u=str(uid);total=totals[j][u]
            if not total:continue
            a,b=d['group_offsets'][n:n+2]
            for row in range(a,b):
                item=str(d['item_ids'][d['row_item_indices'][row]]);den=total*exposure[item]
                ca=sum(channels[j][u][ch]*count for ch,count in item_ch[item].items())/den
                ha=sum(hours[j][u][h]*count for h,count in item_hr[item].items())/den
                extra[row]=ca,ha,ca*ha
        assert np.isfinite(extra).all() and np.all(extra>=0) and np.all(extra<=1.000001)
        np.save(cache/f'{name}_extra.npy',extra)
        d['X']=np.concatenate([d['X'],extra],axis=1);datasets[name]=d
        summaries.append({'window':name,'users_with_raw_context':sum(bool(totals[j][str(u)]) for u in d['users']),
                          'raw_weighted_sessions':weighted[j],'rows':len(extra)})
    db.close()
    if time.perf_counter()-began>300:raise SystemExit('Preprocessing exceeded5min; no fitting performed')
    fit=datasets['fit'];keep=np.zeros(len(fit['y']),dtype=bool);groups=0
    for a,b in zip(fit['group_offsets'][:-1],fit['group_offsets'][1:]):
        positive=int(fit['y'][a:b].sum())
        if 0<positive<b-a:keep[a:b]=True;groups+=1
    model=HistGradientBoostingClassifier(**params);started=time.perf_counter()
    model.fit(fit['X'][keep],fit['y'][keep]);fit_seconds=time.perf_counter()-started
    with (cache/'model.pkl').open('wb') as f:pickle.dump(model,f)
    result={'method':'filtered_raw_context22','fit_rows':int(keep.sum()),'fit_users':groups,'fit_seconds':fit_seconds}
    for name in ('eval','later'):
        d=datasets[name];scores=model.predict_proba(d['X'])[:,1];metric,preds=evaluate(d,scores)
        result[name]={'map5':metric};np.save(cache/f'{name}_scores.npy',scores)
        (out/f'{name}_predictions.json').write_text(json.dumps(preds)+'\n')
    manifest={'result':result,'raw_rows':rows,'context_summary':summaries,'wall_seconds':time.perf_counter()-began,
       'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
       'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
          [raw,Path(__file__),*[BASE/f'cache/context_features/{n}.npz' for n,_,_ in WINDOWS]]}}
    (out/'results.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
