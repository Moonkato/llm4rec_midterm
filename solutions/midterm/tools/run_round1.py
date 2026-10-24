"""Reproducible, bounded first comparison; never accesses Kaggle or submits."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
import csv
import hashlib
import json
import math
import platform
import resource
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'solutions/midterm'
sys.path.insert(0,str(BASE))

def apk(actual, predicted, k=5):
    actual=set(actual)
    if not actual: return 0.0
    hits=0; total=0.0; seen=set()
    for rank,item in enumerate(predicted[:k],1):
        if item in actual and item not in seen:
            hits+=1; total+=hits/rank
        seen.add(item)
    return total/min(k,len(actual))

def main():
    import numpy as np
    import scipy
    from scipy import sparse
    from models.content import fit_content, score_content
    from models.implicit_als import fit_als, score_als
    from build_interactions import tick

    started=time.perf_counter(); out=BASE/'evidence/round1'; cache=BASE/'cache/round1'
    if (out/'results.json').exists(): raise SystemExit('Completed round exists; use explicit new run for new experiments.')
    out.mkdir(parents=True,exist_ok=True); cache.mkdir(parents=True,exist_ok=True)
    progress_file=out/'progress.jsonl'
    def log(event):
        event['wall_seconds']=round(time.perf_counter()-started,3)
        s=json.dumps(event,ensure_ascii=False); print(s,flush=True)
        with progress_file.open('a') as f: f.write(s+'\n')
    protocol={
      'threshold_seconds':180,'threshold_basis':'Fixed before model scores; require three minutes per airing after merging fragments, reducing brief switches; raw session median is 396s.',
      'train_start':'2020-03-09 00:00:00','train_end_exclusive':'2020-05-18 00:00:00',
      'validation_end_exclusive':'2020-07-27 00:00:00','k':5,
      'primary_truth':'all distinct programmes with >=180s per user/airing in validation',
      'primary_cohort':'all 1260 submission users, including empty validation truth with AP=0',
      'candidate_pool':'all nonzero programmes with an airing intersecting validation window, from supplied schedules',
      'repeat_policy':'allowed for every method','fallback':'same historical unique-viewer popularity restricted to candidates',
      'ties':'ascending numeric programme ID','metadata':'train scope only, no future attributes',
      'secondary_labels':'top5 by holdout watched seconds and top5 by holdout airing count; interpretation diagnostics, not model selection targets',
      'seed':20261022,'als_grid':[{'factors':32,'alpha':1.0},{'factors':32,'alpha':10.0},{'factors':64,'alpha':10.0},{'factors':64,'alpha':40.0}],
      'als_regularization':1.0,'als_iterations':10,'hardware_threads_used':1,
      'official_kaggle_target_rule':'not verified; local scores are not claimed to equal Kaggle scores'
    }
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    db=sqlite3.connect('file:'+str(BASE/'cache/interactions.sqlite')+'?mode=ro',uri=True)
    assert json.loads((BASE/'evidence/interaction_audit.json').read_text())['status']=='complete'
    cut=tick(protocol['train_end_exclusive']); end=tick(protocol['validation_end_exclusive'])
    template=[r['user_id'] for r in csv.DictReader((ROOT/'midterm/data/submission.csv').open())]
    users=sorted(set(template)|{r[0] for r in db.execute('SELECT DISTINCT user FROM watches WHERE split=0')},key=int)
    candidates=sorted({r[0] for r in db.execute('SELECT DISTINCT item FROM events WHERE start<? AND end>?',(end,cut))},key=int)
    historical={r[0] for r in db.execute('SELECT item FROM metadata WHERE scope="train"')}
    items=sorted(historical|set(candidates),key=int)
    uidx={u:i for i,u in enumerate(users)}; iidx={x:i for i,x in enumerate(items)}
    eval_idx=np.array([uidx[u] for u in template]); cand_idx=np.array([iidx[i] for i in candidates])
    db.create_function('decay7',1,lambda t:math.exp(-math.log(2)*max(0,cut-t)/(7*86400)))
    db.create_function('decay28',1,lambda t:math.exp(-math.log(2)*max(0,cut-t)/(28*86400)))
    rr=[]; cc=[]; cnt=[]; seconds=[]; recent7=[]; recent28=[]
    query='''SELECT w.user,e.item,COUNT(*),SUM(w.seconds),SUM(decay7(w.last_seen)),SUM(decay28(w.last_seen))
       FROM watches w JOIN events e USING(event_id) WHERE w.split=0 AND w.seconds>=180 GROUP BY w.user,e.item'''
    for u,i,n,s,d7,d28 in db.execute(query):
        rr.append(uidx[u]); cc.append(iidx[i]); cnt.append(n); seconds.append(s); recent7.append(d7); recent28.append(d28)
    shape=(len(users),len(items)); coordinates=(np.array(rr),np.array(cc))
    matrices=[sparse.csr_matrix((np.array(v,dtype=np.float32),coordinates),shape=shape) for v in [cnt,seconds,recent7,recent28]]
    counts,durations,decay7,decay28=matrices
    del rr,cc,cnt,seconds,recent7,recent28,coordinates
    sparse.save_npz(cache/'train_counts.npz',counts)
    metadata={i:json.loads(a) for i,a in db.execute('SELECT item,attributes FROM metadata WHERE scope="train"')}
    truth={u:set() for u in template}; val_time={u:[] for u in template}; val_count={u:[] for u in template}
    for u,i,n,s in db.execute('''SELECT w.user,e.item,COUNT(*),SUM(w.seconds) FROM watches w JOIN events e USING(event_id)
       WHERE w.split=1 AND w.seconds>=180 GROUP BY w.user,e.item'''):
        if u in truth:
            truth[u].add(i); val_time[u].append((s,i)); val_count[u].append((n,i))
    truth_time={u:{i for _,i in sorted(v,key=lambda x:(-x[0],int(x[1])))[:5]} for u,v in val_time.items()}
    truth_count={u:{i for _,i in sorted(v,key=lambda x:(-x[0],int(x[1])))[:5]} for u,v in val_count.items()}
    activity=np.asarray(counts[eval_idx].sum(axis=1)).ravel(); cold=activity==0
    warm_indices=np.flatnonzero(~cold)
    ordered=sorted(warm_indices,key=lambda j:(activity[j],int(template[j])))
    examples=[template[ordered[len(ordered)//4]],template[ordered[3*len(ordered)//4]]]
    pop=np.asarray((counts>0).sum(axis=0)).ravel()[cand_idx].astype(np.float32)
    fallback=np.argsort(-pop,kind='stable')
    candidate_set=set(candidates)
    assert len(candidate_set)>=5 and all('0' not in truth[u] for u in template)
    label_missing=sum(len(v-candidate_set) for v in truth.values())
    assert label_missing==0
    (out/'catalog.json').write_text(json.dumps({'users':template,'items':items,'candidates':candidates,'example_users':examples,
       'train_users':len(users),'train_nonzero_pairs':counts.nnz,'cold_template_users':int(cold.sum())},indent=2))
    log({'phase':'loaded','shape':list(shape),'train_pairs':counts.nnz,'candidates':len(candidates),'empty_validation_users':sum(not v for v in truth.values())})
    results=[]; predictions={}; configurations={}
    def evaluate(name,scores,config,fit_seconds=0):
        began=time.perf_counter(); scores=np.asarray(scores,dtype=np.float32)
        assert scores.shape==(len(template),len(candidates)) and not np.isnan(scores).any()
        predictions[name]={}; peruser=[]; secondary_time=[]; secondary_count=[]; fallback_users=0
        for j,u in enumerate(template):
            ranked=[]
            if not cold[j]:
                order=np.argsort(-scores[j],kind='stable')
                ranked=[int(x) for x in order[:5] if np.isfinite(scores[j,x])]
            if len(ranked)<5:
                fallback_users+=1
                for x in fallback:
                    if int(x) not in ranked: ranked.append(int(x))
                    if len(ranked)==5: break
            recs=[candidates[x] for x in ranked]
            assert len(recs)==5 and len(set(recs))==5 and set(recs)<=candidate_set
            predictions[name][u]=recs; peruser.append(apk(truth[u],recs))
            secondary_time.append(apk(truth_time[u],recs)); secondary_count.append(apk(truth_count[u],recs))
        np.save(cache/(name+'_scores.npy'),scores)
        np.save(cache/(name+'_ap5.npy'),np.array(peruser))
        nonempty=[v for u,v in zip(template,peruser) if truth[u]]
        result={'method':name,'map5':float(np.mean(peruser)),'map5_nonempty_truth':float(np.mean(nonempty)),
          'diagnostic_map5_top5_seconds_truth':float(np.mean(secondary_time)),
          'diagnostic_map5_top5_airing_count_truth':float(np.mean(secondary_count)),
          'cold_map5':float(np.mean(np.array(peruser)[cold])),'warm_map5':float(np.mean(np.array(peruser)[~cold])),
          'users_using_fallback':fallback_users,'fit_seconds':fit_seconds,'rank_evaluation_seconds':time.perf_counter()-began}
        results.append(result); configurations[name]=config
        (out/'results.partial.json').write_text(json.dumps(results,indent=2))
        log(result)
    evaluate('popularity_unique_viewers',np.tile(pop,(len(template),1)),{'method':'historical distinct viewers'})
    event_pop=np.asarray(counts.sum(axis=0)).ravel()[cand_idx]
    evaluate('popularity_event_counts',np.tile(event_pop,(len(template),1)),{'method':'historical qualifying airing counts'})
    for name,matrix in [('personal_frequency',counts),('personal_watch_seconds',durations),('personal_decay7',decay7),('personal_decay28',decay28)]:
        scores=matrix[eval_idx][:,cand_idx].toarray(); scores[scores<=0]=-np.inf
        evaluate(name,scores,{'method':name,'zero_signal':'common popularity fallback','half_life_days':7 if name.endswith('7') else 28 if name.endswith('28') else None})
    began=time.perf_counter(); content=fit_content(items,metadata,counts); elapsed=time.perf_counter()-began
    scores=score_content(content,eval_idx,cand_idx); scores[scores<=0]=-np.inf
    sparse.save_npz(cache/'content_item_features.npz',content['item_features'])
    sparse.save_npz(cache/'content_user_profiles.npz',content['user_profiles'])
    evaluate('content_tfidf',scores,content['config'],elapsed)
    observed=np.asarray(counts.sum(axis=0)).ravel()[cand_idx]>0
    for parameters in protocol['als_grid']:
        name=f"als_f{parameters['factors']}_a{int(parameters['alpha'])}"
        log({'phase':'training','method':name,'parameters':parameters})
        began=time.perf_counter()
        model=fit_als(counts,**parameters,iterations=protocol['als_iterations'],regularization=protocol['als_regularization'],seed=protocol['seed'],
            progress=lambda e:log({'phase':'als_iteration','method':name,**e}))
        elapsed=time.perf_counter()-began
        scores=score_als(model,eval_idx,cand_idx); scores[:,~observed]=-np.inf
        np.savez(cache/(name+'_factors.npz'),users=model['user_factors'],items=model['item_factors'])
        evaluate(name,scores,model['config'],elapsed)
    db.close()
    (out/'predictions.json').write_text(json.dumps(predictions,indent=2))
    (out/'configs.json').write_text(json.dumps(configurations,indent=2))
    (out/'example_truth.json').write_text(json.dumps({u:{'truth':sorted(truth[u],key=int),'top5_seconds':sorted(truth_time[u],key=int),'training_airings':float(activity[template.index(u)])} for u in examples},indent=2))
    manifest={'status':'complete','numpy':np.__version__,'scipy':scipy.__version__,'python':platform.python_version(),
      'platform':platform.platform(),'wall_seconds':time.perf_counter()-started,
      'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
      'code_hashes':{str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),BASE/'models/content.py',BASE/'models/implicit_als.py']},
      'protocol_sha256':hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest(),'submissions':0}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'results.json').write_text(json.dumps(results,indent=2))
    log({'phase':'complete','best_local':max(results,key=lambda r:r['map5']),'peak_rss_mb':manifest['peak_rss_bytes']/1e6})

if __name__=='__main__': main()
