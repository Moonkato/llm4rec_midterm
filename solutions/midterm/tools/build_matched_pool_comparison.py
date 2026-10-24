"""No-fit comparison on exactly the same per-user shortlist and fallback.

Saved full-catalogue scores of core models are restricted to the original v3
shortlist. Advanced scores already use that identical shortlist. The frozen
submission files and historical experiment metrics are never overwritten.
"""
import json,sqlite3,hashlib,time
from pathlib import Path
import numpy as np
from build_common_fallback_comparison import ap
B=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def main():
 start=time.perf_counter();cat=read(B/'evidence/round1/catalog.json');users=cat['users'];items=cat['candidates'];idx={str(i):j for j,i in enumerate(items)}
 common=read(B/'evidence/common_fallback_comparison.json');cold=set(common['common_cold_users']);truth=read(B/'evidence/top5_target_audit/eval_truth.json')['top5_count80']
 with np.load(B/'cache/context_features/eval.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
 assert users==list(map(str,d['users']))
 core=['popularity_unique_viewers','content_tfidf','als_f32_a1','personal_frequency']
 scorepaths={m:B/f'cache/round1/{m}_scores.npy' for m in core}
 scorepaths.update({'weekly_ranker_v3':B/'cache/deep_temporal/top5_count80_24features_eval_scores.npy','airing_model':B/'cache/airing_model/eval_expected_counts.npy','hybrid_v4':B/'cache/final_factorial/v4_eval_scores.npy','identity_replay_v5':B/'cache/final_factorial/combined_eval_scores.npy'})
 scorepaths.update({'identity_ranker':B/'cache/final_identity/standalone_eval_scores.npy','replay_airing':B/'cache/final_replay/eval_replay_expected_counts.npy'})
 scores={m:np.load(p,mmap_mode='r') for m,p in scorepaths.items()}
 pop=scores['popularity_unique_viewers'][0]
 old=read(B/'evidence/common_fallback_predictions.json');old.update({'identity_ranker':read(B/'evidence/final_identity/standalone_eval_predictions.json'),'replay_airing':read(B/'evidence/final_replay/eval_replay_expected_count_predictions.json')});recs={m:{} for m in scores};mixed={m:0 for m in scores};pools={};fallbacks={}
 for j,u in enumerate(users):
  a,e=d['group_offsets'][j:j+2];pool=list(map(str,d['item_ids'][d['row_item_indices'][a:e]]));poolset=set(pool);assert len(poolset)==len(pool) and len(pool)>=5
  fullindices=np.asarray([idx[i] for i in pool]);fallback=sorted(pool,key=lambda i:(-pop[idx[i]],int(i)))[:5]
  pools[u]=pool;fallbacks[u]=fallback
  for m,arr in scores.items():
   if u in cold:chosen=list(fallback)
   else:
    vals=arr[j,fullindices] if m in core else arr[a:e]
    order=sorted(range(len(pool)),key=lambda k:(-vals[k],int(pool[k])))
    chosen=[pool[k] for k in order[:5] if np.isfinite(vals[k])]
    if len(chosen)<5:
     mixed[m]+=1
     for i in fallback:
      if i not in chosen:chosen.append(i)
      if len(chosen)==5:break
   assert len(chosen)==len(set(chosen))==5 and set(chosen)<=poolset
   recs[m][u]=chosen
 common_summary=read(B/'evidence/report_comparison.json')
 db=sqlite3.connect('file:'+str(B/'cache/interactions.sqlite')+'?mode=ro',uri=True);metadata={}
 for scope,i,attrs in db.execute('select scope,item,attributes from metadata order by scope desc'):metadata[i]=json.loads(attrs)
 def desc(i):
  attrs=metadata.get(str(i),{});return {'id':str(i),'title':attrs.get('tv_show_title','Unknown'),'category':attrs.get('tv_show_category','Unknown')}
 examples=[]
 for oldex in common_summary['examples']:
  u=oldex['user_id'];history={str(i) for i, in db.execute('select distinct e.item from watches w join events e using(event_id) where w.user=? and w.split=0 and w.seconds>=180',(u,))};ex={k:v for k,v in oldex.items() if k!='methods'};ex['methods']={}
  for m in recs:
   ex['methods'][m]={'ap5_strict_top5':ap(truth[u],recs[m][u]),'changed_from_original':recs[m][u]!=([r['id'] for r in oldex['methods'][m]['recommendations']] if m in oldex['methods'] else old[m][u]),'recommendations':[dict(desc(i),seen_in_training_180s=i in history,hit_strict_top5=i in truth[u]) for i in recs[m][u]]}
  examples.append(ex)
 db.close()
 methods={m:{'map5':sum(ap(truth[u],recs[m][u]) for u in users)/len(users),'changed_vs_full_eligibility_common_fallback':sum(recs[m][u]!=old[m][u] for u in users),'warm_short_or_zero_signal_fallback_users':mixed[m]} for m in recs}
 result={'status':'complete','purpose':'Identical per-user candidate-pool comparison of frozen systems, no refitting','users':1260,'window':'2020-05-18 through 2020-07-27 exclusive','truth':'same strict top5_count80 full-universe target; AP denominator min(5,truth size) before candidate restriction; empty truth AP0','candidate_pool':'Exactly original v3 per-user eval shortlist, union top100 personal schedule count, top30 >30% historical unique-viewer popularity and top30 positive channel-hour affinity. All10 methods restricted to identical item IDs for each user. Pool uses historical behavior and supplied future schedule, never holdout labels.','candidate_counts':{'min':min(map(len,pools.values())),'max':max(map(len,pools.values())),'total':sum(map(len,pools.values()))},'repeat_policy':'Allowed identically for all methods','cold_cohort_count':len(cold),'cold_definition':common['cold_definition'],'cold_user_ids':common['common_cold_users'],'fallback':'Same historical >=180s distinct-viewer popularity restricted to that user shortlist, numeric-ID ties; same exact ranked fallback for every method for a given user','global180_top5_present_in_every_pool':all(set(common['fallback'])<=set(pool) for pool in pools.values()),'distinct_fallback_lists':len({tuple(v) for v in fallbacks.values()}),'methods':methods,'examples':examples,'preserved_original_csvs':True,'fits':0,'submissions':0,'limitation':'Training histories/thresholds and model fitting objectives remain different; this is a matched-output-policy comparison of frozen systems, not a retrained model-family feature ablation. Primary benchmark supplements historical exact submitted-system scores.','source_hashes':{str(p.relative_to(B)):hashlib.sha256(p.read_bytes()).hexdigest() for p in scorepaths.values()},'runtime_seconds':time.perf_counter()-start,'source_script':'tools/build_matched_pool_comparison.py'}
 (B/'evidence/matched_pool_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');(B/'evidence/matched_pool_predictions.json').write_text(json.dumps(recs)+'\n')
 print(json.dumps({k:result[k] for k in ['candidate_counts','global180_top5_present_in_every_pool','distinct_fallback_lists','methods','runtime_seconds']},indent=2))
 print('Examples',[(e['user_id'],{m:(x['ap5_strict_top5'],x['changed_from_original']) for m,x in e['methods'].items()}) for e in examples])
if __name__=='__main__':main()
