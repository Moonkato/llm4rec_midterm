"""No-refit controlled fallback audit of frozen main-holdout predictions.

Never mutates a model, historical result, or submitted CSV. Core models retain
historical 180-second training and advanced models retain fractional history;
this harmonizes fallback/cold routing, not all training or candidate retrieval.
"""
import json, sqlite3
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parents[1]
def read(path):return json.loads(path.read_text())
def ap(truth,pred):
 truth=set(map(str,truth));seen=set();hits=0;score=0
 for rank,item in enumerate(map(str,pred[:5]),1):
  if item in truth and item not in seen:hits+=1;score+=hits/rank
  seen.add(item)
 return score/min(5,len(truth)) if truth else 0.
def main():
 cat=read(B/'evidence/round1/catalog.json'); users=cat['users'];items=cat['candidates']
 all_original=read(B/'evidence/round1/predictions.json')
 core=['popularity_unique_viewers','content_tfidf','als_f32_a1','personal_frequency']
 extra={'weekly_ranker_v3':'deep_temporal/top5_count80_24features_eval_predictions.json','airing_model':'airing_model/eval_expected_count_predictions.json','hybrid_v4':'airing_model/eval_equal_rrf60_v3_predictions.json','identity_replay_v5':'final_factorial/combined_eval_predictions.json'}
 originals={m:all_original[m] for m in core}; originals.update({m:read(B/'evidence'/p) for m,p in extra.items()})
 truth=read(B/'evidence/top5_target_audit/eval_truth.json')['top5_count80']
 db=sqlite3.connect('file:'+str(B/'cache/interactions.sqlite')+'?mode=ro',uri=True)
 warm180={str(r[0]) for r in db.execute('SELECT DISTINCT user FROM watches WHERE split=0 AND seconds>=180')};db.close()
 cold180=set(users)-warm180
 with np.load(B/'cache/context_features/eval.npz',allow_pickle=False) as z:
  assert list(map(str,z['users']))==users
  cold30={str(u) for u,c in zip(z['users'],z['cold']) if c}
 union=cold180|cold30
 # Popularity is constant for all users, including old cold users.
 fallback=originals['popularity_unique_viewers'][users[0]]
 assert all(p==fallback for p in originals['popularity_unique_viewers'].values())
 corrected={}; methods={}
 for m in core+list(extra):
  recs={}; mixed_fill=0
  scores=np.load(B/f'cache/round1/{m}_scores.npy',mmap_mode='r') if m in core else None
  for j,u in enumerate(users):
   if u in union:rec=list(fallback)
   elif scores is not None:
    order=np.argsort(-scores[j],kind='stable')
    rec=[items[int(k)] for k in order[:5] if np.isfinite(scores[j,k])]
    if len(rec)<5:
     mixed_fill+=1
     # Use identical complete historical popularity ordering for slot-fill,
     # not only its first5; first5 always suffice for a 5-item output.
     for item in fallback:
      if item not in rec:rec.append(item)
      if len(rec)==5:break
   else:rec=list(originals[m][u])
   assert len(rec)==len(set(rec))==5 and set(rec)<=set(items)
   recs[u]=rec
  corrected[m]=recs
  methods[m]={'original_map5':sum(ap(truth[u],originals[m][u]) for u in users)/len(users),'common_fallback_map5':sum(ap(truth[u],recs[u]) for u in users)/len(users),'changed_ranked_lists':sum(recs[u]!=originals[m][u] for u in users),'warm_zero_or_short_signal_fallback_users':mixed_fill}
 examples={u:{'in_common_cold_cohort':u in union,'all_lists_unchanged':all(corrected[m][u]==originals[m][u] for m in corrected),'methods':{m:{'top5':corrected[m][u],'ap5':ap(truth[u],corrected[m][u])} for m in corrected}} for u in cat['example_users']}
 result={'status':'complete','purpose':'Supplementary harmonized-fallback evaluation only; not a new trained model or replacement for the scored submission','window':'2020-05-18 through 2020-07-27 exclusive','users':len(users),'truth':'same full-universe strict top5_count80; ties ascending numeric ID; empty truth AP0','cold_definition':'Union of users without any >=180-second historical qualifying airing and users without any >30%-coverage historical qualifying airing, computed before main holdout','cold180_count':len(cold180),'cold30_count':len(cold30),'common_cold_count':len(union),'common_cold_users':sorted(union,key=int),'fallback':fallback,'fallback_definition':'Historical >=180-second unique-viewer popularity, restricted to the common labelled EPG eligibility universe; numeric-ID ties. Same routing cohort and exact ranked list across all8. Warm zero-signal/short core lists use this same fallback to fill.','candidate_policy':'Same eligible EPG universe and repeats allowed; advanced systems retain original personalized retrieval shortlists. This remains an end-to-end comparison, not a model-family controlled ablation.','methods':methods,'examples':examples,'fits':0,'submissions':0,'frozen_csvs_modified':False,'original_required_method_comparison':'Original round1 four-method comparison already shares exactly the same cold cohort, fallback, histories and eligibility; it satisfies the core two-method comparison. This supplement harmonizes fallback across the later advanced systems as well.','source_script':'tools/build_common_fallback_comparison.py'}
 (B/'evidence/common_fallback_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 (B/'evidence/common_fallback_predictions.json').write_text(json.dumps(corrected,ensure_ascii=False)+'\n')
 print(json.dumps({k:result[k] for k in ['cold180_count','cold30_count','common_cold_count','fallback','methods','examples']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
