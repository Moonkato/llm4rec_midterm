import json, sqlite3
from pathlib import Path
b=Path('solutions/midterm/evidence'); r=b/'round1'
def read(p):return json.loads(p.read_text())
cat=read(r/'catalog.json'); preds=read(r/'predictions.json'); old=read(r/'example_truth.json')
core=['popularity_unique_viewers','content_tfidf','als_f32_a1','personal_frequency']
extra={'weekly_ranker_v3':'deep_temporal/top5_count80_24features_eval_predictions.json','airing_model':'airing_model/eval_expected_count_predictions.json','hybrid_v4':'airing_model/eval_equal_rrf60_v3_predictions.json','identity_replay_v5':'final_factorial/combined_eval_predictions.json'}
for k,p in extra.items():preds[k]=read(b/p)
strict=read(b/'top5_target_audit/eval_truth.json')['top5_count80']
def ap(truth, rec):
 t=set(map(str,truth)); seen=set(); n=0;s=0
 for k,i in enumerate(map(str,rec[:5]),1):
  if i in t and i not in seen:n+=1;s+=n/k
  seen.add(i)
 return s/min(5,len(t)) if t else 0
c=sqlite3.connect('file:solutions/midterm/cache/interactions.sqlite?mode=ro',uri=True)
metadata={}
for scope,i,a in c.execute('select scope,item,attributes from metadata order by scope desc'):metadata[i]=json.loads(a)
def desc(i):
 a=metadata.get(str(i),{});return {'id':str(i),'title':a.get('tv_show_title','Unknown'),'category':a.get('tv_show_category','Unknown')}
examples=[]
for idx,u in enumerate(cat['example_users']):
 history={i:int(n) for i,n in c.execute('select e.item,count(*) from watches w join events e using(event_id) where w.user=? and w.split=0 and w.seconds>=180 group by e.item',(u,))}
 ex={'user_id':u,'selection':'25th percentile of non-cold historical airing count' if idx==0 else '75th percentile of non-cold historical airing count','training_airings':old[u]['training_airings'],'historical_unique_items':len(history),'historical_top5':[dict(desc(i),airings=n) for i,n in sorted(history.items(),key=lambda x:(-x[1],int(x[0])))[:5]],'validation_truth_180s_count':len(old[u]['truth']),'validation_truth_strict_top5':[desc(i) for i in strict[u]],'methods':{}}
 for m in core+list(extra):
  rec=preds[m][u];ex['methods'][m]={'ap5_180s':ap(old[u]['truth'],rec),'ap5_strict_top5':ap(strict[u],rec),'recommendations':[dict(desc(i),seen_in_training_180s=i in history,hit_180s=i in old[u]['truth'],hit_strict_top5=i in strict[u]) for i in rec]}
 examples.append(ex)
res={'sources':['round1/protocol.json','round1/results.json','round1/configs.json','round1/catalog.json','round1/predictions.json','round1/example_truth.json',*extra.values(),'top5_target_audit/eval_truth.json'],'comparison_protocol_180s':read(r/'protocol.json'),'comparison_180s':[x for x in read(r/'results.json') if x['method'] in core],'core_configurations':{k:v for k,v in read(r/'configs.json').items() if k in core},'runtime':read(r/'manifest.json'),'strict_top5_rescoring':{'note':'Read-only re-evaluation of saved predictions. All methods use the same May18-Jul27 holdout, 1260 users and strict top5_count80 truth. Baselines were fitted on >=180s historical labels; advanced models use >30% history and >80% target with separate earlier fit targets. The eligible schedule universe and repeat permission are shared, but advanced rankers retrieve a shortlist. Every pipeline uses historical popularity for cold fallback; different historical qualification thresholds can change the cold cohort and popularity ordering. This is an end-to-end system comparison, not a controlled feature ablation. No training was performed for this report.','scores':{m:sum(ap(strict[u],preds[m][u]) for u in cat['users'])/len(cat['users']) for m in core+list(extra)}},'examples':examples,'gaps':['No authentic screenshot of the designated final Kaggle public score was located by this subtask.','Final v5 has no supplied Kaggle result.','Do not conflate 180-second all-item MAP with strict top5 MAP or Kaggle.']}
(b/'report_comparison.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(res['strict_top5_rescoring'],indent=2));print('saved examples',[(e['user_id'],e['training_airings']) for e in examples])
