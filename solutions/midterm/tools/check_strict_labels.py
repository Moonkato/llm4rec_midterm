"""Sensitivity to participant-described single-session >80% viewing labels."""
import json,sqlite3,time
from collections import Counter,defaultdict
from pathlib import Path
from run_round1 import apk
BASE=Path(__file__).resolve().parents[1]
out=BASE/'evidence/strict_labels';out.mkdir(exist_ok=True)
protocol={'status':'sensitivity analysis, official target not confirmed','source':'https://medium.com/data-science/two-layered-recommender-system-methodology-a-prize-winning-solution-c4bdb815b156','truth':'all distinct items with one logged viewing overlap strictly >80% full airing duration, validation split1','single_session_proxy':'maximum raw overlap fragment per user/airing/split; does not sum interrupted fragments','thresholds_train':[0.3,0.5,0.8],'cohort':'all1260, emptytruth AP0','candidate_pool':'same round1','selection':'exploratory, changed label definition motivated by external source; not directly comparable to original metric'}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
t=time.perf_counter(); cat=json.loads((BASE/'evidence/round1/catalog.json').read_text()); users=cat['users'];candidates=cat['candidates']; allowed=set(candidates)
truth={u:set() for u in users}; counts={x:defaultdict(Counter) for x in protocol['thresholds_train']}
db=sqlite3.connect('file:'+str(BASE/'cache/interactions.sqlite')+'?mode=ro',uri=True)
query='''SELECT o.user,e.item,o.split,e.duration,MAX(o.end-o.start) FROM overlaps o JOIN events e USING(event_id) GROUP BY o.user,o.event_id,o.split'''
for u,i,split,duration,seconds in db.execute(query):
    if split==1:
        if u in truth and seconds>0.8*duration:truth[u].add(i)
    else:
        for fraction in counts:
            if seconds>fraction*duration: counts[fraction][u][i]+=1
results=[];preds={}
def evaluate(name,recs):
    ap=[apk(truth[u],recs[u]) for u in users]
    row={'method':name,'map5_strict':sum(ap)/len(ap),'nonempty_map5_strict':sum(ap)/sum(bool(truth[u]) for u in users)}
    results.append(row);print(json.dumps(row),flush=True)
original=json.loads((BASE/'evidence/round1/predictions.json').read_text())
for name,recs in original.items():evaluate(name,recs)
fused=json.loads((BASE/'evidence/fusion/predictions.json').read_text())
for weight,recs in fused.items():evaluate('previous_rrf_'+weight,recs)
for fraction,train in counts.items():
    pop=Counter()
    for history in train.values():pop.update(history.keys())
    fallback=sorted(candidates,key=lambda i:(-pop[i],int(i)))
    recs={}
    for u in users:
        history=train[u];top=sorted((i for i in history if i in allowed),key=lambda i:(-history[i],int(i)))[:5]
        for i in fallback:
            if len(top)==5:break
            if i not in top:top.append(i)
        recs[u]=top
    name='personal_single_session_gt'+str(fraction);preds[name]=recs;evaluate(name,recs)
    evaluate('pop_single_session_gt'+str(fraction),{u:fallback[:5] for u in users})
(out/'results.json').write_text(json.dumps({'protocol':protocol,'empty_truth_users':sum(not t for t in truth.values()),'truth_pairs':sum(map(len,truth.values())),'results':results,'wall_seconds':time.perf_counter()-t},indent=2))
(out/'predictions.json').write_text(json.dumps(preds,indent=2))
(out/'truth.json').write_text(json.dumps({u:sorted(t,key=int) for u,t in truth.items()},indent=2))
