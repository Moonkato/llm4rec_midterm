"""Exact expected AP under uniformly chosen boundary ties, no fitting or sampling."""
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from build_interactions import tick
from compare_airing_model import ap5

B=Path(__file__).resolve().parents[1]


def expected_ap(pred, fixed, tied, remaining, denominator):
    if not denominator:
        return 0.0
    size=len(tied)
    p=remaining/size if size else 0.0
    joint=remaining*(remaining-1)/(size*(size-1)) if size>1 else 0.0
    total=0.0
    kinds=[2 if i in fixed else 1 if i in tied else 0 for i in pred]
    for k,kind in enumerate(kinds):
        marginal=1.0 if kind==2 else p if kind==1 else 0.0
        numerator=marginal
        for prior in kinds[:k]:
            if kind==2 and prior==2:
                numerator+=1
            elif kind and prior:
                numerator+=joint if kind==prior==1 else p
        total+=numerator/(k+1)
    return total/denominator


def training_audit(db):
    a,b=map(tick,['2020-04-13 00:00:00','2020-05-18 00:00:00'])
    q='''SELECT o.user,e.item,COUNT(*) FROM
      (SELECT user,event_id FROM overlaps WHERE start<? AND end>?
       GROUP BY user,event_id HAVING MAX(MAX(0,MIN(end,?)-MAX(start,?)))>
       .8*(SELECT duration FROM events WHERE events.event_id=overlaps.event_id)) o
      JOIN events e USING(event_id) GROUP BY o.user,e.item'''
    counts=defaultdict(dict)
    for u,i,n in db.execute(q,(b,a,b,a)):
        counts[u][i]=n
    with np.load(B/'cache/context_features/fit.npz') as z:
        d={k:z[k] for k in ('users','item_ids','row_item_indices','group_offsets')}
    names=['users','boundary_tied_users','candidate_rows','hard_positive_rows',
           'hard_zero_to_soft_positive_rows','hard_one_to_fractional_rows',
           'full_target_mass','target_mass_inside_pool','target_mass_outside_pool',
           'hard_target_items_outside_pool']
    summaries={name:{key:0 for key in names} for name in ['all_fit_users','retained_hard_queries']}
    for j,u in enumerate(d['users']):
        lo,hi=d['group_offsets'][j:j+2]
        pool=set(map(str,d['item_ids'][d['row_item_indices'][lo:hi]]))
        ranked=sorted(counts[u],key=lambda i:(-counts[u][i],int(i)))
        hard=set(ranked[:5]);soft={i:1.0 for i in hard}
        ambiguous=len(ranked)>5 and counts[u][ranked[4]]==counts[u][ranked[5]]
        if ambiguous:
            cutoff=counts[u][ranked[4]]
            fixed={i for i in ranked if counts[u][i]>cutoff}
            tied={i for i in ranked if counts[u][i]==cutoff}
            soft={i:1.0 for i in fixed}
            soft.update({i:(5-len(fixed))/len(tied) for i in tied})
        positives=len(pool&hard)
        for group in ['all_fit_users']+(['retained_hard_queries'] if 0<positives<len(pool) else []):
            x=summaries[group]
            x['users']+=1;x['boundary_tied_users']+=ambiguous
            x['candidate_rows']+=len(pool);x['hard_positive_rows']+=positives
            x['hard_zero_to_soft_positive_rows']+=sum(i not in hard and soft.get(i,0)>0 for i in pool)
            x['hard_one_to_fractional_rows']+=sum(soft[i]<1 for i in pool&hard)
            total=sum(soft.values());inside=sum(soft.get(i,0) for i in pool)
            x['full_target_mass']+=total;x['target_mass_inside_pool']+=inside
            x['target_mass_outside_pool']+=total-inside
            x['hard_target_items_outside_pool']+=len(hard-pool)
    for x in summaries.values():
        x['boundary_tied_user_fraction']=x['boundary_tied_users']/x['users']
        x['soft_mass_missing_fraction']=x['target_mass_outside_pool']/x['full_target_mass']
    return summaries


def main():
    started=time.perf_counter()
    db=sqlite3.connect('file:'+str(B/'cache/interactions.sqlite')+'?mode=ro',uri=True)
    db.execute('PRAGMA temp_store=FILE');db.execute('PRAGMA cache_size=-32000')
    results={}
    for window,start,end in [('eval','2020-05-18','2020-07-27'),('later','2020-06-22','2020-07-27')]:
        paths={
            'v3':B/f'evidence/deep_temporal/top5_count80_24features_{window}_predictions.json',
            'v4':B/f'evidence/airing_model/{window}_equal_rrf60_v3_predictions.json',
            'recurrence_fusion':B/f'evidence/airing_recurrence/{window}_recurrence_equal_rrf60_v3_predictions.json',
            'recurrence_standalone':B/f'evidence/airing_recurrence/{window}_recurrence_expected_count_predictions.json',
            'all80_ranker24':B/f'evidence/deep_temporal/all80_24features_{window}_predictions.json'}
        preds={k:json.loads(p.read_text()) for k,p in paths.items()}
        users=sorted(preds['v4'],key=int)
        assert all(set(p)==set(users) for p in preds.values())
        a,b=tick(start+' 00:00:00'),tick(end+' 00:00:00')
        q='''SELECT o.user,e.item,COUNT(*) FROM
          (SELECT user,event_id FROM overlaps WHERE start<? AND end>?
           GROUP BY user,event_id HAVING MAX(MAX(0,MIN(end,?)-MAX(start,?)))>
           .8*(SELECT duration FROM events WHERE events.event_id=overlaps.event_id)) o
          JOIN events e USING(event_id) GROUP BY o.user,e.item'''
        counts=defaultdict(dict)
        for u,i,n in db.execute(q,(b,a,b,a)):
            counts[u][i]=n
        numeric={m:[] for m in preds};marginal={m:[] for m in preds};ambiguous=[]
        for u in users:
            ranked=sorted(counts[u],key=lambda i:(-counts[u][i],int(i)))
            truth=ranked[:5];den=len(truth)
            tied_boundary=len(ranked)>5 and counts[u][ranked[4]]==counts[u][ranked[5]]
            ambiguous.append(tied_boundary)
            if tied_boundary:
                cutoff=counts[u][ranked[4]]
                fixed={i for i in ranked if counts[u][i]>cutoff}
                tied={i for i in ranked if counts[u][i]==cutoff}
                remaining=5-len(fixed)
            else:
                fixed,tied,remaining=set(truth),set(),0
            for method,pred in preds.items():
                p=pred[u];assert len(p)==len(set(p))==5
                numeric[method].append(ap5(p,truth))
                marginal[method].append(expected_ap(p,fixed,tied,remaining,den))
        mask=np.array(ambiguous);methods={}
        for method in preds:
            methods[method]={}
            for metric,values in [('numeric_tie_map5',numeric),('tie_marginal_expected_map5',marginal)]:
                x=np.array(values[method]);baseline=np.array(values['v4'])
                methods[method][metric]={name:{'users':int(ix.sum()),'score':float(x[ix].mean()),
                   'delta_vs_v4':float((x-baseline)[ix].mean())}
                   for name,ix in [('all',np.ones(len(users),bool)),('boundary_tied',mask),('unique_boundary',~mask)]}
        results[window]={'methods':methods,'source_predictions':{k:str(p.relative_to(B)) for k,p in paths.items()}}
    training=training_audit(db)
    db.close()
    report={'status':'complete','results':results,'wall_seconds':time.perf_counter()-started,
            'definition':'Uniform random subset of r slots from t programmes tied at fifth qualifying-airing count; items above boundary always relevant. Exact expected AP uses marginal r/t and joint r(r-1)/(t(t-1)). Full denominator min(5,relevantcount), emptyusers0.',
            'limitations':'This is a sensitivity diagnostic under an explicit tie distribution, not a claim about official targets. Unique-boundary stratum includes users with fewer than six relevant items, including empty users. Reused overlapping windows; no model selection proof.',
            'training_fit_apr13_may18':training,'fits':0,'submissions':0}
    (B/'evidence/tie_marginal_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    for window,result in results.items():
        print(window,json.dumps({m:v for m,v in result['methods'].items()},indent=2))


if __name__=='__main__':
    main()
