"""Fixed 2x2 RRF60 comparison after both fits finish; never trains."""
import hashlib
import json
import sqlite3
import time
from pathlib import Path
import numpy as np
from run_deep_temporal import Metric
from run_objective_diversity import reciprocal_ranks
from compare_airing_model import ap5, paired
from tie_control_utils import count_qualified, tie_membership, expected_ap

B=Path(__file__).resolve().parents[1]
ARMS={'v4':('v3','old_event'),'replay_only':('v3','replay'),
      'identity_only':('identity','old_event'),'combined':('identity','replay')}


def main():
    started=time.perf_counter()
    markers={name:B/f'evidence/final_{name}/results.json' for name in ('identity','replay')}
    if not all(p.exists() for p in markers.values()):
        raise SystemExit('Both final identity and replay completion markers are required')
    completed={k:json.loads(p.read_text()) for k,p in markers.items()}
    assert all(r['status']=='complete' and r['fits']==1 for r in completed.values())
    out=B/'evidence/final_factorial';cache=B/'cache/final_factorial'
    if (out/'results.json').exists():
        raise SystemExit('Preserve completed factorial comparison')
    out.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    plan={'primary':'combined','arms':ARMS,'fusion':'equal0.5/0.5 RRF60; numeric item ties',
          'gate':{'main_hard_delta':.01,'later_hard_delta':.005,'paired95_lower_bounds':'positive both',
                  'tie_expected_delta':'positive both','all80_delta_min_each':-.002},
          'metrics':'all80, hardtop5count80, exact uniform boundary-tie expected AP; fulltruthdenominator, empty0',
          'bootstrap':{'replicates':5000,'seed':20261023},'fits':0,'submissions':0,
          'selection':'Combined prespecified primary. Other arms and conditional contributions diagnostic, no blend tuning or posthoc replacement of primary.'}
    (out/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
    hashes={};results={};contributions={}
    def record(p):
        hashes[str(p.relative_to(B))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for p in markers.values():record(p)
    db=sqlite3.connect('file:'+str(B/'cache/interactions.sqlite')+'?mode=ro',uri=True)
    db.execute('PRAGMA temp_store=FILE');db.execute('PRAGMA cache_size=-32000')
    for window,start,end in [('eval','2020-05-18','2020-07-27'),('later','2020-06-22','2020-07-27')]:
        context=B/f'cache/context_features/{window}.npz';record(context)
        with np.load(context,allow_pickle=False) as z:d={k:z[k] for k in z.files}
        users=list(map(str,d['users']));assert len(users)==len(set(users))==1260
        counts=count_qualified(db,start,end)
        info={u:tie_membership(counts.get(u,{})) for u in users}
        truth={'all80':{u:set(counts.get(u,{})) for u in users},'top5_count80':{u:info[u]['hard'] for u in users}}
        metric=Metric(d,truth['all80'])
        paths={'v3':B/f'cache/deep_temporal/top5_count80_24features_{window}_scores.npy',
               'old_event':B/f'cache/airing_model/{window}_expected_counts.npy',
               'identity':B/f'cache/final_identity/standalone_{window}_scores.npy',
               'replay':B/f'cache/final_replay/{window}_replay_expected_counts.npy'}
        rr={}
        for name,p in paths.items():
            record(p);scores=np.load(p)
            assert scores.shape==(len(d['y']),) and np.isfinite(scores).all()
            rr[name]=reciprocal_ranks(d,scores)
        values={};results[window]={}
        for arm,(ranker,event) in ARMS.items():
            fused=(rr[ranker]+rr[event])/2
            np.save(cache/f'{arm}_{window}_scores.npy',fused)
            pred=metric.predictions(fused)
            assert set(pred)==set(users)
            if arm=='v4':
                p=B/f'evidence/airing_model/{window}_equal_rrf60_v3_predictions.json';record(p)
                assert pred==json.loads(p.read_text()),'Frozen v4 ranking parity failed'
            (out/f'{arm}_{window}_predictions.json').write_text(json.dumps(pred,indent=2)+'\n')
            values[arm]={t:np.array([ap5(pred[u],v[u]) for u in users]) for t,v in truth.items()}
            values[arm]['tie_expected_ap5']=np.array([expected_ap(pred[u],info[u]) for u in users])
            assert np.isclose(values[arm]['all80'].mean(),metric.rank(fused)[0],atol=1e-12,rtol=0)
            if arm=='identity_only':
                ref=completed['identity']['results'][window]['old_event_rrf60']['scores']
            elif arm=='replay_only':
                ref=next(x for x in completed['replay']['results'] if x['window']==window)['variants']['replay']['equal_rrf60_v3']
            else:ref=None
            if ref:
                for t in truth:assert np.isclose(values[arm][t].mean(),ref[t],atol=1e-12,rtol=0)
        for arm,v in values.items():
            results[window][arm]={t:{'score':float(x.mean()),'v4_score':float(values['v4'][t].mean()),
                                     **paired(x-values['v4'][t])} for t,x in v.items()}
        contributions[window]={}
        for label,new,old in [('identity_given_replay','combined','replay_only'),
                              ('replay_given_identity','combined','identity_only')]:
            contributions[window][label]={t:paired(values[new][t]-values[old][t]) for t in values[new]}
        contributions[window]['factorial_interaction_descriptive']={
            t:float((values['combined'][t]-values['identity_only'][t]-values['replay_only'][t]+values['v4'][t]).mean())
            for t in values['v4']}
    db.close()
    checks={
        'main_hard_delta_at_least_001':results['eval']['combined']['top5_count80']['mean_delta']>=.01,
        'later_hard_delta_at_least_0005':results['later']['combined']['top5_count80']['mean_delta']>=.005,
        'hard_paired95_lower_positive_both':all(results[w]['combined']['top5_count80']['paired_user_percentile95'][0]>0 for w in results),
        'tie_expected_positive_both':all(results[w]['combined']['tie_expected_ap5']['mean_delta']>0 for w in results),
        'all80_loss_no_more_0002':all(results[w]['combined']['all80']['mean_delta']>=-.002 for w in results)}
    payload={'status':'complete','primary':'combined','results':results,'conditional_contributions':contributions,
             'primary_success_checks':checks,'primary_passes_screen':all(checks.values()),'source_hashes':hashes,
             'wall_seconds':time.perf_counter()-started,'new_training_fits':2,'reviewer_fits':0,'submissions':0,
             'caveats':['Four fixed arms include the frozen baseline; selecting a different winner after reading results is adaptive model selection.',
                        'Paired intervals are unadjusted and condition on these fitted models and reused overlapping windows; no unseen Kaggle guarantee.',
                        'Tie expectation is a sensitivity metric, not a confirmed organizer target.',
                        'Factorial differences describe predictive scores under fixed recipes, not causal effects.']}
    (out/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'primary':{w:results[w]['combined'] for w in results},'gate':checks},indent=2))


if __name__=='__main__':
    main()
