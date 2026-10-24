"""No-fit fixed-prediction error and count calibration audit."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[k] = '1'
import json
import sqlite3
import time
from pathlib import Path
import numpy as np
from scipy import sparse
from build_interactions import tick

B = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def tier(n):
    return '0_to_10' if n <= 10 else '11_to_100' if n <= 100 else 'over_100'


def main():
    began = time.perf_counter()
    result = {}
    db = sqlite3.connect('file:' + str(B/'cache/interactions.sqlite') + '?mode=ro', uri=True)
    db.execute('PRAGMA temp_store=FILE')
    db.execute('PRAGMA cache_size=-32000')
    for window, start, end in [('eval', '2020-05-18', '2020-07-27'), ('later', '2020-06-22', '2020-07-27')]:
        with np.load(B/f'cache/context_features/{window}.npz') as z:
            d = {k:z[k] for k in z.files}
        with np.load(B/f'cache/collaborative/{window}_catalog.npz') as z:
            catalog = {k:z[k] for k in z.files}
        counts = sparse.load_npz(B/f'cache/collaborative/{window}_counts.npz')
        global_seen = set(map(str, catalog['items'][np.asarray(counts.sum(axis=0)).ravel()>0]))
        del counts
        users = list(map(str,d['users']))
        ids = list(map(str,d['item_ids']))
        truth = read(B/f'evidence/top5_target_audit/{window}_truth.json')['top5_count80']
        preds = {'v3':read(B/f'evidence/deep_temporal/top5_count80_24features_{window}_predictions.json'),
                 'v4':read(B/f'evidence/airing_model/{window}_equal_rrf60_v3_predictions.json')}
        a,b = tick(start+' 00:00:00'),tick(end+' 00:00:00')
        future_counts = dict(db.execute('SELECT item,COUNT(*) FROM events WHERE start<? AND end>? GROUP BY item',(b,a)))
        q='''SELECT o.user,e.item,COUNT(*) FROM
          (SELECT user,event_id FROM overlaps WHERE start<? AND end>?
           GROUP BY user,event_id HAVING MAX(MAX(0,MIN(end,?)-MAX(start,?)))>
           .8*(SELECT duration FROM events WHERE events.event_id=overlaps.event_id)) o
          JOIN events e USING(event_id) GROUP BY o.user,e.item'''
        actual = {(str(u),str(i)):n for u,i,n in db.execute(q,(b,a,b,a))}
        expected = np.load(B/f'cache/airing_model/{window}_expected_counts.npy')
        observed = np.zeros(len(expected))
        sources = np.empty(len(expected),dtype=object)
        airing_tiers = np.empty(len(expected),dtype=object)
        groups = np.repeat(np.arange(len(users)),np.diff(d['group_offsets']))
        user_tiers=[]; rowlookup={}
        for j,u in enumerate(users):
            lo,hi=d['group_offsets'][j:j+2]
            user_tiers.append(tier(round(float(np.expm1(d['X'][lo,7])))))
            for row in range(lo,hi):
                i=ids[d['row_item_indices'][row]];rowlookup[u,i]=row
                observed[row]=actual.get((u,i),0)
                sources[row]='seen' if d['X'][row,0]>0 else 'new_known' if i in global_seen else 'globally_new'
                airing_tiers[row]=tier(future_counts[i])
        ap={}
        for name,pred in preds.items():
            credit={k:0. for k in ('seen','new_known','globally_new')}
            airing_credit={k:0. for k in ('0_to_10','11_to_100','over_100')}
            ap_user=np.zeros(len(users));wrong={k:0 for k in credit}
            for j,u in enumerate(users):
                relevant=set(truth[u]);den=min(5,len(relevant));hits=0
                for rank,i in enumerate(pred[u],1):
                    row=rowlookup[u,i];source=sources[row]
                    if i in relevant:
                        hits+=1;v=hits/rank/den;ap_user[j]+=v
                        credit[source]+=v/len(users);airing_credit[airing_tiers[row]]+=v/len(users)
                    else:wrong[source]+=1
            ap[name]={'map5':float(ap_user.mean()),'additive_credit_by_source':credit,
                      'additive_credit_by_future_airings':airing_credit,'irrelevant_recommendations_by_source':wrong,
                      'activity_slices':{t:{'users':sum(x==t for x in user_tiers),
                        'map5':float(ap_user[np.array(user_tiers)==t].mean())} for t in sorted(set(user_tiers))}}
        def calibration(mask):
            ix=np.flatnonzero(mask);err=expected[ix]-observed[ix]
            ug=groups[ix];n=np.bincount(ug,minlength=len(users));valid=n>0
            bias=np.bincount(ug,weights=err,minlength=len(users))[valid]/n[valid]
            mae=np.bincount(ug,weights=np.abs(err),minlength=len(users))[valid]/n[valid]
            return {'pairs':len(ix),'users':int(valid.sum()),'expected_total':float(expected[ix].sum()),
                    'actual_total':float(observed[ix].sum()),'macro_user_pair_bias':float(bias.mean()),
                    'macro_user_pair_mae':float(mae.mean()),'pair_mean_expected':float(expected[ix].mean()),
                    'pair_mean_actual':float(observed[ix].mean())}
        result[window]={'ap_top5_count80':ap,'count80_calibration':{'all':calibration(np.ones(len(expected),bool)),
            'source':{s:calibration(sources==s) for s in sorted(set(sources))},
            'future_airings':{s:calibration(airing_tiers==s) for s in sorted(set(airing_tiers))}},
            'v4_minus_v3':ap['v4']['map5']-ap['v3']['map5']}
        assert time.perf_counter()-began<120, 'Diagnostic time budget exceeded'
    db.close()
    payload={'status':'complete','results':result,'wall_seconds':time.perf_counter()-began,
             'protocol':'Fixed v3/v4 predictions; top5_count80 full denominator and emptyusers0; historical activity count30 bins0-10/11-100/>100; futureairing bins same. Calibration uses original retrieved candidate pairs and compares expectedcount to actual count80.',
             'limitations':['Local proxy labels remain uncertain.','Stored fragments lack originalsessionID; MAX clips at forecast bounds and does not reunite split sessions.','Calibration is aggregate over selected candidate pairs, not exposure completeness or calibrated hidden-target probability.','AP credits are additive allocations of full MAP, not standalone source metrics.','Reused overlapping windows and diagnostic slices are not independent tests.'],
             'fits':0,'submissions':0}
    (B/'evidence/post_v4_diagnostic.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))


if __name__=='__main__':
    main()
