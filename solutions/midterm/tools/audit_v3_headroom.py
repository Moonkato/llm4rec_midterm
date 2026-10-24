"""Descriptive retrieval and repeat-viewing headroom; no model fitting."""
import json
from pathlib import Path
import numpy as np
from run_round1 import apk
B=Path(__file__).resolve().parents[1]

def main():
    out=B/'evidence/v3_headroom.json'
    if out.exists():raise SystemExit('Preserve completed audit')
    results={}
    for window in ('eval','later'):
        with np.load(B/f'cache/context_features/{window}.npz') as z:d={k:z[k] for k in z.files}
        truth=json.loads((B/f'evidence/top5_target_audit/{window}_truth.json').read_text())['top5_count80']
        pred=json.loads((B/f'evidence/deep_temporal/top5_count80_24features_{window}_predictions.json').read_text())
        oracle=[];seen_oracle=[];recall=[];groups={k:[] for k in ('cold','1_to_20','21_to_100','over_100')}
        relevant=missing=known=0
        for n,uid in enumerate(d['users']):
            u=str(uid);a,b=d['group_offsets'][n:n+2];items=list(map(str,d['item_ids'][d['row_item_indices'][a:b]]))
            t=set(truth[u]);cand=set(items);seen={i for i,x in zip(items,d['X'][a:b,0]) if x>0}
            denominator=min(5,len(t));hits=len(t&cand)
            oracle.append(min(5,hits)/denominator if denominator else 0)
            seen_oracle.append(min(5,len(t&seen))/denominator if denominator else 0)
            relevant+=len(t);missing+=len(t-cand);known+=len(t&seen)
            count=round(float(np.expm1(d['X'][a,7])))
            group='cold' if d['cold'][n] else '1_to_20' if count<=20 else '21_to_100' if count<=100 else 'over_100'
            groups[group].append(apk(truth[u],pred[u]))
        results[window]={'actual_map5':float(np.mean([apk(truth[str(u)],pred[str(u)]) for u in d['users']])),
            'hindsight_candidate_oracle_map5':float(np.mean(oracle)),
            'hindsight_seen_candidate_oracle_map5':float(np.mean(seen_oracle)),
            'relevant_user_item_pairs':relevant,'relevant_pairs_missing_candidates':missing,
            'relevant_pairs_seen_and_retrieved':known,
            'historical_count30_cohorts':{k:{'users':len(v),'map5':float(np.mean(v)) if v else None} for k,v in groups.items()}}
    report={'target':'top5_count80 proxy','results':results,'fits':0,'submissions':0,
        'warning':'Hindsight oracles use labels and are diagnostic upper bounds, not achievable model scores. Seen fraction refers to retrieved items; nonretrieved items are not classified. Descriptive cohorts on explored windows, no causal inference.'}
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
