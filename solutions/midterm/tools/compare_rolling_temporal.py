"""Paired comparison of the prespecified pooled temporal model, no fitting."""
import json
from pathlib import Path
import numpy as np
from run_round1 import apk
B=Path(__file__).resolve().parents[1]

def main():
    out=B/'evidence/rolling_temporal/comparison.json'
    if out.exists():raise SystemExit('Preserve completed comparison')
    assert (B/'evidence/rolling_temporal/results.json').exists()
    truth_all=json.loads((B/'evidence/rolling_temporal/later_truth.json').read_text())
    truth_top=json.loads((B/'evidence/top5_target_audit/later_truth.json').read_text())['top5_count80']
    candidate=json.loads((B/'evidence/rolling_temporal/pooled_early_recent_later_predictions.json').read_text())
    refs={'same_rolling_input':B/'evidence/rolling_temporal/frozen_v3_rolling_later_predictions.json',
          'original_full_history':B/'evidence/deep_temporal/top5_count80_24features_later_predictions.json'}
    users=list(candidate);results={}
    for ref,path in refs.items():
        base=json.loads(path.read_text());assert set(base)==set(users);results[ref]={}
        for metric,t in [('top5_count80',truth_top),('all80',truth_all)]:
            a=np.array([apk(t[u],candidate[u]) for u in users]);b=np.array([apk(t[u],base[u]) for u in users]);delta=a-b
            rng=np.random.default_rng(20261023)
            means=np.concatenate([delta[rng.integers(0,len(users),(100,len(users)))].mean(axis=1) for _ in range(50)])
            results[ref][metric]={'candidate':float(a.mean()),'baseline':float(b.mean()),'delta':float(delta.mean()),
                'paired_user_interval95':np.quantile(means,[.025,.975]).tolist()}
    report={'primary':'pooled_early_recent','window':'Jun22-Jul27 only','results':results,'fits':0,'submissions':0,
      'limitations':'Full-history reference changes history and candidate pool as well as fitting; same-rolling reference is controlled comparison. Reused observations; intervals conditional on predictions and ignore selection/user dependence.'}
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
