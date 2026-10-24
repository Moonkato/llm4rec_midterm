"""Prespecified combined regularity candidate versus frozen v3."""
import json
import numpy as np
from compare_deep_temporal import B, ap_vectors

def main():
    directory=B/'evidence/regularity_round';out=directory/'comparison.json'
    assert (directory/'results.json').exists()
    if out.exists():raise SystemExit('Preserve completed comparison')
    results={}
    for window in ('eval','later'):
        with np.load(B/f'cache/context_features/{window}.npz') as z:d={k:z[k] for k in z.files}
        truth=json.loads((B/f'evidence/top5_target_audit/{window}_truth.json').read_text())['top5_count80']
        base=json.loads((B/f'evidence/deep_temporal/top5_count80_24features_{window}_predictions.json').read_text())
        candidate=json.loads((directory/f'all_{window}_predictions.json').read_text())
        before=ap_vectors(d,base,truth);after=ap_vectors(d,candidate,truth);results[window]={}
        for metric in before:
            delta=after[metric]-before[metric];rng=np.random.default_rng(20261023)
            boot=np.concatenate([delta[rng.integers(0,len(delta),(100,len(delta)))].mean(axis=1) for _ in range(50)])
            results[window][metric]={'v3':float(before[metric].mean()),'combined_candidate':float(after[metric].mean()),
                'delta':float(delta.mean()),'paired_user_interval95':np.quantile(boot,[.025,.975]).tolist(),
                'users_improved':int((delta>0).sum()),'users_worsened':int((delta<0).sum())}
    report={'primary_candidate':'all, fixed before results','results':results,'fits':0,'submissions':0,
        'limitations':'Exploratory reused overlapping periods; user dependence and model selection not covered by bootstrap. Not Kaggle intervals.'}
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
