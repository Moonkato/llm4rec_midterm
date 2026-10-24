"""Saved-list paired comparison against frozen v4; no fitting."""
import json
from pathlib import Path
import numpy as np
from compare_airing_model import ap5, paired

B=Path(__file__).resolve().parents[1]


def read(p):
    return json.loads(p.read_text())


def main():
    out=B/'evidence/airing_recurrence'
    marker=out/'results.json'
    if not marker.exists():
        raise SystemExit('Recurrence results not complete yet')
    result=read(marker)
    assert result['status']=='complete' and result['fits']==1
    refs={r['window']:r for r in result['results']}
    report={}
    for window in ('eval','later'):
        baseline=read(B/f'evidence/airing_model/{window}_equal_rrf60_v3_predictions.json')
        users=sorted(baseline,key=int)
        assert len(users)==1260
        truths={'all80':read(B/f'evidence/context_features/{window}_truth.json'),
                'top5_count80':read(B/f'evidence/top5_target_audit/{window}_truth.json')['top5_count80']}
        report[window]={}
        for method in ('equal_rrf60_v3','expected_count'):
            new=read(out/f'{window}_recurrence_{method}_predictions.json')
            assert set(new)==set(baseline)
            report[window][method]={}
            for target,truth in truths.items():
                assert set(truth)==set(baseline)
                a=np.array([ap5(baseline[u],truth[u]) for u in users])
                b=np.array([ap5(new[u],truth[u]) for u in users])
                assert np.isclose(a.mean(),refs[window]['frozen_v4_equal_rrf60'][target],atol=1e-12,rtol=0)
                assert np.isclose(b.mean(),refs[window]['variants']['recurrence'][method][target],atol=1e-12,rtol=0)
                report[window][method][target]=dict(baseline_v4_map5=float(a.mean()),new_map5=float(b.mean()),**paired(b-a))
    payload=dict(status='complete',primary='equal_rrf60_v3 compared against frozen v4',diagnostic='expected_count',
                 comparisons=report,bootstrap_replicates=5000,seed=20261023,
                 caveat='Full-truth denominator and emptyusers0; fixed models, reused overlapping windows, unadjusted paired percentile intervals. No independent hidden-score guarantee.',
                 reviewer_fits=0,submissions=0)
    (out/'comparison.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
