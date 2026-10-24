"""Independent fixed-stack versus v4 paired AP comparison; no fitting."""
import json
import sqlite3
from pathlib import Path
import numpy as np
from compare_airing_model import ap5, paired
from tie_control_utils import count_qualified, tie_membership, expected_ap

B=Path(__file__).resolve().parents[1]


def main():
    out=B/'evidence/event_stack'
    marker=out/'results.json'
    if not marker.exists():
        raise SystemExit('Stack results not complete')
    run=json.loads(marker.read_text())
    assert run['status']=='complete' and run['fits']==1
    controls_dir=B/'evidence/tie_controls'
    controls_marker=controls_dir/'results.json'
    if not controls_marker.exists():
        raise SystemExit('Tie controls not complete')
    controls=json.loads(controls_marker.read_text())
    oof=json.loads((B/'evidence/event_oof/results.json').read_text())
    assert controls['status']=='complete' and controls['fits']==2
    assert oof['status']=='complete' and oof['fits']==3
    controls_refs={r['variant']:r for r in controls['results']}
    refs={r['window']:r for r in run['results']}
    db=sqlite3.connect('file:'+str(B/'cache/interactions.sqlite')+'?mode=ro',uri=True)
    db.execute('PRAGMA temp_store=FILE');db.execute('PRAGMA cache_size=-32000')
    results={};control_comparisons={}
    for window,start,end in [('eval','2020-05-18','2020-07-27'),('later','2020-06-22','2020-07-27')]:
        counts=count_qualified(db,start,end)
        old=json.loads((B/f'evidence/airing_model/{window}_equal_rrf60_v3_predictions.json').read_text())
        new=json.loads((out/f'{window}_predictions.json').read_text())
        users=sorted(old,key=int)
        assert len(users)==1260 and set(new)==set(old)
        info={u:tie_membership(counts.get(u,{})) for u in users}
        arrays={}
        for name,pred in [('v4',old),('stack',new)]:
            arrays[name]={
                'all80':np.array([ap5(pred[u],counts.get(u,{})) for u in users]),
                'top5_count80':np.array([ap5(pred[u],info[u]['hard']) for u in users]),
                'tie_marginal_expected_ap5':np.array([expected_ap(pred[u],info[u]) for u in users])}
            for metric in ('all80','top5_count80'):
                expected=refs[window]['scores' if name=='stack' else 'v4'][metric]
                assert np.isclose(arrays[name][metric].mean(),expected,atol=1e-12,rtol=0)
        results[window]={}
        for metric in arrays['v4']:
            a,b=arrays['v4'][metric],arrays['stack'][metric]
            results[window][metric]={'v4':float(a.mean()),'stack':float(b.mean()),**paired(b-a)}
        control_values={}
        for variant in ('hard_numeric_top5','fractional_boundary_membership'):
            for method in ('standalone','event_rrf60'):
                pred=json.loads((controls_dir/f'{variant}_{method}_{window}_predictions.json').read_text())
                assert set(pred)==set(old)
                value={
                    'all80':np.array([ap5(pred[u],counts.get(u,{})) for u in users]),
                    'top5_count80':np.array([ap5(pred[u],info[u]['hard']) for u in users]),
                    'tie_marginal_expected_ap5':np.array([expected_ap(pred[u],info[u]) for u in users])}
                aliases={'all80':'all80_map5','top5_count80':'hard_top5_map5',
                         'tie_marginal_expected_ap5':'exact_tie_expected_ap5'}
                for metric,alias in aliases.items():
                    assert np.isclose(value[metric].mean(),controls_refs[variant][window][method][alias],atol=1e-12,rtol=0)
                control_values[variant,method]=value
        control_comparisons[window]={}
        for method in ('standalone','event_rrf60'):
            comparisons={}
            for metric in arrays['v4']:
                hard=control_values['hard_numeric_top5',method][metric]
                soft=control_values['fractional_boundary_membership',method][metric]
                v4=arrays['v4'][metric]
                comparisons[metric]={'hard':float(hard.mean()),'soft':float(soft.mean()),
                                     'soft_minus_hard':paired(soft-hard),
                                     'hard_minus_v4':paired(hard-v4),'soft_minus_v4':paired(soft-v4)}
            control_comparisons[window][method]=comparisons
    db.close()
    payload={'status':'complete','comparisons':results,'secondary_controls':control_comparisons,
             'overall_authorized_fits':{'event_oof':3,'CE_controls':2,'fixed_stack':1,'total':6},
             'bootstrap_replicates':5000,'seed':20261023,
             'primary':'Prespecified fixed hardtop5 LambdaRank30 against frozen v4; no posthoc objective or fusion selection',
             'caveats':['Fulltruth denominators and emptyusers0; no candidate-conditioned denominator.',
                        'Uniform boundary-tie expectation is a declared sensitivity metric, not the known Kaggle target.',
                        'Paired user intervals conditional on fixed models and reused overlapping holdouts; not adjusted for multiple comparisons.',
                        'A counterfactual choice of the best control/objective after seeing these holdouts is posthoc model selection, not the prespecified primary result. No control outcome changed the fixed stack target.'],
             'fits':0,'submissions':0}
    (out/'comparison.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))


if __name__=='__main__':
    main()
