"""Independent saved-list AP@5 and paired user bootstrap; no fitting."""
import hashlib
import json
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[1]


def ap5(prediction, truth):
    assert len(prediction) == 5 and len(set(prediction)) == 5
    relevant = set(truth)
    if not relevant:
        return 0.0
    hits = 0
    credit = 0.0
    for rank, item in enumerate(prediction, 1):
        if item in relevant:
            hits += 1
            credit += hits / rank
    return credit / min(5, len(relevant))


def paired(delta):
    rng = np.random.default_rng(20261023)
    means = np.concatenate([
        delta[rng.integers(0, len(delta), (100, len(delta)))].mean(axis=1)
        for _ in range(50)
    ])
    return dict(mean_delta=float(delta.mean()),
                paired_user_percentile95=np.quantile(means, [.025, .975]).tolist(),
                users_improved=int((delta > 0).sum()),
                users_worsened=int((delta < 0).sum()),
                users_equal=int((delta == 0).sum()))


def main():
    out = BASE / 'evidence/airing_model'
    marker = out / 'results.json'
    if not marker.exists():
        raise SystemExit('Event results not complete yet; no comparison written')
    result = json.loads(marker.read_text())
    assert result['status'] == 'complete'
    refs = {x['window']: x for x in result['results']}
    sources = {}

    def read(path):
        raw = path.read_bytes()
        sources[str(path.relative_to(BASE))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    report = {}
    for window in ('eval', 'later'):
        old = read(BASE / f'evidence/deep_temporal/top5_count80_24features_{window}_predictions.json')
        truths = {'all80': read(BASE / f'evidence/context_features/{window}_truth.json'),
                  'top5_count80': read(BASE / f'evidence/top5_target_audit/{window}_truth.json')['top5_count80']}
        users = sorted(old, key=int)
        assert len(users) == 1260
        report[window] = {}
        for method in ('expected_count', 'equal_rrf60_v3'):
            new = read(out / f'{window}_{method}_predictions.json')
            assert set(new) == set(old)
            report[window][method] = {}
            for target, truth in truths.items():
                assert set(truth) == set(old)
                a = np.array([ap5(old[u], truth[u]) for u in users])
                b = np.array([ap5(new[u], truth[u]) for u in users])
                assert np.isclose(b.mean(), refs[window][method][target], atol=1e-12, rtol=0)
                report[window][method][target] = dict(
                    baseline_map5=float(a.mean()), model_map5=float(b.mean()),
                    empty_truth_users=sum(not truth[u] for u in users),
                    **paired(b - a))
    payload = dict(status='complete', comparisons=report, bootstrap_replicates=5000,
                   seed=20261023, users_per_window=1260,
                   metric='AP@5 denominator min(5, full truth size); empty truth zero; saved ordered lists',
                   caveat='Paired intervals condition on fitted models and reused cohorts. Overlapping windows are not independent confirmation; intervals are not adjusted for multiple comparisons.',
                   source_hashes=sources, reviewer_fits=0)
    path = out / 'comparison.json'
    path.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
