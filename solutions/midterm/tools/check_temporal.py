"""Frozen-rule temporal robustness check; no learned models or submissions."""
import csv
import hashlib
import itertools
import json
import resource
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

from build_interactions import tick
from run_round1 import apk

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]


def split_union(intervals, origin, cut, end):
    """Union ordered intervals before splitting, including old split fragments."""
    totals = [0, 0]
    low = high = None

    def add(a, b):
        totals[0] += max(0, min(b, cut) - max(a, origin))
        totals[1] += max(0, min(b, end) - max(a, cut))

    for a, b in intervals:
        if b <= a:
            raise ValueError("nonpositive overlap interval")
        if low is None:
            low, high = a, b
        elif a <= high:
            high = max(high, b)
        else:
            add(low, high)
            low, high = a, b
    if low is not None:
        add(low, high)
    return totals


def main():
    began = time.perf_counter()
    out = BASE / 'evidence/temporal'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'results.json').exists():
        raise SystemExit('Completed temporal check exists; preserve it.')
    protocol = {
        'train_start': '2020-03-09 00:00:00',
        'cutoff': '2020-06-22 00:00:00',
        'validation_end_exclusive': '2020-07-27 00:00:00',
        'threshold_seconds': 180,
        'truth': 'all distinct qualifying programmes, >=180 unioned seconds per airing',
        'cohort': 'all 1260 template users, empty truth AP=0', 'k': 5,
        'candidate_pool': 'nonzero programme IDs with airing intersecting forecast window',
        'repeats': 'allowed', 'ties': 'ascending numeric programme ID',
        'methods': ['popularity_unique_viewers', 'personal_frequency', 'frozen_rrf'],
        'frozen_rrf': {'personal_weight': 0.75, 'popularity_weight': 0.25, 'constant': 60,
                       'unseen_personal_items': 'zero reciprocal-rank contribution'},
        'fallback': 'historical distinct-viewer popularity, identical for every method',
        'interval_handling': 'reunite original split fragments; union by user/event; clip on each side of new cutoff',
        'selection': 'frozen from earlier holdout; no weight search here',
        'limitation': 'overlapping time windows with earlier experiment; not statistically independent confirmation',
        'training_runs': 0, 'kaggle_submissions': 0,
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    # Numerical fixture specifically covers duplicated and old-boundary fragments.
    assert split_union([(0, 100), (50, 100), (100, 200), (300, 400)], 0, 150, 350) == [150, 100]
    origin, cut, end = map(tick, [protocol['train_start'], protocol['cutoff'], protocol['validation_end_exclusive']])
    audit = json.loads((BASE / 'evidence/interaction_audit.json').read_text())
    assert audit['status'] == 'complete'
    with (ROOT / 'midterm/data/submission.csv').open() as f:
        users = [r['user_id'] for r in csv.DictReader(f)]
    assert len(users) == len(set(users)) == 1260
    personal = {u: Counter() for u in users}
    truth = {u: set() for u in users}
    db = sqlite3.connect('file:' + str(BASE / 'cache/interactions.sqlite') + '?mode=ro', uri=True)
    db.execute('PRAGMA temp_store=FILE')
    db.execute('PRAGMA cache_size=-64000')
    events = dict(db.execute('SELECT event_id,item FROM events'))
    candidates = sorted({r[0] for r in db.execute('SELECT DISTINCT item FROM events WHERE start<? AND end>?', (end, cut))}, key=int)
    assert len(candidates) >= 5 and '0' not in candidates
    population = Counter()
    stats = Counter()
    rows = db.execute('SELECT user,event_id,start,end FROM overlaps ORDER BY user,event_id,start,end')
    for user, user_rows in itertools.groupby(rows, key=lambda r: r[0]):
        history_items = set()
        for event_id, fragments in itertools.groupby(user_rows, key=lambda r: r[1]):
            def intervals():
                for _, _, a, b in fragments:
                    stats['raw_fragments'] += 1
                    yield a, b
            train_seconds, val_seconds = split_union(intervals(), origin, cut, end)
            item = events[event_id]
            if train_seconds >= 180:
                history_items.add(item)
                stats['train_user_airings'] += 1
                if user in personal:
                    personal[user][item] += 1
            if val_seconds >= 180:
                stats['validation_user_airings'] += 1
                if user in truth:
                    truth[user].add(item)
        population.update(history_items)
        stats['train_user_item_pairs'] += len(history_items)
        stats['users_processed'] += 1
        if stats['users_processed'] % 1000 == 0:
            print(json.dumps({'phase': 'reunion', 'users': stats['users_processed'], 'seconds': round(time.perf_counter() - began, 2)}), flush=True)
    db.close()
    candidate_set = set(candidates)
    assert all(v <= candidate_set for v in truth.values())
    pop_order = sorted(candidates, key=lambda i: (-population[i], int(i)))
    pop_rrf = {i: 1.0 / (60 + rank) for rank, i in enumerate(pop_order, 1)}
    predictions = {method: {} for method in protocol['methods']}
    per_user = []
    for user in users:
        known = sorted((i for i in personal[user] if i in candidate_set), key=lambda i: (-personal[user][i], int(i)))
        personal_top = known[:5]
        for i in pop_order:
            if len(personal_top) == 5:
                break
            if i not in personal_top:
                personal_top.append(i)
        personal_rrf = {i: 1.0 / (60 + rank) for rank, i in enumerate(known, 1)}
        fused = sorted(candidates, key=lambda i: (-(0.75 * personal_rrf.get(i, 0.0) + 0.25 * pop_rrf[i]), int(i)))[:5]
        recs = {'popularity_unique_viewers': pop_order[:5], 'personal_frequency': personal_top, 'frozen_rrf': fused}
        row = {'user': user, 'truth_size': len(truth[user]), 'cold': not bool(personal[user])}
        for method, top in recs.items():
            assert len(set(top)) == 5 and set(top) <= candidate_set
            predictions[method][user] = top
            row[method] = apk(truth[user], top)
        per_user.append(row)
    results = []
    for method in protocol['methods']:
        nonempty = [r[method] for r in per_user if r['truth_size']]
        cold = [r[method] for r in per_user if r['cold']]
        results.append({'method': method, 'map5': sum(r[method] for r in per_user) / len(users),
                        'map5_nonempty': sum(nonempty) / len(nonempty) if nonempty else None,
                        'cold_map5': sum(cold) / len(cold) if cold else None})
    report = {'status': 'complete', 'results': results, 'counts': dict(stats),
              'candidates': len(candidates), 'empty_truth_users': sum(not r['truth_size'] for r in per_user),
              'cold_users': sum(r['cold'] for r in per_user), 'wall_seconds': time.perf_counter() - began,
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'protocol_sha256': hashlib.sha256((out / 'protocol.json').read_bytes()).hexdigest(),
              'limitation': protocol['limitation'], 'fixture': 'split_union duplicates and reunited fragments passed'}
    (out / 'predictions.json').write_text(json.dumps(predictions, indent=2))
    (out / 'per_user.json').write_text(json.dumps(per_user, indent=2))
    (out / 'results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
