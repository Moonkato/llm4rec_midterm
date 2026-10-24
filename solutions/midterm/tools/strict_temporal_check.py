"""Frozen simple strict-label rules on a later, overlapping temporal window."""
import csv
import hashlib
import itertools
import json
import math
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


def main():
    began = time.perf_counter()
    out = BASE / 'evidence/strict_temporal'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'results.json').exists():
        raise SystemExit('Completed strict temporal check exists; preserve it.')
    protocol = {
        'train_start': '2020-03-09 00:00:00', 'cutoff': '2020-06-22 00:00:00',
        'validation_end_exclusive': '2020-07-27 00:00:00',
        'historical_label': 'maximum raw fragment clipped to history >30% full airing duration',
        'truth': 'maximum raw fragment clipped to forecast >80% full airing duration',
        'single_session_limitation': 'original May18 boundary can split one session into two fragments; no session identifier remains, so MAX can underestimate it; never union separate sessions',
        'cohort': 'all1260 template users, empty truth AP=0', 'k': 5,
        'candidate_pool': 'nonzero IDs with event intersecting forecast window',
        'repeats': 'allowed', 'ties': 'ascending numeric programme ID',
        'methods': ['popularity_strict30', 'personal_strict30', 'personal_strict30_airing_rate_sqrt'],
        'availability': 'sqrt((future_airings/forecast_days)/(historical_airings/train_days))',
        'airing_count_rule': 'distinct cached event rows intersecting period, across all channels',
        'duration_rule': 'days are clock-normalized period seconds/86400',
        'smoothing': None, 'clipping': None,
        'fallback': 'same distinct-viewer popularity from >30% historical labels',
        'selection': 'frozen rules; no parameter tuning or learned models',
        'limitation': 'forecast overlaps earlier validation; robustness check, not unseen independent confirmation; official Kaggle target still unverified',
        'training_runs': 0, 'kaggle_submissions': 0,
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    assert json.loads((BASE / 'evidence/interaction_audit.json').read_text())['status'] == 'complete'
    origin, cut, end = map(tick, [protocol['train_start'], protocol['cutoff'], protocol['validation_end_exclusive']])
    oldcut = tick('2020-05-18 00:00:00')
    with (ROOT / 'midterm/data/submission.csv').open() as f:
        users = [row['user_id'] for row in csv.DictReader(f)]
    assert len(users) == len(set(users)) == 1260
    personal = {u: Counter() for u in users}
    truth = {u: set() for u in users}
    db = sqlite3.connect('file:' + str(BASE / 'cache/interactions.sqlite') + '?mode=ro', uri=True)
    db.execute('PRAGMA temp_store=FILE')
    db.execute('PRAGMA cache_size=-64000')
    events = {eid: (item, duration) for eid, item, duration in db.execute('SELECT event_id,item,duration FROM events')}
    train_airings = dict(db.execute('SELECT item,COUNT(*) FROM events WHERE start<? AND end>? GROUP BY item', (cut, origin)))
    future_airings = dict(db.execute('SELECT item,COUNT(*) FROM events WHERE start<? AND end>? GROUP BY item', (end, cut)))
    candidates = sorted(future_airings, key=int)
    allowed = set(candidates)
    assert len(candidates) >= 5 and '0' not in allowed
    stats = Counter()
    stats['events_spanning_old_cutoff'] = db.execute('SELECT COUNT(*) FROM events WHERE start<? AND end>?', (oldcut, oldcut)).fetchone()[0]
    pop = Counter()
    query = '''SELECT user,event_id,
        MAX(MAX(0,MIN(end,?)-MAX(start,?))),
        MAX(MAX(0,MIN(end,?)-MAX(start,?))),COUNT(DISTINCT split)
        FROM overlaps GROUP BY user,event_id ORDER BY user,event_id'''
    rows = db.execute(query, (cut, origin, end, cut))
    for user, grouped in itertools.groupby(rows, key=lambda row: row[0]):
        history_items = set()
        for _, eid, train_seconds, val_seconds, old_splits in grouped:
            item, duration = events[eid]
            stats['user_event_groups'] += 1
            stats['groups_with_both_original_splits'] += old_splits == 2
            # Integer comparisons preserve strict >30% and >80% semantics.
            if 10 * train_seconds > 3 * duration:
                history_items.add(item)
                stats['train_qualifying_user_airings'] += 1
                if user in personal:
                    personal[user][item] += 1
            if 5 * val_seconds > 4 * duration:
                stats['forecast_qualifying_user_airings'] += 1
                if user in truth:
                    truth[user].add(item)
        pop.update(history_items)
        stats['train_user_item_pairs'] += len(history_items)
        stats['users_processed'] += 1
        if stats['users_processed'] % 1000 == 0:
            print(json.dumps({'phase': 'strict_aggregation', 'users': stats['users_processed'], 'seconds': round(time.perf_counter() - began, 2)}), flush=True)
    db.close()
    assert all(items <= allowed for items in truth.values())
    fallback = sorted(candidates, key=lambda i: (-pop[i], int(i)))
    rate_factor = {i: math.sqrt(future_airings[i] * (cut - origin) / (train_airings[i] * (end - cut)))
                   for i in candidates if train_airings.get(i, 0)}

    def finish(order):
        result = order[:5]
        for item in fallback:
            if len(result) == 5:
                break
            if item not in result:
                result.append(item)
        return result

    predictions = {method: {} for method in protocol['methods']}
    per_user = []
    for user in users:
        counts = personal[user]
        known = [i for i in counts if i in allowed]
        assert all(i in rate_factor for i in known)
        lists = {
            'popularity_strict30': fallback[:5],
            'personal_strict30': finish(sorted(known, key=lambda i: (-counts[i], int(i)))),
            'personal_strict30_airing_rate_sqrt': finish(sorted(known, key=lambda i: (-counts[i] * rate_factor[i], int(i)))),
        }
        row = {'user': user, 'truth_size': len(truth[user]), 'cold': not bool(counts)}
        for method, top in lists.items():
            assert len(top) == len(set(top)) == 5 and set(top) <= allowed
            predictions[method][user] = top
            row[method] = apk(truth[user], top)
        per_user.append(row)
    results = []
    for method in protocol['methods']:
        nonempty = [r[method] for r in per_user if r['truth_size']]
        results.append({'method': method, 'map5': sum(r[method] for r in per_user) / len(users),
                        'map5_nonempty': sum(nonempty) / len(nonempty) if nonempty else None})
    report = {'status': 'complete', 'results': results, 'counts': dict(stats),
              'candidates': len(candidates), 'empty_truth_users': sum(not r['truth_size'] for r in per_user),
              'cold_users': sum(r['cold'] for r in per_user), 'truth_pairs': sum(len(v) for v in truth.values()),
              'wall_seconds': time.perf_counter() - began,
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'protocol_sha256': hashlib.sha256((out / 'protocol.json').read_bytes()).hexdigest(),
              'limitation': protocol['limitation']}
    (out / 'predictions.json').write_text(json.dumps(predictions, indent=2))
    (out / 'per_user.json').write_text(json.dumps(per_user, indent=2))
    (out / 'results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
