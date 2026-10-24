"""Bounded schedule-rate adjustment of historical frequency; no model fitting."""
import json
import math
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

from build_interactions import tick
from run_round1 import apk

BASE = Path(__file__).resolve().parents[1]


def main():
    out = BASE/'evidence/schedule'
    if (out/'results.json').exists():
        raise SystemExit('Preserve completed schedule experiment.')
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        'selection': 'exploratory same-holdout strict-label analysis; no independent confirmation',
        'idea_source': 'https://medium.com/data-science/two-layered-recommender-system-methodology-a-prize-winning-solution-c4bdb815b156',
        'source_use': 'schedule-change adjustment motivation; these formulas are our implementation',
        'train': '2020-03-09 inclusive to 2020-05-18 exclusive',
        'forecast': '2020-05-18 inclusive to 2020-07-27 exclusive',
        'window_days': {'train': 70, 'forecast': 70},
        'train_qualification': 'MAX raw overlap fragment per user/airing > 0.30 * full airing duration',
        'counts': 'one event per qualified user-airing, not one fragment',
        'truth': 'unchanged evidence/strict_labels/truth.json, all 1260 users, empty truth AP=0',
        'exposure': 'distinct deduplicated events intersecting each window; aggregate across channels',
        'ratio': '(forecast airings/70)/(train airings/70); zero train exposure yields zero ratio',
        'count_powers': [0.0, 0.5, 1.0], 'recency_powers': [0.0, 0.5],
        'recency': '28-day half-life at training cutoff; last endpoint among qualifying raw fragments per airing',
        'candidate_pool': 'same round1 candidates; repeats allowed',
        'fallback': 'same train unique-user popularity under >30% single-fragment rule, restricted to candidates',
        'cold': 'no qualifying historical events => full common popularity fallback',
        'ties': 'ascending numeric item ID', 'training_runs': 0, 'submissions': 0,
    }
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    began = time.perf_counter()
    cat = json.loads((BASE/'evidence/round1/catalog.json').read_text())
    users, candidates = cat['users'], cat['candidates']
    allowed = set(candidates)
    truth = json.loads((BASE/'evidence/strict_labels/truth.json').read_text())
    assert set(truth) == set(users)
    origin, cut, end = map(tick, ['2020-03-09 00:00:00', '2020-05-18 00:00:00', '2020-07-27 00:00:00'])
    db = sqlite3.connect('file:'+str(BASE/'cache/interactions.sqlite')+'?mode=ro', uri=True)
    exposures = []
    for start, stop in [(origin, cut), (cut, end)]:
        exposures.append(dict(db.execute('SELECT item,COUNT(*) FROM events WHERE start<? AND end>? GROUP BY item', (stop, start))))
    train_exposure, future_exposure = exposures
    ratio = {i: (future_exposure.get(i, 0)/70)/(train_exposure[i]/70) if train_exposure.get(i, 0) else 0.0 for i in candidates}
    counts, recent = defaultdict(Counter), defaultdict(Counter)
    query = '''SELECT o.user,e.item,o.event_id,MAX(o.end)
               FROM overlaps o JOIN events e USING(event_id)
               WHERE o.split=0 AND o.end-o.start>0.30*e.duration
               GROUP BY o.user,o.event_id'''
    for u, item, event, last in db.execute(query):
        counts[u][item] += 1
        recent[u][item] += math.exp(-math.log(2)*max(0, cut-last)/(28*86400))
    db.close()
    pop = Counter()
    for history in counts.values():
        pop.update(history.keys())
    fallback = sorted(candidates, key=lambda i: (-pop[i], int(i)))
    cold = {u for u in users if not counts.get(u)}
    results, predictions = [], {}

    def evaluate(name, recs, **kwargs):
        values = [apk(truth[u], recs[u]) for u in users]
        row = {'method': name, 'map5_strict': sum(values)/len(users),
               'nonempty_map5_strict': sum(values)/sum(bool(truth[u]) for u in users),
               'cold_users': len(cold),
               'cold_map5_strict': sum(v for u, v in zip(users, values) if u in cold)/len(cold) if cold else None,
               **kwargs}
        results.append(row)
        predictions[name] = recs
        print(json.dumps(row), flush=True)

    evaluate('pop_single_session_gt0.3', {u: fallback[:5] for u in users})
    for kind, matrix, powers in [('count', counts, protocol['count_powers']), ('decay28', recent, protocol['recency_powers'])]:
        for power in powers:
            recs = {}
            for u in users:
                history = matrix.get(u, {})
                scores = {i: value*(ratio[i]**power if power else 1.0) for i, value in history.items() if i in allowed}
                top = sorted((i for i in scores if scores[i] > 0), key=lambda i: (-scores[i], int(i)))[:5]
                for i in fallback:
                    if len(top) == 5:
                        break
                    if i not in top:
                        top.append(i)
                assert len(top) == len(set(top)) == 5
                recs[u] = top
            evaluate(f'personal_{kind}_schedule_power{power:.1f}', recs, power=power)
    (out/'predictions.json').write_text(json.dumps(predictions, indent=2)+'\n')
    (out/'results.json').write_text(json.dumps({'status': 'complete', 'results': results,
        'wall_seconds': time.perf_counter()-began, 'empty_truth_users': sum(not truth[u] for u in users),
        'interpretation': protocol['selection']}, indent=2)+'\n')


if __name__ == '__main__':
    main()
