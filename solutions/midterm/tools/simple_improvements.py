"""Ten fixed lightweight frequency/schedule ablations; no fitted model."""
import hashlib
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
    out = BASE/'evidence/simple_improvements'
    if (out/'results.json').exists():
        raise SystemExit('Preserve completed experiment.')
    out.mkdir(parents=True, exist_ok=True)
    grid = [dict(name=f'count_p{p}_prior{prior}', power=p, prior=prior, half_life=None, strict_weight=0.0)
            for prior in (0, 10) for p in (.25, .5, .75)]
    grid += [dict(name=f'decay{days}_p0.5', power=.5, prior=0, half_life=days, strict_weight=0.0)
             for days in (60, 120)]
    grid += [dict(name=f'strict_blend{w}_p0.5', power=.5, prior=0, half_life=None, strict_weight=w)
             for w in (.25, .5)]
    protocol = {
        'grid': grid, 'selection': 'exploratory original strict-label holdout; not independent confirmation',
        'truth': 'unchanged evidence/strict_labels/truth.json, all1260, empty truth AP0',
        'train': 'split0, original cutoff 2020-05-18',
        'count_rule': 'one user-airing when MAX(raw overlap duration)>30% full duration',
        'strict_count_rule': 'one user-airing when MAX(raw overlap duration)>80% full duration',
        'formula': '((1-strict_weight)*count30+strict_weight*count80)*ratio**power; decay variants replace count30 with decayed count30',
        'ratio': '(forecast_airings+prior)/(train_airings+prior); equal 70-day windows; zero denominator => zero',
        'exposure': 'all deduplicated scheduled events intersecting corresponding window',
        'recency': 'last end among >30% qualifying raw fragments; half-life days before cutoff',
        'fallback': 'fixed >30% train unique-user popularity, restricted to same round1 candidates',
        'repeats': 'allowed', 'ties': 'ascending numeric item ID',
        'idea_source': 'https://medium.com/data-science/two-layered-recommender-system-methodology-a-prize-winning-solution-c4bdb815b156',
        'training_runs': 0, 'submissions': 0,
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
    exposures = [dict(db.execute('SELECT item,COUNT(*) FROM events WHERE start<? AND end>? GROUP BY item', (b, a)))
                 for a, b in [(origin, cut), (cut, end)]]
    counts, strict = defaultdict(Counter), defaultdict(Counter)
    recent = {d: defaultdict(Counter) for d in (60, 120)}
    query = '''SELECT o.user,e.item,o.event_id,e.duration,MAX(o.end-o.start),MAX(o.end)
               FROM overlaps o JOIN events e USING(event_id)
               WHERE o.split=0 AND o.end-o.start>0.30*e.duration
               GROUP BY o.user,o.event_id'''
    events = 0
    for u, item, event, duration, longest, last in db.execute(query):
        events += 1
        counts[u][item] += 1
        if longest > .8*duration:
            strict[u][item] += 1
        age = max(0, cut-last)/86400
        for days in recent:
            recent[days][u][item] += math.exp(-math.log(2)*age/days)
    db.close()
    pop = Counter()
    for history in counts.values():
        pop.update(history.keys())
    fallback = sorted(candidates, key=lambda i: (-pop[i], int(i)))
    cold = {u for u in users if not counts.get(u)}
    results, predictions = [], {}
    for config in grid:
        prior, power = config['prior'], config['power']
        multiplier = {i: ((exposures[1].get(i, 0)+prior)/(exposures[0].get(i, 0)+prior))**power
                      if exposures[0].get(i, 0)+prior else 0.0 for i in candidates}
        matrix = recent[config['half_life']] if config['half_life'] else counts
        recs = {}
        w = config['strict_weight']
        for u in users:
            scores = {i: ((1-w)*value+w*strict.get(u, {}).get(i, 0))*multiplier[i]
                      for i, value in matrix.get(u, {}).items() if i in allowed}
            top = sorted((i for i in scores if scores[i] > 0), key=lambda i: (-scores[i], int(i)))[:5]
            for i in fallback:
                if len(top) == 5:
                    break
                if i not in top:
                    top.append(i)
            assert len(top) == len(set(top)) == 5
            recs[u] = top
        values = [apk(truth[u], recs[u]) for u in users]
        result = {'method': config['name'], 'map5_strict': sum(values)/len(users),
                  'nonempty_map5_strict': sum(values)/sum(bool(truth[u]) for u in users),
                  'cold_users': len(cold), 'cold_map5_strict': sum(v for u, v in zip(users, values) if u in cold)/len(cold)}
        results.append(result)
        predictions[config['name']] = recs
        print(json.dumps(result), flush=True)
    (out/'predictions.json').write_text(json.dumps(predictions, indent=2)+'\n')
    (out/'results.json').write_text(json.dumps({'status': 'complete', 'results': results,
        'wall_seconds': time.perf_counter()-began, 'train_qualifying_events': events,
        'interpretation': protocol['selection'],
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2)+'\n')


if __name__ == '__main__':
    main()
