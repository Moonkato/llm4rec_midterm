"""Apply frozen logistic coefficients; export a candidate, never submit."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[key] = '1'
import csv
import hashlib
import json
import math
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from build_interactions import tick
from run_round1 import apk

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]


def predict(db, users, model, cut, end):
    origin = tick('2020-03-09 00:00:00')
    hist = defaultdict(dict)
    query = '''SELECT o.user,e.item,e.duration,
        MAX(MAX(0,MIN(o.end,?)-MAX(o.start,?))),MAX(MIN(o.end,?))
        FROM overlaps o JOIN events e USING(event_id)
        WHERE o.start<? AND o.end>? GROUP BY o.user,o.event_id'''
    for u, i, duration, seconds, last in db.execute(query, (cut, origin, cut, cut, origin)):
        if seconds <= .3*duration:
            continue
        v = hist[u].setdefault(i, [0, 0, 0., 0])
        v[0] += 1
        v[1] += seconds > .8*duration
        v[2] += math.exp(-math.log(2)*(cut-last)/(60*86400))
        v[3] = max(v[3], last)
    past = dict(db.execute('SELECT item,COUNT(*) FROM events WHERE start<? AND end>? GROUP BY item', (cut, origin)))
    future = dict(db.execute('SELECT item,COUNT(*) FROM events WHERE start<? AND end>? GROUP BY item', (end, cut)))
    pop = Counter()
    for history in hist.values():
        pop.update(history.keys())
    fallback = sorted(future, key=lambda i: (-pop[i], int(i)))
    assert len(fallback) >= 5 and '0' not in future
    glob = fallback[:30]
    predictions, cold_count = {}, 0
    for u in users:
        history = hist.get(u, {})
        total = sum(v[0] for v in history.values())
        def base(i):
            return history[i][0]*math.sqrt((future[i]/(end-cut))/(past.get(i, 1)/(cut-origin)))
        personal = sorted((i for i in history if i in future), key=lambda i: (-base(i), int(i)))[:100]
        candidates = list(dict.fromkeys(personal+glob))
        x = []
        for i in candidates:
            c, s, r, last = history.get(i, [0, 0, 0., 0])
            fp = future[i]*86400/(end-cut)
            pp = past.get(i, 0)*86400/(cut-origin)
            score = c*math.sqrt(fp/pp) if pp else 0
            x.append([math.log1p(c), math.log1p(s), math.log1p(r), math.log1p(fp),
                      math.log1p(pp), math.log1p(score), math.log1p(pop[i]), math.log1p(total),
                      min((cut-last)/86400, 365) if last else 365])
        scores = ((np.array(x)-model['mean'])/model['scale'])@model['coefficients'][:-1]+model['coefficients'][-1]
        assert np.isfinite(scores).all()
        if not history:
            top = fallback[:5]
            cold_count += 1
        else:
            top = [candidates[j] for j in sorted(range(len(candidates)), key=lambda j: (-scores[j], int(candidates[j])))[:5]]
        assert len(top) == len(set(top)) == 5 and set(top) <= set(future)
        predictions[u] = top
    return predictions, {'eligible_items': len(future), 'cold_users': cold_count,
                         'history_population_users': len(hist), 'max_history_endpoint': cut}


def main():
    began = time.perf_counter()
    output = BASE/'candidates/logistic_v1.csv'
    evidence = BASE/'evidence/candidate_logistic_v1.json'
    if output.exists() or evidence.exists():
        raise SystemExit('Candidate exists; preserve it.')
    output.parent.mkdir(parents=True, exist_ok=True)
    protocol = {'status': 'candidate only; not submitted', 'model': 'frozen cache/simple_reranker.npz',
        'history': '2020-03-09 to 2020-07-27 exclusive', 'forecast': '2020-07-27 to 2020-10-19 exclusive',
        'features': json.loads((BASE/'evidence/simple_reranker/protocol.json').read_text())['features'],
        'candidate_generation': 'same fitted model: top100 personal schedule frequency + top30 train popularity',
        'fragment_limitation': 'maximum clipped raw fragment across stored splits; original sessions cannot be reconstructed, artificial boundary fragments are not joined',
        'test_usage': 'only schedule item IDs, event times and exposure counts; no test views or metadata',
        'training_runs': 0, 'submissions': 0}
    (BASE/'evidence/candidate_logistic_v1_protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    with (ROOT/'midterm/data/submission.csv').open() as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        users = [r['user_id'] for r in reader]
    assert fields == ['user_id', 'tv_show_id'] and len(users) == len(set(users)) == 1260
    model_path = BASE/'cache/simple_reranker.npz'
    model = np.load(model_path)
    db = sqlite3.connect('file:'+str(BASE/'cache/interactions.sqlite')+'?mode=ro', uri=True)
    recs, checks = predict(db, users, model, tick('2020-07-27 00:00:00'), tick('2020-10-19 00:00:00'))
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({'user_id': u, 'tv_show_id': ' '.join(recs[u])} for u in users)
    cut, end = tick('2020-06-22 00:00:00'), tick('2020-07-27 00:00:00')
    val_recs, val_checks = predict(db, users, model, cut, end)
    truth = {u: set() for u in users}
    query = '''SELECT o.user,e.item,e.duration,MAX(MAX(0,MIN(o.end,?)-MAX(o.start,?)))
        FROM overlaps o JOIN events e USING(event_id) WHERE o.start<? AND o.end>?
        GROUP BY o.user,o.event_id'''
    for u, item, duration, seconds in db.execute(query, (end, cut, end, cut)):
        if u in truth and seconds > .8*duration:
            truth[u].add(item)
    db.close()
    ref = BASE/'evidence/strict_temporal/per_user.json'
    if ref.exists():
        rows = json.loads(ref.read_text())
        assert all(len(truth[r['user']]) == r['truth_size'] for r in rows)
    values = [apk(truth[u], val_recs[u]) for u in users]
    validation = {'cutoff': '2020-06-22', 'end': '2020-07-27', 'map5_strict': sum(values)/len(users),
                  'nonempty_map5_strict': sum(values)/sum(bool(v) for v in truth.values()),
                  'empty_truth_users': sum(not v for v in truth.values()), **val_checks,
                  'limitation': 'frozen coefficients; temporal robustness check overlaps previously explored validation periods'}
    (BASE/'evidence/candidate_logistic_v1_temporal_predictions.json').write_text(json.dumps(val_recs, indent=2)+'\n')
    with output.open() as f:
        actual = list(csv.DictReader(f))
    assert [r['user_id'] for r in actual] == users
    assert all(len(r['tv_show_id'].split()) == len(set(r['tv_show_id'].split())) == 5 and '0' not in r['tv_show_id'].split() for r in actual)
    result = {**protocol, 'file': str(output.relative_to(ROOT)), 'rows': len(actual), 'checks': checks,
              'validation': validation, 'wall_seconds': time.perf_counter()-began,
              'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [output, model_path, Path(__file__), ROOT/'midterm/data/submission.csv']}}
    evidence.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
