"""Exploratory cached-score fusion; no fitting, downloads, or submissions."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import hashlib
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
from run_round1 import apk

BASE = Path(__file__).resolve().parents[1]


def main():
    out = BASE/'evidence/model_fusion'
    if (out/'results.json').exists():
        raise SystemExit('Completed experiment exists; preserve it.')
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        'selection': 'exploratory same-holdout selection; no independent confirmation',
        'source': 'round1 immutable saved scores',
        'cbf_weights': [0.05, 0.10, 0.20],
        'cbf_formula': '(1-w)*popularity/max(popularity)+w*cosine; nonfinite cosine becomes zero',
        'als_weights': [0.05, 0.10, 0.20],
        'als_formula': '(1-w)*(.75*RRF(personal_frequency)+.25*RRF(popularity))+w*RRF(als_f32_a1)',
        'rrf': '1/(60+rank), rank starts at 1; nonfinite component scores contribute zero',
        'truth': 'round1 split=1, distinct programme IDs with >=180s per user-airing',
        'cohort': 'all 1260 round1 users; empty truth AP=0',
        'candidate_pool': 'unchanged round1 validation-airing catalogue',
        'repeats': 'allowed', 'ties': 'ascending numeric programme ID',
        'cold': 'no split=0 qualifying history; common historical popularity fallback',
        'training_runs': 0, 'kaggle_submissions': 0,
    }
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    began = time.perf_counter()
    catalog = json.loads((BASE/'evidence/round1/catalog.json').read_text())
    users, candidates = catalog['users'], catalog['candidates']
    assert candidates == sorted(candidates, key=int)
    truth = {u: set() for u in users}
    db = sqlite3.connect('file:'+str(BASE/'cache/interactions.sqlite')+'?mode=ro', uri=True)
    warm = {u for (u,) in db.execute('SELECT DISTINCT user FROM watches WHERE split=0 AND seconds>=180')}
    for u, item in db.execute('SELECT DISTINCT w.user,e.item FROM watches w JOIN events e USING(event_id) WHERE w.split=1 AND w.seconds>=180'):
        if u in truth:
            truth[u].add(item)
    db.close()
    cold = np.array([u not in warm for u in users])
    cache = BASE/'cache/round1'
    sources = ['popularity_unique_viewers', 'personal_frequency', 'content_tfidf', 'als_f32_a1']
    raw = {name: np.load(cache/f'{name}_scores.npy', mmap_mode='r') for name in sources}
    pop = raw['popularity_unique_viewers']
    fallback = np.argsort(-pop[0], kind='stable')[:5]

    def rrf(scores):
        order = np.argsort(-scores, axis=1, kind='stable')
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.broadcast_to(np.arange(1, len(candidates)+1), order.shape), axis=1)
        return np.where(np.isfinite(scores), 1.0/(60+ranks), 0.0)

    pop_normalized = pop / np.maximum(pop.max(axis=1, keepdims=True), 1)
    cosine = np.where(np.isfinite(raw['content_tfidf']), raw['content_tfidf'], 0.0)
    frozen = .75*rrf(raw['personal_frequency']) + .25*rrf(pop)
    als = rrf(raw['als_f32_a1'])
    results, predictions = [], {}

    def evaluate(name, scores, extra):
        top = np.argsort(-scores, axis=1, kind='stable')[:, :5]
        top[cold] = fallback
        recs = {u: [candidates[i] for i in row] for u, row in zip(users, top)}
        assert all(len(v) == len(set(v)) == 5 for v in recs.values())
        values = np.array([apk(truth[u], recs[u]) for u in users])
        result = {'method': name, 'map5': float(values.mean()),
                  'map5_nonempty_truth': float(np.mean([v for u, v in zip(users, values) if truth[u]])),
                  'cold_map5': float(values[cold].mean()), 'cold_users': int(cold.sum()), **extra}
        results.append(result)
        predictions[name] = recs
        print(json.dumps(result), flush=True)

    evaluate('reference_popularity', pop, {})
    evaluate('reference_frozen_personal75_pop25_rrf60', frozen, {})
    for w in protocol['cbf_weights']:
        evaluate(f'pop_content_w{w:.2f}', (1-w)*pop_normalized+w*cosine, {'content_weight': w})
    for w in protocol['als_weights']:
        evaluate(f'frozen_als_w{w:.2f}', (1-w)*frozen+w*als, {'als_weight': w})
    (out/'results.json').write_text(json.dumps({'results': results, 'wall_seconds': time.perf_counter()-began,
        'status': 'complete', 'interpretation': protocol['selection'],
        'source_score_sha256': {name: hashlib.sha256((cache/f'{name}_scores.npy').read_bytes()).hexdigest() for name in sources},
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2)+'\n')
    (out/'predictions.json').write_text(json.dumps(predictions, indent=2)+'\n')


if __name__ == '__main__':
    main()
