"""Frozen airing feature formulas copied from reviewed training source; explicit dependencies.

No fitting or execution on import. Helper equivalence is checked structurally by exporter.
"""
import bisect, csv, math
from collections import defaultdict
from datetime import datetime
import numpy as np
from run_airing_model import ROOT, EXTRA
from build_interactions import tick
from build_weekly_context import segments

def histories(d, cut, chmap, guard, log):
    wanted = set(map(str, d['users']))
    lens = defaultdict(list)
    ulens = defaultdict(list)
    ct = defaultdict(float)
    slotsec = defaultdict(float)
    with (ROOT / 'midterm/data/dataset11-30.csv').open(encoding='utf-8-sig', newline='') as f:
        for n, r in enumerate(csv.DictReader(f)):
            if n % 100000 == 0:
                guard('historical_context')
            u = r['user_id']
            ch = r['vsetv_id']
            if u not in wanted or ch not in chmap:
                continue
            a = max(0, tick(r['start_time']))
            if a >= cut:
                continue
            b = min(cut, tick(r['stop_time']))
            if b <= a:
                continue
            duration = b - a
            lens[u, ch].append(duration)
            ulens[u].append(duration)
            weight = min(1.0, 3600 / duration)
            ct[u, ch] += duration * weight
            for slot, day, left, right in segments(a, b):
                slotsec[u, ch, slot] += (right - left) * weight
    for v in lens.values():
        v.sort()
    for v in ulens.values():
        v.sort()
    denom = np.zeros(168)
    for slot, day, a, b in segments(0, cut):
        denom[slot] += b - a
    for key in slotsec:
        slotsec[key] = min(1.0, slotsec[key] / denom[key[2]])
    guard()
    log({'phase': 'history_ready', 'users': len(wanted), 'channel_groups': len(lens), 'slots': len(slotsec)})
    return (lens, ulens, ct, slotsec)

def schedule(cut, end, db):
    events = defaultdict(list)
    groups = defaultdict(dict)
    for eid, item, ch, a, b, dur, civil in db.execute('SELECT event_id,item,channel,start,end,duration,civil_start FROM events WHERE start<? AND end>?', (end, cut)):
        dt = datetime.fromisoformat(civil)
        t = (ch, dt.weekday(), dt.hour, (dur + 299) // 300 * 300, (min(end, b) - max(cut, a)) / dur)
        events[item].append((eid, t))
        groups[item][t] = groups[item].get(t, 0) + 1
    return (events, groups)

def eventfeatures(base, u, tuples, h, cut, chmap):
    lens, ulens, ct, slots = h
    arr = np.empty((len(tuples), len(base) + len(EXTRA)), dtype=np.float32)
    arr[:, :len(base)] = base
    fallback = ulens.get(u, [])
    for j, (ch, dow, hour, duration, coverage) in enumerate(tuples):
        seq = lens.get((u, ch), fallback)
        n = len(seq)
        survival = (n - bisect.bisect_left(seq, 0.8 * duration)) / n if n else 0.0
        arr[j, len(base):] = [chmap[ch], dow, math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24), math.log1p(duration), coverage, min(1.0, ct.get((u, ch), 0) / cut), slots.get((u, ch, dow * 24 + hour), 0), survival, math.log1p(len(lens.get((u, ch), []))), int((u, ch) not in lens), (math.expm1(float(base[1])) + 0.1) / (math.expm1(float(base[4])) * (cut / 86400) + 1), (math.expm1(float(base[0])) + 0.1) / (math.expm1(float(base[4])) * (cut / 86400) + 1), math.expm1(float(base[7])) / (cut / 86400)]
    return arr
