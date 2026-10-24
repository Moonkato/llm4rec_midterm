"""One streaming input audit. No joins, model fitting, or submissions."""
import csv
from collections import Counter
from datetime import datetime, timedelta
import json
from pathlib import Path
import platform
import time

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT/'midterm/data'
OUT = ROOT/'solutions/midterm/evidence/input_audit.json'

def quantiles(hist):
    total = sum(hist.values()); ordered = sorted(hist.items()); result = {}
    for q in (.0, .5, .9, .99, 1.):
        target = max(1, int(total*q+.999999)); acc = 0
        for value, count in ordered:
            acc += count
            if acc >= target:
                result[str(q)] = value; break
    return result

def schedule_time(s):
    return s[6:10]+'-'+s[3:5]+'-'+s[:2]+' '+s[11:]

def main():
    start = time.perf_counter()
    with (DATA/'submission.csv').open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f); submission_columns = reader.fieldnames
        rows = list(reader)
    users = {r['user_id'] for r in rows}
    report = {'environment': {'python': platform.python_version(), 'system': platform.system()},
              'submission': {'columns': submission_columns, 'rows': len(rows), 'distinct_users': len(users),
                'id_lengths': dict(Counter(map(len, users))),
                'users_above_signed_int64': sum(int(u)>2**63-1 for u in users),
                'users_above_js_safe_integer': sum(int(u)>2**53-1 for u in users),
                'list_lengths': dict(Counter(len(r['tv_show_id'].split()) for r in rows))}}
    historical_ids = set(); schedule_channels = set(); schedules = {}
    for p in sorted(DATA.glob('export*.csv')):
        n = labelled = invalid = out_of_order = overlapping = adjacent_dupes = gaps_equal_duration = 0
        channels = set(); ids = set(); days = Counter(); durations = Counter(); last = {}
        earliest = latest = last_end = None
        with p.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                n += 1; channel = r['channel_id']; item = r['tv_show_id']; dt = schedule_time(r['start_time'])
                duration = int(r['duration']); channels.add(channel); days[dt[:10]] += 1; durations[duration] += 1
                labelled += item != '0'
                if item != '0': ids.add(item)
                if duration <= 0: invalid += 1
                current = datetime.fromisoformat(dt); end = current+timedelta(seconds=duration)
                earliest = dt if earliest is None else min(earliest,dt)
                latest = dt if latest is None else max(latest,dt)
                last_end = end if last_end is None else max(last_end,end)
                prev = last.get(channel)
                if prev:
                    out_of_order += current < prev[0]
                    overlapping += current < prev[1]
                    gaps_equal_duration += current == prev[1]
                    adjacent_dupes += (current,end,item) == prev
                last[channel] = (current,end,item)
        schedules[p.name] = {'rows':n,'bytes':p.stat().st_size,'labelled_rows':labelled,
            'labelled_fraction':labelled/n,'distinct_labelled_items':len(ids),'channels':len(channels),
            'start_min':earliest,'start_max':latest,'end_max':str(last_end),
            'nonpositive_durations':invalid,'duration_seconds_quantiles':quantiles(durations),
            'out_of_order_within_channel':out_of_order,'overlap_with_previous_row_same_channel':overlapping,
            'exact_adjacent_duplicates':adjacent_dupes,'consecutive_starts_equal_previous_end':gaps_equal_duration,
            'new_item_ids_vs_previous_files':len(ids-historical_ids),'daily_rows':dict(sorted(days.items()))}
        historical_ids.update(ids); schedule_channels.update(channels)
        print(json.dumps({'schedule':p.name, **{k:v for k,v in schedules[p.name].items() if k!='daily_rows'}}),flush=True)
    report['schedules'] = schedules
    cutoff = '2020-05-18 00:00:00'; test_start = '2020-07-27 00:00:00'
    n = mismatches = invalid = crosses_val = crosses_test = after_test = 0
    all_users = set(); channels = Counter(); daily = Counter(); durations = Counter()
    train_users = Counter(); val_users = Counter(); min_start = max_start = max_stop = None
    with (DATA/'dataset11-30.csv').open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            n += 1; uid = r['user_id']; a = r['start_time']; b = r['stop_time']; duration = int(r['duraton'])
            all_users.add(uid); channels[r['vsetv_id']] += 1; daily[a[:10]] += 1; durations[duration] += 1
            elapsed = (datetime.fromisoformat(b)-datetime.fromisoformat(a)).total_seconds()
            mismatches += elapsed != duration; invalid += elapsed <= 0
            crosses_val += a < cutoff < b; crosses_test += a < test_start < b; after_test += a >= test_start
            if uid in users:
                if a < cutoff: train_users[uid] += 1
                if b > cutoff and a < test_start: val_users[uid] += 1
            min_start = a if min_start is None else min(min_start,a)
            max_start = a if max_start is None else max(max_start,a)
            max_stop = b if max_stop is None else max(max_stop,b)
    report['views'] = {'rows':n,'distinct_users':len(all_users),'channels':len(channels),
        'start_min':min_start,'start_max':max_start,'stop_max':max_stop,'nonpositive_intervals':invalid,
        'duration_mismatches':mismatches,'duration_seconds_quantiles':quantiles(durations),
        'sessions_crossing_validation_boundary':crosses_val,'sessions_crossing_test_boundary':crosses_test,
        'sessions_starting_in_test':after_test,'template_users_absent_all_history':len(users-all_users),
        'template_users_absent_11_20':len(users-set(train_users)),
        'template_users_absent_21_30':len(users-set(val_users)),
        'view_channels_absent_from_all_schedules':sorted(set(channels)-schedule_channels),
        'view_rows_channel_absent_from_schedules':sum(v for k,v in channels.items() if k not in schedule_channels),
        'daily_rows':dict(sorted(daily.items())),
        'template_train_session_count_quantiles':quantiles(Counter(train_users.get(u,0) for u in users)),
        'template_val_session_count_quantiles':quantiles(Counter(val_users.get(u,0) for u in users))}
    report['wall_seconds'] = round(time.perf_counter()-start,3)
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({'views':{k:v for k,v in report['views'].items() if k!='daily_rows'},
        'submission':report['submission'],'wall_seconds':report['wall_seconds'],'saved':str(OUT.relative_to(ROOT))}),flush=True)

if __name__ == '__main__': main()
