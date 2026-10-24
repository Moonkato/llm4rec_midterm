"""Inference-only frozen event/v3 equal-RRF60 candidate; explicit --run required."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse, ast, csv, json, resource, sqlite3, sys, time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from threadpoolctl import threadpool_limits
from run_airing_model import B, ROOT, sha, atomic_json
from airing_inference_features import histories, schedule, eventfeatures
from build_interactions import tick
from check_history_normalization import evaluate


def check_helper_parity():
    """Ensure function bodies exactly match executed historical experiment."""
    original=ast.parse((B/'tools/run_airing_model.py').read_text())
    main=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    extracted=ast.parse((B/'tools/airing_inference_features.py').read_text())
    for name in ('histories','schedule','eventfeatures'):
        a=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in extracted.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert [ast.dump(n) for n in a.body]==[ast.dump(n) for n in b.body],name


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true')
    if not ap.parse_args().run:raise SystemExit('Prepared only; explicit root approval needed for --run')
    started=time.perf_counter();last=[started]
    out=B/'evidence/candidate_event_v3_rrf_v4';cache=B/'cache/airing_candidate'
    dest=B/'candidates/event_v3_rrf_v4.csv'
    if dest.exists() or (out/'protocol.json').exists():raise SystemExit('Preserve existing candidate/run')
    check_helper_parity()
    completed=json.loads((B/'evidence/airing_model/results.json').read_text())
    assert completed['status']=='complete'
    assert completed['code_sha256']==sha(B/'tools/run_airing_model.py')
    out.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    def log(value):
        value['wall_seconds']=time.perf_counter()-started;print(json.dumps(value),flush=True)
        with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
    def guard(phase=None):
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        if time.perf_counter()-started>1200 or rss>2000000000:raise RuntimeError('Inference20min/2GB budget exceeded')
        if phase and time.perf_counter()-last[0]>25:log({'phase':phase,'peak_rss_bytes':rss});last[0]=time.perf_counter()
    inputs=[B/'cache/context_features/test.npz',B/'cache/test_weekly/test_row_features.npy',
        B/'cache/airing_model/model.txt',B/'cache/deep_temporal/top5_count80_24features.txt',
        B/'cache/interactions.sqlite',ROOT/'midterm/data/dataset11-30.csv',ROOT/'midterm/data/submission.csv',
        B/'candidates/weekly_ranker_v3.csv',B/'tools/run_airing_model.py',B/'tools/airing_inference_features.py',Path(__file__)]
    protocol={'status':'inference only; not submitted','history_cutoff':'2020-07-27 00:00:00',
        'forecast_end':'2020-10-19 00:00:00','models':'frozen event300trees + frozen v3 417trees',
        'ranking':'equal .5/.5 RRF60 with numeric ID ties and unchanged common cold popularity fallback',
        'event_features':'identical33 to executed experiment; static helper function body AST parity checked',
        'event_aggregation':'sum identical tuple probability times multiplicity; boundary fraction<=.8 hard zero',
        'inputs':'historical raw sessions + supplied future schedule only; no test behaviour or labels queried',
        'candidate_pool':'unchanged original test context rows','fits':0,'submissions':0,
        'budget':'20min2GB1CPU; expected3-5min under1GB',
        'sha256':{str(p.relative_to(ROOT)):sha(p) for p in inputs}}
    atomic_json(out/'protocol.json',protocol)
    with np.load(inputs[0],allow_pickle=False) as z:d={k:z[k] for k in z.files}
    weekly=np.load(inputs[1]);assert d['X'].shape[1]==19 and weekly.shape==(len(d['X']),5)
    v3=lgb.Booster(model_file=str(inputs[3]));event=lgb.Booster(model_file=str(inputs[2]))
    assert v3.num_feature()==24 and v3.num_trees()==417
    assert event.num_feature()==33 and event.num_trees()==300
    x=np.concatenate([d['X'],weekly],axis=1);assert np.isfinite(x).all()
    v3scores=v3.predict(x,num_threads=1);del x,weekly,v3
    _,v3pred=evaluate(d,v3scores)
    with inputs[7].open(newline='') as f:prior={r['user_id']:r['tv_show_id'].split() for r in csv.DictReader(f)}
    assert v3pred==prior,'Frozen v3 prediction parity failed'
    np.save(cache/'v3_test_scores.npy',v3scores)
    db=sqlite3.connect('file:'+str(inputs[4])+'?mode=ro',uri=True)
    db.execute('PRAGMA temp_store=FILE');db.execute('PRAGMA cache_size=-64000')
    channels=sorted((r[0] for r in db.execute('SELECT DISTINCT channel FROM events')),key=int)
    chmap={c:j for j,c in enumerate(channels)}
    cut,end=tick(protocol['history_cutoff']),tick(protocol['forecast_end'])
    ev,groups=schedule(cut,end,db);del ev
    eligible={r[0] for r in db.execute('SELECT DISTINCT item FROM events WHERE start<? AND end>?',(end,cut))}
    h=histories(d,cut,chmap,guard,log)
    expected=np.zeros(len(d['X']),dtype=np.float64);pending=[];destrows=[];multiplicity=[];feasible=[];scored=0
    def flush():
        nonlocal scored
        if not pending:return
        matrix=np.concatenate(pending);prob=event.predict(matrix,num_threads=1)
        prob[~np.asarray(feasible,dtype=bool)]=0
        np.add.at(expected,np.asarray(destrows,dtype=np.int64),prob*np.asarray(multiplicity));scored+=len(prob)
        pending.clear();destrows.clear();multiplicity.clear();feasible.clear();guard('forecast_batches')
    for j,u0 in enumerate(d['users']):
        u=str(u0);a,b=d['group_offsets'][j:j+2]
        for row in range(a,b):
            item=str(d['item_ids'][d['row_item_indices'][row]]);tuples=list(groups[item])
            pending.append(eventfeatures(d['X'][row],u,tuples,h,cut,chmap))
            destrows.extend([row]*len(tuples));multiplicity.extend(groups[item].values());feasible.extend(t[-1]>.8 for t in tuples)
            if len(destrows)>=65536:flush()
        if j%100==0:guard('forecast_assembly')
    flush();db.close();assert np.isfinite(expected).all() and np.isfinite(v3scores).all()
    np.save(cache/'event_test_expected_counts.npy',expected)
    fused=np.zeros(len(expected))
    for a,b in zip(d['group_offsets'][:-1],d['group_offsets'][1:]):
        ids=d['item_ids'][d['row_item_indices'][a:b]]
        for scores in (expected,v3scores):
            order=sorted(range(b-a),key=lambda k:(-scores[a+k],int(ids[k])))
            ranks=np.empty(b-a);ranks[order]=np.arange(1,b-a+1);fused[a:b]+=.5/(60+ranks)
    np.save(cache/'rrf_test_scores.npy',fused)
    _,pred=evaluate(d,fused)
    with (ROOT/'midterm/data/submission.csv').open(newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;users=[r['user_id'] for r in reader]
    assert fields==['user_id','tv_show_id']
    assert len(users)==len(set(users))==1260 and users==list(map(str,d['users'])) and set(pred)==set(users)
    rows=[]
    for j,u in enumerate(users):
        top=pred[u];a,b=d['group_offsets'][j:j+2]
        assert len(top)==len(set(top))==5 and '0' not in top and set(top)<=eligible
        assert set(top)<=set(map(str,d['item_ids'][d['row_item_indices'][a:b]]))
        if d['cold'][j]:assert top==list(map(str,d['item_ids'][d['fallback_item_indices'][:5]]))
        rows.append({'user_id':u,'tv_show_id':' '.join(top)})
    tmp=dest.with_suffix('.csv.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    with tmp.open(newline='') as f:assert list(csv.DictReader(f))==rows
    tmp.replace(dest)
    manifest={'status':'prepared, not submitted','rows':len(rows),'cold_users':int(d['cold'].sum()),
        'fits':0,'submissions':0,'scored_collapsed_rows':scored,'frozen_v3_ranking_parity':True,
        'changed_ranked_lists_vs_v3':sum(pred[u]!=prior[u] for u in users),
        'checks':['1260 ordered string users','5 unique nonzero eligible IDs','original candidate membership','cold fallback','finite scores','CSV readback','helper AST parity'],
        'wall_seconds':time.perf_counter()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        'sha256':{**protocol['sha256'],str(dest.relative_to(ROOT)):sha(dest)},
        'local_evidence':'evidence/airing_model/comparison.json; reused windows; later improvement uncertainty near zero',
        'test_labels_used':False}
    atomic_json(out/'manifest.json',manifest);log(manifest)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
