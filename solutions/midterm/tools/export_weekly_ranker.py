"""Export frozen 417-tree weekly ranker with full-history features; no fitting/network."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import lightgbm as lgb
from check_history_normalization import B, evaluate

def main():
    started=time.perf_counter();root=B.parents[1]
    dest=B/'candidates/weekly_ranker_v3.csv'
    manifest=B/'evidence/candidate_weekly_ranker_v3.json'
    if dest.exists() or manifest.exists():raise SystemExit('Preserve existing candidate')
    assert (B/'evidence/test_weekly/results.json').exists()
    base=B/'cache/context_features/test.npz'
    extra=B/'cache/test_weekly/test_row_features.npy'
    model_path=B/'cache/deep_temporal/top5_count80_24features.txt'
    with np.load(base,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    weekly=np.load(extra,allow_pickle=False)
    assert d['X'].shape[1]==19 and weekly.shape==(len(d['X']),5)
    x=np.concatenate([d['X'],weekly],axis=1)
    assert np.isfinite(x).all()
    model=lgb.Booster(model_file=str(model_path))
    assert model.num_feature()==24 and model.num_trees()==417
    scores=model.predict(x,num_threads=1)
    assert np.isfinite(scores).all()
    _,pred=evaluate(d,scores)
    with (root/'midterm/data/submission.csv').open(newline='') as f:template=list(csv.DictReader(f))
    users=[r['user_id'] for r in template]
    assert len(users)==len(set(users))==1260
    assert users==list(map(str,d['users'])) and set(users)==set(pred)
    eligible=set(map(str,d['item_ids']))
    recs=[]
    for n,u in enumerate(users):
        top=pred[u]
        assert len(top)==len(set(top))==5 and '0' not in top and set(top)<=eligible
        a,b=d['group_offsets'][n:n+2]
        assert set(top)<=set(map(str,d['item_ids'][d['row_item_indices'][a:b]]))
        if d['cold'][n]:assert top==list(map(str,d['item_ids'][d['fallback_item_indices'][:5]]))
        recs.append({'user_id':u,'tv_show_id':' '.join(top)})
    with dest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['user_id','tv_show_id']);w.writeheader();w.writerows(recs)
    with dest.open(newline='') as f:
        reader=csv.DictReader(f);assert reader.fieldnames==['user_id','tv_show_id'];assert list(reader)==recs
    with (B/'candidates/filtered_context_v2.csv').open(newline='') as f:old={r['user_id']:r['tv_show_id'].split() for r in csv.DictReader(f)}
    comparison=json.loads((B/'evidence/deep_temporal/comparison.json').read_text())
    files=[dest,base,extra,model_path,Path(__file__),B/'tools/build_weekly_context.py']
    result={'status':'prepared, not submitted','proposed_submission_name':'gleb_alektorov_midterm_03_weekly_lambdarank',
      'file':str(dest.relative_to(root)),'rows':len(recs),'cold_users':int(d['cold'].sum()),
      'changed_ranked_lists_vs_v2':sum(pred[u]!=old[u] for u in users),
      'changed_top1_vs_v2':sum(pred[u][0]!=old[u][0] for u in users),
      'changed_item_sets_vs_v2':sum(set(pred[u])!=set(old[u]) for u in users),
      'model':'frozen LightGBM LambdaRank top5_count80,24features,417trees',
      'training_labels':'Apr13-May18 only; unchanged model from validated run',
      'inference_history':'All available sessions before2020-07-27; forecast schedule July27-Oct19 IDs/channel/time/duration only',
      'test_labels_used':False,'additional_fits':0,'submissions_added':0,
      'checks':['template order and string IDs','1260uniqueusers','5unique nonzero IDs per user','all recommendations eligible and retrieved','cold fallback','finite features and scores','CSV readback'],
      'local_scores':{w:comparison['results'][w]['scores']['top5_count80_24features'] for w in ('eval','later')},
      'limitations':'Official preference construction unknown; local periods repeatedly explored; no guaranteed Kaggle improvement',
      'sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
      'wall_seconds':time.perf_counter()-started}
    manifest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
