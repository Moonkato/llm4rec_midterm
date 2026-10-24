"""Prepare full-history context features or export an explicitly chosen model."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
import csv
import hashlib
import json
import pickle
import sqlite3
import time
from pathlib import Path
import numpy as np
from build_context_features import build_window, FEATURES

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--model',choices=['boost_old9','boost_context','boost_context_deeper'])
    args=parser.parse_args()
    if not args.prepare_only and not args.model:parser.error('Explicit --model selection required for export')
    began=time.perf_counter();out=BASE/'cache/context_features';evidence=BASE/'evidence/context_candidate'
    evidence.mkdir(parents=True,exist_ok=True)
    path=out/'test.npz'
    if not path.exists():
        protocol={'cutoff':'2020-07-27 00:00:00','forecast_end':'2020-10-19 00:00:00',
          'features':FEATURES,'no_training':True,'no_submissions':True,
          'labels':'unavailable; y/truth_counts are zero placeholders only; no forecast overlaps queried',
          'history':'all available actual history before cutoff; maximum raw fragment across stored splits',
          'future_inputs':'schedule item IDs, channels and airing times only; no future attributes or behaviour',
          'candidate_generation':'100 personal schedule +30 popularity +30 channel/hour context',
          'builder_sha256':hashlib.sha256((BASE/'tools/build_context_features.py').read_bytes()).hexdigest()}
        (evidence/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        db=sqlite3.connect('file:'+str(BASE/'cache/interactions.sqlite')+'?mode=ro',uri=True)
        db.execute('PRAGMA temp_store=FILE');db.execute('PRAGMA cache_size=-64000')
        result=build_window(db,'test',protocol['cutoff'],protocol['forecast_end'],False,out,evidence,collect_truth=False)
        db.close();(evidence/'features.json').write_text(json.dumps(result,indent=2)+'\n')
    if args.prepare_only:
        print(json.dumps({'prepared':str(path),'exported':False}),flush=True);return
    csv_path=BASE/'candidates/boost_context_v2.csv'
    if csv_path.exists():raise SystemExit('Preserve existing candidate.')
    model_path=BASE/'cache'/f'{args.model}.pkl'
    with model_path.open('rb') as f:model=pickle.load(f)
    with np.load(path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    assert list(data['feature_names'])==FEATURES
    with (ROOT/'midterm/data/submission.csv').open() as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;users=[r['user_id'] for r in reader]
    assert users==data['users'].tolist() and len(users)==len(set(users))==1260
    n=model.n_features_in_;scores=model.predict_proba(data['X'][:,:n])[:,1]
    assert np.isfinite(scores).all()
    predictions={};items=data['item_ids'];available=set(items)
    for j,u in enumerate(users):
        a,b=data['group_offsets'][j:j+2];idx=data['row_item_indices'][a:b]
        if data['cold'][j]:top=items[data['fallback_item_indices'][:5]].tolist()
        else:
            rank=sorted(range(b-a),key=lambda k:(-scores[a+k],int(items[idx[k]])))[:5]
            top=[str(items[idx[k]]) for k in rank]
        assert len(top)==len(set(top))==5 and '0' not in top and set(top)<=available
        predictions[u]=top
    csv_path.parent.mkdir(parents=True,exist_ok=True)
    with csv_path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({'user_id':u,'tv_show_id':' '.join(predictions[u])} for u in users)
    manifest={'status':'candidate only, not submitted','model':args.model,'features':n,'rows':len(users),
        'cold_users':int(data['cold'].sum()),'eligible_items':len(items),'wall_seconds':time.perf_counter()-began,
        'checks':'original ordered string IDs; 5 unique available nonzero programme IDs each',
        'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [csv_path,model_path,path,Path(__file__)]}}
    (evidence/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
