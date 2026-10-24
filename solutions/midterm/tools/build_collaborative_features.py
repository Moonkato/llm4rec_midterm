"""Expand historical CF retrieval and append four scores; no model fitting."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import hashlib
import json
import sqlite3
import time
from pathlib import Path
import numpy as np
from build_context_features import BASE, FEATURES, WINDOWS, build_window

EXTRA=['usercf_score','itemassociation_score','usercf_rrf60','itemassociation_rrf60']


def main():
    source_evidence=BASE/'evidence/item_association/results.json'
    if not source_evidence.exists():raise SystemExit('Item association completion required before feature preparation')
    out=BASE/'cache/collaborative_features';evidence=BASE/'evidence/collaborative_features'
    if (evidence/'results.json').exists():raise SystemExit('Preserve completed collaborative features')
    out.mkdir(parents=True,exist_ok=True);evidence.mkdir(parents=True,exist_ok=True)
    protocol={'windows':WINDOWS,'features':FEATURES+EXTRA,'fits':0,'submissions':0,
      'generation':'original personal100+popularity30+channel/hour30; append top20 positive userCF and top20 positive itemassociation scores; preserve previous candidate order',
      'sources':'historical-only CF scores computed separately before respective cutoffs',
      'ties':'numeric item ID ascending; reciprocal rank 1/(60+rank) across full forecast pool',
      'rrf_zero_policy':'zero score gets zero reciprocal-rank feature',
      'truth':'original strict single-fragment >80% labels; never used to select candidates',
      'schema':'same context_features NPZ, with23 feature columns and extended groups',
      'builder_sha256':hashlib.sha256((BASE/'tools/build_context_features.py').read_bytes()).hexdigest()}
    (evidence/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    db=sqlite3.connect('file:'+str(BASE/'cache/interactions.sqlite')+'?mode=ro',uri=True)
    db.execute('PRAGMA temp_store=FILE');db.execute('PRAGMA cache_size=-64000')
    results=[];began=time.perf_counter()
    for name,cut,end,isfit in WINDOWS:
        with np.load(BASE/f'cache/collaborative/{name}_catalog.npz',allow_pickle=False) as z:c={k:z[k] for k in z.files}
        candidates=c['candidate_ids'];users=c['target_users']
        assert candidates.tolist()==sorted(candidates.tolist(),key=int)
        paths=[BASE/f'cache/user_neighbors/{name}_scores.npy',BASE/f'cache/item_association/{name}_scores.npy']
        scores=[np.load(p,mmap_mode='r') for p in paths]
        assert all(s.shape==(len(users),len(candidates)) for s in scores)
        extra={};rank_scores=[]
        for m,s in enumerate(scores):
            assert np.isfinite(s).all()
            order=np.argsort(-s,axis=1,kind='stable')
            ranks=np.empty_like(order,dtype=np.int32)
            np.put_along_axis(ranks,order,np.broadcast_to(np.arange(1,len(candidates)+1),order.shape),axis=1)
            rank_scores.append(np.where(s>0,1.0/(60+ranks),0).astype(np.float32))
            for j,u in enumerate(users):extra.setdefault(str(u),[]).extend(str(candidates[k]) for k in order[j,:20] if s[j,k]>0)
            del order,ranks
        summary=build_window(db,name,cut,end,isfit,out,evidence,extra_candidates=extra)
        with np.load(out/f'{name}.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
        assert np.array_equal(d['users'],users)
        lookup={str(i):j for j,i in enumerate(candidates)}
        added=np.empty((len(d['y']),4),dtype=np.float32)
        for j in range(len(users)):
            a,b=d['group_offsets'][j:j+2]
            ids=d['item_ids'][d['row_item_indices'][a:b]]
            ix=[lookup[str(i)] for i in ids]
            added[a:b]=np.stack([scores[0][j,ix],scores[1][j,ix],rank_scores[0][j,ix],rank_scores[1][j,ix]],axis=1)
        d['X']=np.concatenate([d['X'],added],axis=1);d['feature_names']=np.asarray(FEATURES+EXTRA)
        assert np.isfinite(d['X']).all()
        np.savez_compressed(out/f'{name}.npz',**d)
        with np.load(BASE/f'cache/context_features/{name}.npz',allow_pickle=False) as z:
            assert np.array_equal(z['users'],d['users']) and np.array_equal(z['truth_counts'],d['truth_counts'])
            old_rows=len(z['y'])
        if name!='fit':
            oldtruth=json.loads((BASE/f'evidence/context_features/{name}_truth.json').read_text())
            assert oldtruth==json.loads((evidence/f'{name}_truth.json').read_text())
        summary.update(features=23,old_rows=old_rows,added_rows=len(d['y'])-old_rows,
            npz_sha256=hashlib.sha256((out/f'{name}.npz').read_bytes()).hexdigest(),
            source_score_hashes={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        results.append(summary);print(json.dumps(summary),flush=True)
        del d,added,scores,rank_scores
    db.close()
    (evidence/'results.json').write_text(json.dumps({'results':results,'wall_seconds':time.perf_counter()-began},indent=2)+'\n')


if __name__=='__main__':main()
