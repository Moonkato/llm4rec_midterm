"""One fixed user-neighbour CF model, with temporal dense/row feature scores."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
import numpy as np
from scipy import sparse
from run_round1 import apk

BASE=Path(__file__).resolve().parents[1]


def main():
    completion=BASE/'evidence/collaborative_inputs/results.json'
    if not completion.exists():raise SystemExit('Shared preprocessing is not complete')
    completed=json.loads(completion.read_text())
    assert {r['window'] for r in completed}=={'fit','eval','later'}
    out=BASE/'evidence/user_neighbors';cache=BASE/'cache/user_neighbors'
    if (out/'results.json').exists():raise SystemExit('Preserve completed experiment')
    out.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    protocol={'method':'userCF','neighbors':50,'similarity':'cosine of log1p historical >30% airing counts, row L2',
      'self':'always excluded','weights':'nonnegative cosine; selected positive weights normalized to sum1 per target',
      'score':'weighted average of neighbor binary item interactions, not calibrated watch probability',
      'neighbor_ties':'numeric user ID ascending','item_ties':'numeric programme ID ascending',
      'repeats':'allowed','fallback':'same >30% historical unique-user popularity for cold users or missing positive score slots',
      'labels':'none used to form similarities or scores; eval/later use frozen strict truth',
      'inputs':'completed shared collaborative matrices','grid':'none; one fixed top50 method',
      'outputs':'dense target-by-future-candidate scores and row features aligned to context_features',
      'submissions':0,'selection':'exploratory on previously explored overlapping validation windows'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    began=time.perf_counter();results=[]
    for name in ('fit','eval','later'):
        started=time.perf_counter();source=BASE/'cache/collaborative'
        counts=sparse.load_npz(source/f'{name}_counts.npz').astype(np.float32)
        with np.load(source/f'{name}_catalog.npz',allow_pickle=False) as z:c={k:z[k] for k in z.files}
        users,items=c['users'],c['items'];target=c['target_user_indices'];candidate=c['candidate_item_indices']
        assert counts.shape==(len(users),len(items))
        assert users.tolist()==sorted(users.tolist(),key=int)
        assert c['candidate_ids'].tolist()==sorted(c['candidate_ids'].tolist(),key=int)
        assert np.array_equal(users[target],c['target_users'])
        assert np.array_equal(items[candidate],c['candidate_ids'])
        x=counts.copy();x.data=np.log1p(x.data)
        norm=np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel())
        inverse=np.zeros_like(norm);np.divide(1,norm,out=inverse,where=norm>0)
        x=x.multiply(inverse[:,None]).tocsr()
        binary=counts.copy();binary.data[:]=1;binary=binary[:,candidate].tocsr()
        score_path=cache/f'{name}_scores.npy'
        scores=np.lib.format.open_memmap(score_path,mode='w+',dtype=np.float32,shape=(len(target),len(candidate)))
        zero_neighbors=0
        for a in range(0,len(target),256):
            b=min(a+256,len(target));similarities=(x[target[a:b]]@x.T).toarray()
            similarities[np.arange(b-a),target[a:b]]=-np.inf
            order=np.argsort(-similarities,axis=1,kind='stable')[:,:min(50,len(users)-1)]
            weights=np.take_along_axis(similarities,order,axis=1)
            weights=np.where(np.isfinite(weights)&(weights>0),weights,0)
            denom=weights.sum(axis=1);zero_neighbors+=int(np.sum(denom==0))
            np.divide(weights,denom[:,None],out=weights,where=denom[:,None]>0)
            rows=np.repeat(np.arange(b-a),order.shape[1])
            selected=sparse.csr_matrix((weights.ravel(),(rows,order.ravel())),shape=(b-a,len(users)))
            scores[a:b]=(selected@binary).toarray()
        scores.flush();assert np.isfinite(scores).all()
        pop=np.asarray(binary.sum(axis=0)).ravel();fallback=np.argsort(-pop,kind='stable')
        with np.load(BASE/f'cache/context_features/{name}.npz',allow_pickle=False) as z:context={k:z[k] for k in z.files if k!='X'}
        assert np.array_equal(context['users'],c['target_users'])
        item_lookup={str(v):i for i,v in enumerate(c['candidate_ids'])}
        row_features=np.zeros(len(context['y']),dtype=np.float32)
        for n in range(len(target)):
            lo,hi=context['group_offsets'][n:n+2]
            ids=context['item_ids'][context['row_item_indices'][lo:hi]]
            row_features[lo:hi]=scores[n,[item_lookup[str(i)] for i in ids]]
        np.save(cache/f'{name}_row_features.npy',row_features)
        np.savez(cache/f'{name}_catalog.npz',users=c['target_users'],candidate_ids=c['candidate_ids'])
        result={'window':name,'targets':len(target),'candidates':len(candidate),'zero_neighbors':zero_neighbors,
                'row_features':len(row_features)}
        if name!='fit':
            truth=json.loads((BASE/f'evidence/context_features/{name}_truth.json').read_text());predictions={};aps=[]
            for n,u in enumerate(c['target_users']):
                ranking=np.argsort(-scores[n],kind='stable')[:5]
                top=[] if norm[target[n]]==0 else [int(i) for i in ranking if scores[n,i]>0]
                for i in fallback:
                    if len(top)==5:break
                    if int(i) not in top:top.append(int(i))
                recs=[str(c['candidate_ids'][i]) for i in top]
                assert len(set(recs))==5 and '0' not in recs
                predictions[str(u)]=recs;aps.append(apk(truth[str(u)],recs))
            result.update(map5=float(np.mean(aps)),cold_users=int(np.sum(norm[target]==0)))
            (out/f'{name}_predictions.json').write_text(json.dumps(predictions)+'\n')
        result['wall_seconds']=time.perf_counter()-started
        results.append(result);print(json.dumps(result),flush=True)
        del scores,x,binary,counts
    manifest={'results':results,'wall_seconds':time.perf_counter()-began,
       'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
       'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'shared_completion_sha256':hashlib.sha256(completion.read_bytes()).hexdigest()}
    (out/'results.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
