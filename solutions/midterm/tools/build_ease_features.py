"""Fixed EASE identity features on historical binary interactions; no reranker."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='2'
import argparse,json,time,tempfile,hashlib,resource,sys
from pathlib import Path
import numpy as np
from scipy import sparse,linalg
from threadpoolctl import threadpool_limits
BASE=Path(__file__).resolve().parents[1]
NAMES=['ease_score','ease_score_percentile','ease_user_unobserved','ease_item_unobserved']
WINDOWS=[('fit','2020-04-13','2020-05-18'),('eval','2020-05-18','2020-07-27'),('later','2020-06-22','2020-07-27')]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''):h.update(part)
    return h.hexdigest()

def ease(binary,regularization):
    gram=(binary.T@binary).toarray().astype(np.float64)
    gram.flat[::len(gram)+1]+=regularization
    factor=linalg.cho_factor(gram,lower=True,overwrite_a=True,check_finite=False)
    precision=linalg.cho_solve(factor,np.eye(len(gram),dtype=np.float64),check_finite=False)
    coefficients=-precision/np.diag(precision)[None,:]
    np.fill_diagonal(coefficients,0.)
    return coefficients

def fixture():
    x=np.array([[1.,1.,0.],[1.,0.,1.],[0.,1.,1.],[1.,1.,1.]])
    b=ease(sparse.csr_matrix(x),2.)
    direct=np.zeros_like(b)
    for j in range(x.shape[1]):
        other=np.arange(x.shape[1])!=j
        direct[other,j]=np.linalg.solve(x[:,other].T@x[:,other]+2*np.eye(other.sum()),x[:,other].T@x[:,j])
    assert np.array_equal(np.diag(b),np.zeros(3))
    assert np.allclose(b,direct,rtol=1e-12,atol=1e-12)
    assert np.allclose(x@b,x@direct,rtol=1e-12,atol=1e-12)
    return {'status':'passed','max_coefficient_error':float(np.max(np.abs(b-direct))),'meaning':'closed form equals independent per-target constrained ridge solve'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--fixture-only',action='store_true');args=parser.parse_args()
    test=fixture()
    if args.fixture_only:print(json.dumps(test));return
    if not args.run:raise SystemExit('Prepared only; explicit root --run required')
    began=time.perf_counter();cache=BASE/'cache/ease_features';out=BASE/'evidence/ease_features'
    if cache.exists() or (out/'results.json').exists():raise SystemExit('Preserve completed EASE outputs')
    out.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.ease_features-',dir=BASE/'cache'))
    protocol={'features':NAMES,'windows':WINDOWS,'model':'EASE','lambda':500.0,'interaction':'binary >30% historical single-fragment qualification',
      'fit_universe':'compact historically observed users/items only; no future-only IDs',
      'formula':'P=(X.T X+500I)^-1; B=-P/diag(P)[None,:]; diag(B)=0; score=X B',
      'solver':'float64 scipy Cholesky factor/solve','repeats':'retained; diagonal0 is reconstruction constraint, not a seen-item exclusion',
      'percentile':'full eligible observed item scores; top1 bottom0, numeric ID ties, singleitem=>1',
      'cold':'if either entity unseen, score/percentile0; flags independently identify user/item',
      'candidate_pool':'original context_features unchanged','labels':'forecast labels not loaded or used',
      'threads':2,'fits':3,'hyperparameter_grid':False,'submissions':0,'budget_seconds':600,'budget_bytes':2000000000,
      'fixture':test,'code_sha256':digest(Path(__file__)),
      'source_protocol':json.loads((BASE/'evidence/collaborative_inputs/protocol.json').read_text())}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    def guard():
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        if time.perf_counter()-began>600 or rss>2000000000:raise RuntimeError('EASE budget exceeded; incomplete staged output not promoted')
    results=[]
    for name,cut,end in WINDOWS:
        started=time.perf_counter();source=BASE/'cache/collaborative'
        cp=source/f'{name}_counts.npz';catp=source/f'{name}_catalog.npz';dp=BASE/f'cache/context_features/{name}.npz'
        counts=sparse.load_npz(cp).tocsr().astype(np.float64);counts.sum_duplicates();counts.eliminate_zeros()
        assert np.isfinite(counts.data).all() and np.all(counts.data>0)
        with np.load(catp,allow_pickle=False) as z:c={k:z[k] for k in z.files}
        with np.load(dp,allow_pickle=False) as z:d={k:z[k] for k in ('users','group_offsets','item_ids','row_item_indices')}
        assert counts.shape==(len(c['users']),len(c['items']))
        assert np.array_equal(c['users'][c['target_user_indices']],c['target_users']) and np.array_equal(d['users'],c['target_users'])
        assert np.array_equal(c['items'][c['candidate_item_indices']],c['candidate_ids'])
        assert c['candidate_ids'].tolist()==sorted(c['candidate_ids'].tolist(),key=int)
        active_u=np.flatnonzero(counts.getnnz(axis=1)>0);active_i=np.flatnonzero(counts.getnnz(axis=0)>0)
        umap=np.full(len(c['users']),-1,dtype=np.int32);umap[active_u]=np.arange(len(active_u))
        imap=np.full(len(c['items']),-1,dtype=np.int32);imap[active_i]=np.arange(len(active_i))
        binary=counts[active_u][:,active_i].tocsr();binary.data[:]=1
        guard();fit_start=time.perf_counter();b=ease(binary,500.);fit_seconds=time.perf_counter()-fit_start;guard()
        assert np.isfinite(b).all() and np.all(np.diag(b)==0)
        np.save(stage/f'{name}_B.npy',b)
        compact_target=umap[c['target_user_indices']];compact_candidates=imap[c['candidate_item_indices']]
        valid=np.flatnonzero(compact_candidates>=0);lookup={str(i):j for j,i in enumerate(c['candidate_ids'])}
        features=np.zeros((int(d['group_offsets'][-1]),4),dtype=np.float32)
        scores=np.lib.format.open_memmap(stage/f'{name}_scores.npy',mode='w+',dtype=np.float32,shape=(len(d['users']),len(c['candidate_ids'])));scores[:]=0
        for start in range(0,len(d['users']),128):
            stop=min(start+128,len(d['users']));warm=compact_target[start:stop]>=0
            block=np.zeros((stop-start,len(active_i)),dtype=np.float64)
            if warm.any():block[warm]=binary[compact_target[start:stop][warm]]@b
            for local,row in enumerate(range(start,stop)):
                a,z=d['group_offsets'][row:row+2];ui=compact_target[row]
                ids=d['item_ids'][d['row_item_indices'][a:z]];pos=np.array([lookup[str(i)] for i in ids]);ii=compact_candidates[pos]
                features[a:z,2]=ui<0;features[a:z,3]=ii<0
                if ui<0:continue
                s=block[local,compact_candidates[valid]];scores[row,valid]=s
                order=np.argsort(-s,kind='stable');rank=np.empty(len(valid),dtype=np.int32);rank[order]=np.arange(len(valid))
                pct=np.zeros(len(c['candidate_ids']),dtype=np.float64);pct[valid]=1-rank/max(1,len(valid)-1)
                known=ii>=0;dest=np.arange(a,z)[known];features[dest,0]=s[np.searchsorted(valid,pos[known])];features[dest,1]=pct[pos[known]]
            guard()
        assert np.isfinite(features).all();cold=(features[:,2]>0)|(features[:,3]>0);assert np.all(features[cold,:2]==0)
        scores.flush();del scores;np.save(stage/f'{name}_row_features.npy',features)
        np.savez(stage/f'{name}_catalog.npz',users=d['users'],candidate_ids=c['candidate_ids'],active_user_ids=c['users'][active_u],active_item_ids=c['items'][active_i],feature_names=np.asarray(NAMES))
        result={'window':name,'cutoff':cut,'forecast_end':end,'observed_users':len(active_u),'observed_items':len(active_i),'positive_pairs':binary.nnz,'rows':len(features),'cold_pair_rows':int(cold.sum()),'fit_seconds':fit_seconds,'wall_seconds':time.perf_counter()-started,
          'source_hashes':{str(p.relative_to(BASE)):digest(p) for p in (cp,catp,dp)},'feature_sha256':digest(stage/f'{name}_row_features.npy')}
        results.append(result);print(json.dumps(result),flush=True);del counts,binary,b,features
    stage.rename(cache)
    manifest={'status':'complete','features':NAMES,'results':results,'fits':3,'fixture':test,'wall_seconds':time.perf_counter()-began,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),'protocol_sha256':digest(out/'protocol.json'),'submissions':0}
    temp=out/'results.json.tmp';temp.write_text(json.dumps(manifest,indent=2)+'\n');temp.replace(out/'results.json')
if __name__=='__main__':
    with threadpool_limits(limits=2):main()
