"""Three explicitly authorized WARP identity fits; fixed-row scalar features.

Preparation/inspection is safe without importing LightFM. Execution requires
--run and a separately installed LightFM; this script never installs anything.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='2'
import argparse
import hashlib
import importlib.metadata
import json
import pickle
import resource
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
from scipy import sparse

BASE=Path(__file__).resolve().parents[1]
NAMES=['warp_score','warp_score_percentile','warp_item_bias','warp_user_bias',
       'warp_item_norm','warp_user_norm','warp_user_unobserved','warp_item_unobserved']
WINDOWS=[('fit','2020-04-13','2020-05-18'),('eval','2020-05-18','2020-07-27'),('later','2020-06-22','2020-07-27')]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true')
    if not parser.parse_args().run:
        raise SystemExit('Prepared only. Explicit root-approved --run required; no fitting occurred.')
    # Missing package stops before creating outputs or importing network tools.
    from lightfm import LightFM
    began=time.perf_counter();cache=BASE/'cache/warp_features';out=BASE/'evidence/warp_features'
    if cache.exists() or (out/'results.json').exists():
        raise SystemExit('Preserve existing WARP outputs; explicit cleanup/recovery required.')
    out.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.warp_features-',dir=BASE/'cache'))
    protocol={'status':'running','features':NAMES,'windows':WINDOWS,'history_origin':'2020-03-09',
      'interactions':'binary positives from historical MAX clipped raw overlap>30% wholeairing duration',
      'model':'LightFM identity-only WARP','parameters':{'no_components':16,'loss':'warp','epochs':10,'max_sampled':10,'random_state':20261023,'num_threads':2},
      'training_universe':'Only historically observed users and items; remove zero rows/columns before fitting so future-only IDs never enter negative sampling.',
      'features_definition':{'warp_score':'dot(user,item)+item_bias+user_bias',
        'warp_score_percentile':'1-(rank-1)/(N_observed_future_items-1), rank1best numeric IDties; single item=>1; cold=>0',
        'bias_norms':'learned scalar biases and L2 norms; no latent coordinate features',
        'cold':'If either entity historically unobserved, all first6features zero; flags independently identify unobserved user/item'},
      'repeats':'allowed, no history exclusion','candidate_pool':'unchanged original context_features rows',
      'labels':'No forecast labels used for WARP fitting or features; context y not loaded',
      'budgets':{'wall_seconds':1200,'peak_rss_bytes':2000000000},'fits':3,'submissions':0,
      'reproducibility':'Fixed seed; two-thread asynchronous updates may not be bitwise deterministic.',
      'source_preprocessing':json.loads((BASE/'evidence/collaborative_inputs/protocol.json').read_text()),
      'lightfm_version':importlib.metadata.version('lightfm'),'code_sha256':digest(Path(__file__))}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    def guard():
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        if time.perf_counter()-began>1200 or rss>2000000000:
            raise RuntimeError('WARP time/memory budget reached; staged files not promoted as complete.')
    results=[]
    for name,cut,end in WINDOWS:
        started=time.perf_counter();source=BASE/'cache/collaborative'
        count_path=source/f'{name}_counts.npz';catalog_path=source/f'{name}_catalog.npz';context_path=BASE/f'cache/context_features/{name}.npz'
        counts=sparse.load_npz(count_path).tocsr().astype(np.float32);counts.sum_duplicates();counts.eliminate_zeros()
        assert np.isfinite(counts.data).all() and np.all(counts.data>0)
        with np.load(catalog_path,allow_pickle=False) as z:c={k:z[k] for k in z.files}
        with np.load(context_path,allow_pickle=False) as z:d={k:z[k] for k in ('users','item_ids','row_item_indices','group_offsets')}
        assert counts.shape==(len(c['users']),len(c['items']))
        assert np.array_equal(c['users'][c['target_user_indices']],c['target_users'])
        assert np.array_equal(d['users'],c['target_users'])
        assert np.array_equal(c['items'][c['candidate_item_indices']],c['candidate_ids'])
        assert c['candidate_ids'].tolist()==sorted(c['candidate_ids'].tolist(),key=int)
        observed_u=np.asarray(counts.getnnz(axis=1))>0;observed_i=np.asarray(counts.getnnz(axis=0))>0
        active_u=np.flatnonzero(observed_u);active_i=np.flatnonzero(observed_i)
        umap=np.full(len(observed_u),-1,dtype=np.int32);umap[active_u]=np.arange(len(active_u))
        imap=np.full(len(observed_i),-1,dtype=np.int32);imap[active_i]=np.arange(len(active_i))
        binary=counts[active_u][:,active_i].tocoo();binary.data[:]=1
        assert binary.nnz>0 and np.all(binary.data==1)
        model=LightFM(no_components=16,loss='warp',max_sampled=10,random_state=20261023)
        fit_start=time.perf_counter()
        for epoch in range(10):
            guard();model.fit_partial(binary,epochs=1,num_threads=2);guard()
            print(json.dumps({'window':name,'epoch':epoch+1,'fit_seconds':time.perf_counter()-fit_start}),flush=True)
        fit_seconds=time.perf_counter()-fit_start
        ub,uf=model.get_user_representations();ib,itf=model.get_item_representations()
        assert all(np.isfinite(x).all() for x in (ub,uf,ib,itf))
        un=np.linalg.norm(uf,axis=1);itn=np.linalg.norm(itf,axis=1)
        target_compact=umap[c['target_user_indices']];candidate_compact=imap[c['candidate_item_indices']]
        valid=np.flatnonzero(candidate_compact>=0);valid_compact=candidate_compact[valid]
        lookup={str(item):j for j,item in enumerate(c['candidate_ids'])}
        features=np.zeros((int(d['group_offsets'][-1]),8),dtype=np.float32)
        scores=np.lib.format.open_memmap(stage/f'{name}_scores.npy',mode='w+',dtype=np.float32,shape=(len(d['users']),len(c['candidate_ids'])))
        scores[:]=0
        for row,uid in enumerate(d['users']):
            a,b=d['group_offsets'][row:row+2];ui=target_compact[row]
            itemids=d['item_ids'][d['row_item_indices'][a:b]]
            candidate_positions=np.array([lookup[str(i)] for i in itemids],dtype=np.int32)
            ii=candidate_compact[candidate_positions];features[a:b,6]=ui<0;features[a:b,7]=ii<0
            if ui<0:continue
            s=itf[valid_compact]@uf[ui]+ib[valid_compact]+ub[ui]
            scores[row,valid]=s
            # Full eligible observed-item order, not shortlist-only rank.
            order=np.argsort(-s,kind='stable');ranks=np.empty(len(valid),dtype=np.int32);ranks[order]=np.arange(len(valid))
            pct=np.zeros(len(c['candidate_ids']),dtype=np.float32)
            pct[valid]=1-ranks/max(1,len(valid)-1)
            known=ii>=0;dest=np.arange(a,b)[known];known_i=ii[known]
            features[dest,0]=scores[row,candidate_positions[known]];features[dest,1]=pct[candidate_positions[known]]
            features[dest,2]=ib[known_i];features[dest,3]=ub[ui]
            features[dest,4]=itn[known_i];features[dest,5]=un[ui]
        assert np.isfinite(features).all();scores.flush();del scores
        cold=(features[:,6]>0)|(features[:,7]>0);assert np.all(features[cold,:6]==0)
        np.save(stage/f'{name}_row_features.npy',features)
        np.savez(stage/f'{name}_catalog.npz',users=d['users'],candidate_ids=c['candidate_ids'],
                 active_user_ids=c['users'][active_u],active_item_ids=c['items'][active_i],feature_names=np.asarray(NAMES))
        with (stage/f'{name}_model.pkl').open('wb') as f:pickle.dump(model,f)
        result={'window':name,'cutoff':cut,'forecast_end':end,'fit_users':len(active_u),'fit_items':len(active_i),'positive_pairs':binary.nnz,
          'rows':len(features),'cold_pair_rows':int(cold.sum()),'fit_seconds':fit_seconds,'wall_seconds':time.perf_counter()-started,
          'input_hashes':{str(p.relative_to(BASE)):digest(p) for p in (count_path,catalog_path,context_path)},
          'feature_sha256':digest(stage/f'{name}_row_features.npy')}
        results.append(result);print(json.dumps(result),flush=True);guard()
        del counts,binary,model,features
    # One atomic directory rename makes all three complete windows visible together.
    stage.rename(cache)
    manifest={'status':'complete','features':NAMES,'results':results,'wall_seconds':time.perf_counter()-began,
       'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
       'protocol_sha256':digest(out/'protocol.json'),'fits':3,'submissions':0}
    temp=out/'results.json.tmp';temp.write_text(json.dumps(manifest,indent=2)+'\n');temp.replace(out/'results.json')


if __name__=='__main__':main()
