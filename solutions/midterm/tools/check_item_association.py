"""Memory-bounded item co-view association; historical counts only."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from run_round1 import apk

BASE = Path(__file__).resolve().parents[1]


def association_graph(counts, numeric_ids):
    binary = counts.copy().astype(np.float32)
    binary.data.fill(1)
    binary = binary.tocsc()
    freq = np.asarray(binary.sum(axis=0)).ravel()
    n = binary.shape[1]
    # Never form full C=B.T@B. Upper-bound one dense CSR block at 2M cells.
    block = min(128, max(1, 2_000_000 // max(n, 1)))
    if n > 100_000 or block * n * 12 > 64_000_000:
        raise MemoryError('Item universe exceeds bounded block policy.')
    parts_i, parts_s, indptr = [], [], [0]
    for start in range(0, n, block):
        stop = min(start + block, n)
        product = (binary[:, start:stop].T @ binary).tocsr()
        for local, source in enumerate(range(start, stop)):
            a, b = product.indptr[local:local + 2]
            indices, co = product.indices[a:b], product.data[a:b]
            keep = indices != source
            indices, co = indices[keep], co[keep]
            values = co / np.sqrt(freq[source] * freq[indices]) * co / (co + 20)
            order = np.lexsort((numeric_ids[indices], -values))[:100]
            parts_i.append(indices[order].astype(np.int32))
            parts_s.append(values[order].astype(np.float32))
            indptr.append(indptr[-1] + len(order))
    graph = sparse.csr_matrix((np.concatenate(parts_s), np.concatenate(parts_i),
                               np.asarray(indptr, dtype=np.int32)), shape=(n, n))
    graph.sort_indices()
    assert not np.any(graph.diagonal()) and np.isfinite(graph.data).all()
    return graph, freq, block


def main():
    began = time.perf_counter()
    out, cache = BASE / 'evidence/item_association', BASE / 'cache/item_association'
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    if (out / 'results.json').exists():
        raise SystemExit('Preserve completed association experiment.')
    assert (BASE / 'evidence/collaborative_inputs/results.json').exists()
    protocol = {
        'windows': ['fit', 'eval', 'later'], 'history': 'shared >30% single-fragment counts; cutoff-specific',
        'binary': 'B_ui=1[count_ui>0]',
        'similarity': 'C_ij/sqrt(freq_i*freq_j) * C_ij/(C_ij+20), C=B.T@B, diagonal zero',
        'neighbors_per_source': 100, 'neighbor_ties': 'ascending numeric item ID',
        'score': '(log1p(counts) / row_sum(log1p(counts))) @ sparse_neighbor_graph',
        'repeats': 'allowed', 'fallback': 'same historical >30% distinct-viewer popularity for cold users and empty score slots',
        'score_output': 'dense float32 target_users x candidate_ids in corresponding collaborative catalog order; zero means no association signal',
        'selection': 'one fixed scorer, no tuning; eval/later reuse explored windows',
        'fit_scores_caveat': 'historical associative features, not fitted on future labels; fitting-window y is never used',
        'memory': 'co-view products in at most128-source blocks, at most2million cells/block; full score arrays written using memmap',
        'submissions': 0,
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    results = []
    for name in protocol['windows']:
        started = time.perf_counter()
        source = BASE / 'cache/collaborative'
        counts = sparse.load_npz(source / f'{name}_counts.npz').astype(np.float32)
        counts.sum_duplicates(); counts.eliminate_zeros()
        assert np.isfinite(counts.data).all() and (counts.data > 0).all()
        with np.load(source / f'{name}_catalog.npz', allow_pickle=False) as archive:
            cat = {k: archive[k] for k in archive.files}
        assert counts.shape == (len(cat['users']), len(cat['items']))
        graph, freq, block = association_graph(counts, cat['items'].astype(np.int64))
        sparse.save_npz(cache / f'{name}_graph.npz', graph)
        target = cat['target_user_indices']; candidate = cat['candidate_item_indices']
        names = cat['candidate_ids']; numeric = names.astype(np.int64)
        assert np.array_equal(cat['users'][target], cat['target_users'])
        assert np.array_equal(cat['items'][candidate], names)
        profiles = counts[target].copy()
        profiles.data = np.log1p(profiles.data)
        mass = np.asarray(profiles.sum(axis=1)).ravel()
        inverse = np.zeros_like(mass)
        np.divide(1, mass, out=inverse, where=mass > 0)
        profiles = profiles.multiply(inverse[:, None]).tocsr()
        fallback = np.lexsort((numeric, -freq[candidate]))
        scores = np.lib.format.open_memmap(cache / f'{name}_scores.npy', mode='w+', dtype=np.float32,
                                          shape=(len(target), len(candidate)))
        truth = json.loads((BASE / f'evidence/context_features/{name}_truth.json').read_text()) if name != 'fit' else None
        predictions, ap, fallback_users = {}, [], 0
        for lo in range(0, len(target), 128):
            hi = min(lo + 128, len(target))
            dense = (profiles[lo:hi] @ graph[:, candidate]).toarray()
            assert np.isfinite(dense).all()
            scores[lo:hi] = dense
            if truth is None:
                continue
            for offset, user in enumerate(cat['target_users'][lo:hi]):
                values = dense[offset]
                positive = np.flatnonzero(values > 0)
                top = positive[np.lexsort((numeric[positive], -values[positive]))[:5]].tolist()
                if len(top) < 5:
                    fallback_users += 1
                    for j in fallback:
                        if len(top) == 5:
                            break
                        if int(j) not in top:
                            top.append(int(j))
                recs = [str(names[j]) for j in top]
                assert len(recs) == len(set(recs)) == 5
                predictions[str(user)] = recs
                ap.append(apk(truth[str(user)], recs))
        scores.flush()
        del scores
        result = {'window': name, 'shape': list(counts.shape), 'graph_nnz': graph.nnz,
                  'source_block_size': block, 'target_users': len(target), 'candidates': len(candidate),
                  'cold_users': int((mass == 0).sum()), 'wall_seconds': time.perf_counter() - started,
                  'scores_file': str((cache / f'{name}_scores.npy').relative_to(BASE)),
                  'catalog_sha256': hashlib.sha256((source / f'{name}_catalog.npz').read_bytes()).hexdigest(),
                  'counts_sha256': hashlib.sha256((source / f'{name}_counts.npz').read_bytes()).hexdigest()}
        if truth is not None:
            assert len(ap) == 1260
            result.update(map5=float(np.mean(ap)), fallback_users=fallback_users,
                          empty_truth_users=sum(not truth[str(u)] for u in cat['target_users']))
            (out / f'{name}_predictions.json').write_text(json.dumps(predictions))
        results.append(result)
        (out / 'results.partial.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(result), flush=True)
    report = {'status': 'complete', 'results': results, 'wall_seconds': time.perf_counter() - began,
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'submissions': 0}
    (out / 'results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'status': 'complete', 'wall_seconds': report['wall_seconds'],
                      'peak_rss_bytes': report['peak_rss_bytes']}), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
