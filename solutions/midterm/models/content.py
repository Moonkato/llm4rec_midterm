"""Historical-metadata TF-IDF profiles; no titles, item IDs or future metadata.

The caller must supply metadata available before its training cutoff. Scores are
cosine similarities, not probabilities. Candidate eligibility and the common
cold-start fallback belong to the calling evaluation pipeline.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

import numpy as np
from scipy import sparse


BLOCK_WEIGHTS = {"category": 1.0, "genre": 1.5, "director": 0.75,
                 "actors": 1.0, "year": 0.35}
_MISSING = {"", "0", "0000", "none", "null", "nan", "unknown", "неизвестно", "-"}


def _clean(value) -> str:
    if value is None:
        return ""
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    value = re.sub(r"\s+", " ", value).strip(" \t\n.\"'")
    return "" if value in _MISSING else value


def _names(value) -> set[str]:
    # Data samples separate full actor names by commas. Preserve full names:
    # splitting on whitespace would spuriously connect unrelated surnames.
    text = re.sub(r"\s+и\s+др\.?", "", _clean(value))
    return {name for part in re.split(r"[,;|]", text)
            if (name := _clean(part))}


def _tokens(attrs: dict) -> dict[str, set[str]]:
    category = _clean(attrs.get("tv_show_category"))
    genres = set()
    for i in (1, 2, 3):
        genres.update(_names(attrs.get(f"tv_show_genre_{i}")))
    years = {int(y) for y in re.findall(r"\b\d{4}\b", str(attrs.get("year_of_production", "")))
             if 1900 <= int(y) <= 2100}
    # Include range endpoints without expanding e.g. 1985-1992 into eight years.
    year_tokens = {f"year:{y}" for y in years} | {f"decade:{y // 10 * 10}" for y in years}
    return {"category": {category} if category else set(), "genre": genres,
            "director": _names(attrs.get("director")),
            "actors": _names(attrs.get("actors")), "year": year_tokens}


def _l2_rows(matrix):
    matrix = sparse.csr_matrix(matrix, dtype=np.float32)
    lengths = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    inv = np.zeros_like(lengths)
    np.divide(1.0, lengths, out=inv, where=lengths > 0)
    return matrix.multiply(inv[:, None]).tocsr()


def fit_content(item_ids: list[str], metadata: dict[str, dict], counts) -> dict:
    """Fit five weighted attribute blocks using historical metadata only.

    IDF = log((1 + historical_item_count)/(1 + document_frequency)) + 1.
    Each block is L2-normalized before weighting, then complete item vectors
    and log1p(count)-weighted user profiles are L2-normalized once more.
    Missing metadata yields a zero item vector; no synthetic identity tokens.
    """
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("item_ids must be unique")
    counts = sparse.csr_matrix(counts, dtype=np.float32, copy=True)
    counts.sum_duplicates()
    counts.eliminate_zeros()
    if counts.shape[1] != len(item_ids):
        raise ValueError("counts columns must match item_ids")
    if not np.isfinite(counts.data).all() or np.any(counts.data < 0):
        raise ValueError("counts must be finite and nonnegative")
    historical = {str(item): _tokens(attrs) for item, attrs in metadata.items()}
    n_documents = len(historical)
    blocks, vocabularies, block_sizes = [], {}, {}
    for block, weight in BLOCK_WEIGHTS.items():
        df = Counter(token for tokens in historical.values() for token in tokens[block])
        vocabulary = {token: j for j, token in enumerate(sorted(df))}
        vocabularies[block] = vocabulary
        block_sizes[block] = len(vocabulary)
        row_indices, columns, values = [], [], []
        for row, item in enumerate(item_ids):
            tokens = historical.get(str(item))
            if tokens is None:
                continue
            for token in sorted(tokens[block]):
                row_indices.append(row)
                columns.append(vocabulary[token])
                values.append(np.log((1 + n_documents) / (1 + df[token])) + 1)
        features = sparse.csr_matrix((np.asarray(values, dtype=np.float32),
                                      (row_indices, columns)),
                                     shape=(len(item_ids), len(vocabulary)))
        blocks.append(_l2_rows(features) * np.float32(weight))
    item_features = _l2_rows(sparse.hstack(blocks, format="csr", dtype=np.float32))
    counts.data = np.log1p(counts.data)
    profiles = _l2_rows(counts @ item_features)
    return {"item_features": item_features, "user_profiles": profiles,
            "feature_count": item_features.shape[1],
            "vocabularies": vocabularies,
            "config": {"method": "historical_attribute_tfidf_cosine",
                       "block_weights": dict(BLOCK_WEIGHTS),
                       "block_feature_counts": block_sizes,
                       "historical_metadata_items": n_documents,
                       "profile_weight": "log1p(event_count)",
                       "idf": "log((1+n_historical_items)/(1+df))+1",
                       "normalization": "L2 per block, item and user profile",
                       "repeat_items_allowed": True,
                       "missing_metadata": "zero vector; caller handles fallback",
                       "zero_profile": "all-zero scores; caller handles fallback"}}


def score_content(model: dict, user_indices: np.ndarray,
                  candidate_indices: np.ndarray) -> np.ndarray:
    """Return float32 cosine scores, users x candidates, in requested order."""
    users = np.asarray(user_indices, dtype=np.int64)
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    if users.ndim != 1 or candidates.ndim != 1:
        raise ValueError("user_indices and candidate_indices must be 1-D")
    if np.any(users < 0) or np.any(users >= model["user_profiles"].shape[0]):
        raise IndexError("user index outside fitted profiles")
    if np.any(candidates < 0) or np.any(candidates >= model["item_features"].shape[0]):
        raise IndexError("candidate index outside fitted item catalog")
    scores = (model["user_profiles"][users] @ model["item_features"][candidates].T).toarray()
    return np.asarray(scores, dtype=np.float32)
