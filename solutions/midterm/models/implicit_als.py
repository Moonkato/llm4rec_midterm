"""Implicit-feedback ALS, fitted exclusively to a supplied historical count matrix.

For user factors X and item factors Y, minimize
    sum_ui c_ui * (p_ui - X[u] @ Y[i])**2
    + regularization * (sum(X**2) + sum(Y**2)),
where p_ui=1 for positive counts, otherwise 0, and
c_ui=1 + alpha*log1p(count_ui). Missing pairs have confidence 1.

Normal equations are solved exactly for each alternating block, using float64
arithmetic. Returned factors are float32. No validation labels or future data
are accepted. Fitting never constructs a dense user-by-item matrix.
"""

import time

import numpy as np
from scipy import sparse


def _counts_csr(counts):
    matrix = sparse.csr_matrix(counts, dtype=np.float64, copy=True)
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError("counts must be finite and nonnegative")
    matrix.sum_duplicates()
    if not np.isfinite(matrix.data).all():
        raise ValueError("summed counts must be finite")
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def _solve_block(confidence_extra, fixed, regularization):
    """Solve rows against fixed opposite-side factors; extra stores c-1."""
    width = fixed.shape[1]
    shared = fixed.T @ fixed + regularization * np.eye(width)
    solved = np.zeros((confidence_extra.shape[0], width), dtype=np.float64)
    for row in range(confidence_extra.shape[0]):
        lo, hi = confidence_extra.indptr[row:row + 2]
        if lo == hi:
            continue
        other = fixed[confidence_extra.indices[lo:hi]]
        extra = confidence_extra.data[lo:hi]
        lhs = shared + other.T @ (extra[:, None] * other)
        rhs = other.T @ (1.0 + extra)
        solved[row] = np.linalg.solve(lhs, rhs)
    return solved


def fit_als(counts, factors=32, regularization=1.0, alpha=10.0,
            iterations=10, seed=20261022, progress=None):
    """Fit user×item historical counts and return factors plus configuration.

    ``progress``, if supplied, receives one dict after each complete user/item
    update: iteration (one-based), iterations, elapsed_seconds. Rows/columns
    with no observations have zero factors. Positive regularization ensures
    nonsingular normal equations even for empty or rank-deficient data.
    ``alpha=0`` is supported: observed preferences still have confidence one.
    """
    if not isinstance(factors, (int, np.integer)) or factors < 1:
        raise ValueError("factors must be a positive integer")
    if not isinstance(iterations, (int, np.integer)) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be finite and positive")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    matrix = _counts_csr(counts)
    extra = matrix.copy()
    extra.data = alpha * np.log1p(extra.data)
    if not np.isfinite(extra.data).all():
        raise ValueError("confidence weights overflowed")
    # Preserve explicit zero extra weights when alpha=0: indices encode p=1.
    transposed = extra.T.tocsr()
    n_users, n_items = matrix.shape
    rng = np.random.default_rng(seed)
    item_factors = rng.normal(0.0, 0.01, (n_items, factors))
    item_factors[np.diff(transposed.indptr) == 0] = 0.0
    began = time.perf_counter()
    for iteration in range(iterations):
        user_factors = _solve_block(extra, item_factors, regularization)
        item_factors = _solve_block(transposed, user_factors, regularization)
        if progress is not None:
            progress({"iteration": iteration + 1, "iterations": int(iterations),
                      "elapsed_seconds": time.perf_counter() - began})
    return {
        "user_factors": user_factors.astype(np.float32),
        "item_factors": item_factors.astype(np.float32),
        "config": {
            "method": "implicit_als", "factors": int(factors),
            "regularization": float(regularization), "alpha": float(alpha),
            "iterations": int(iterations), "seed": int(seed),
            "confidence": "1 + alpha * log1p(count)",
            "preference": "count > 0", "regularization_scaling": "constant",
            "n_users": n_users, "n_items": n_items, "nnz": matrix.nnz,
            "fit_seconds": time.perf_counter() - began,
        },
    }


def _indices(values, length, name):
    result = np.asarray(values)
    if result.ndim != 1:
        raise ValueError(name + " must be a one-dimensional index sequence")
    if result.size == 0:
        return np.empty(0, dtype=np.intp)
    if not np.issubdtype(result.dtype, np.integer):
        raise ValueError(name + " must contain integers")
    if (result < 0).any() or (result >= length).any():
        raise IndexError(name + " contains an out-of-range index")
    return result.astype(np.intp, copy=False)


def score_als(model, user_indices, candidate_indices):
    """Return dense scores with shape (requested users, requested candidates).

    Preserves supplied order; does not filter history, add fallback, or rank.
    Call in user batches to bound the returned dense array's memory use.
    """
    users = model["user_factors"]
    items = model["item_factors"]
    ui = _indices(user_indices, users.shape[0], "user_indices")
    ii = _indices(candidate_indices, items.shape[0], "candidate_indices")
    return users[ui] @ items[ii].T


def objective_als(counts, model):
    """Evaluate the stated objective without a dense user×item array.

    Intended for small numerical verification fixtures, not iteration logging.
    Accounts for all missing pairs via factor Gram matrices and corrects the
    observed pairs explicitly. Slight roundoff is expected for float32 output.
    """
    matrix = _counts_csr(counts)
    users = np.asarray(model["user_factors"], dtype=np.float64)
    items = np.asarray(model["item_factors"], dtype=np.float64)
    if matrix.shape != (users.shape[0], items.shape[0]):
        raise ValueError("counts shape does not match model")
    alpha = model["config"]["alpha"]
    reg = model["config"]["regularization"]
    value = float(np.sum((users.T @ users) * (items.T @ items)))
    for u in range(matrix.shape[0]):
        lo, hi = matrix.indptr[u:u + 2]
        predictions = items[matrix.indices[lo:hi]] @ users[u]
        confidence = 1.0 + alpha * np.log1p(matrix.data[lo:hi])
        value += float(np.sum(confidence * (1.0 - predictions)**2 - predictions**2))
    value += reg * (np.sum(users**2) + np.sum(items**2))
    return float(value)
