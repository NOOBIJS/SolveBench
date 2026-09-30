# Row permutation to pick a better diagonal before Jacobi/Gauss-Seidel/SOR run.
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

OBJECTIVES = ("bottleneck", "minsum", "mc64", "best", "none")

# "best" tries all of these and keeps whichever gives the lowest estimated rho(T_GS).
PORTFOLIO = ("none", "mc64", "minsum", "bottleneck")

# above this size fall back to sparse matching; dense is fine up to here on Kaggle's RAM.
DENSE_ASSIGNMENT_CAP = 12000


def row_ratios(A):
    # Row ratio each entry would give if picked as that row's diagonal.
    M = abs(A).tocsr()
    rowsum = np.asarray(M.sum(axis=1)).ravel()
    R = M.copy().astype(np.float64)
    counts = np.diff(R.indptr)
    R.data = (np.repeat(rowsum, counts) - M.data) / M.data
    return R


def worst_row_ratio(A):
    # Worst row ratio for the diagonal A has now; below 1 = strict diagonal dominance.
    M = abs(A).tocsr()
    rowsum = np.asarray(M.sum(axis=1)).ravel()
    d = np.abs(A.diagonal())
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (rowsum - d) / d
    r[d == 0] = np.inf
    return float(np.max(r)) if r.size else np.inf


def _perm_from_matching(match_rows_to_cols, n):
    p = np.empty(n, dtype=int)
    p[match_rows_to_cols] = np.arange(n)
    return p


def bottleneck_permutation(A):
    # Minimise the worst row ratio, via binary search + bipartite matching.
    n = A.shape[0]
    if n > DENSE_ASSIGNMENT_CAP:
        return None, np.inf

    R = row_ratios(A)
    if R.nnz == 0:
        return None, np.inf

    candidates = np.unique(R.data)

    # MC64's own worst ratio upper-bounds the optimum, so anything above it can be dropped.
    mc_perm, _ = mc64_permutation(A)
    if mc_perm is not None:
        upper = worst_row_ratio(A[mc_perm, :])
        if np.isfinite(upper):
            kept = candidates[candidates <= upper]
            if kept.size:
                candidates = kept

    # cap the number of thresholds tried, quantiles beyond this many distinct values.
    if candidates.size > 1024:
        candidates = np.unique(np.quantile(candidates, np.linspace(0, 1, 1024)))

    dense = R.toarray()
    pattern = abs(A).toarray() > 0

    def feasible(threshold):
        # Perfect matching exists using only entries <= threshold?
        allowed = pattern & (dense <= threshold)
        cost = np.where(allowed, 0.0, 1.0)
        rows, cols = linear_sum_assignment(cost)
        if cost[rows, cols].sum() > 0:
            return False, None
        m = np.empty(n, dtype=int)
        m[rows] = cols
        return True, m

    lo, hi, best = 0, candidates.size - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        ok, m = feasible(candidates[mid])
        if ok:
            best = (np.asarray(m).copy(), candidates[mid])
            hi = mid - 1
        else:
            lo = mid + 1
    if best is None:
        return None, np.inf
    m, worst = best
    return _perm_from_matching(m, n), float(worst)


def minsum_permutation(A):
    # Minimise total row ratio -- turns out to be exactly MC64's own objective.
    n = A.shape[0]
    if n > DENSE_ASSIGNMENT_CAP:
        return None, np.inf

    R = row_ratios(A)
    if R.nnz == 0:
        return None, np.inf

    # log1p so one huge-ratio row can't swamp the sum and make this collapse to bottleneck.
    mask = abs(A).toarray() > 0
    return _dense_assignment(np.log1p(R.toarray()), mask, n)


def _dense_assignment(weights, mask, n):
    # Minimum-cost perfect assignment, dense (sparse matching hung on real inputs).
    W = np.full((n, n), np.inf)
    W[mask] = weights[mask]
    finite = W[np.isfinite(W)]
    if finite.size == 0:
        return None, np.inf
    penalty = (finite.max() + 1.0) * n + 1.0
    W[~np.isfinite(W)] = penalty
    rows, cols = linear_sum_assignment(W)
    if np.any(W[rows, cols] >= penalty):
        return None, np.inf
    return _perm_from_matching(cols[np.argsort(rows)], n), float(W[rows, cols].sum())


def mc64_permutation(A):
    # Baseline: maximise the product of |diagonal| entries, as MC64 does.
    n = A.shape[0]
    M = abs(A).tocsr().astype(np.float64)
    if M.nnz == 0:
        return None, np.inf

    if n <= DENSE_ASSIGNMENT_CAP:
        d = M.toarray()
        mask = d > 0
        with np.errstate(divide="ignore"):
            C = -np.log(d, out=np.full_like(d, -np.inf), where=mask)
        C[mask] -= C[mask].min() - 1.0
        return _dense_assignment(C, mask, n)

    return None, np.inf


def apply_permutation(A, b, p):
    # A' = PA, b' = Pb; the solution x is unchanged.
    return A[p, :].tocsr(), b[p]


def select_diagonal(A, b=None, objective="bottleneck"):
    # Choose the diagonal, returning (A', b', info). objective="none" is the control.
    n = A.shape[0]
    info = {"objective": objective, "permuted": False, "cost": np.nan,
            "zero_diagonal_before": int((np.abs(A.diagonal()) < 1e-14).sum()),
            "worst_ratio_before": worst_row_ratio(A)}

    if objective == "none":
        return A, b, info

    if objective == "best":
        return _select_best(A, b, info)

    solver = {"bottleneck": bottleneck_permutation,
              "minsum": minsum_permutation,
              "mc64": mc64_permutation}[objective]
    p, cost = solver(A)
    if p is None:
        info["note"] = "no perfect matching: no permutation can fill the diagonal"
        return A, b, info

    A2 = A[p, :].tocsr()
    b2 = None if b is None else b[p]
    info.update(permuted=not np.array_equal(p, np.arange(n)), cost=cost,
                zero_diagonal_after=int((np.abs(A2.diagonal()) < 1e-14).sum()),
                worst_ratio_after=worst_row_ratio(A2))
    return A2, b2, info


def _estimate_rho_gs(A, iters=40):
    # Cheap power-iteration estimate of rho(T_GS), for ranking candidates only.
    from . import spectral
    op = spectral.gauss_seidel_operator(A)
    if op is None:
        return np.inf
    v = np.random.default_rng(0).standard_normal(A.shape[0])
    nv = np.linalg.norm(v)
    if nv == 0:
        return np.inf
    v /= nv
    rho = np.inf
    for _ in range(iters):
        w = op @ v
        nw = np.linalg.norm(w)
        if not np.isfinite(nw):
            return np.inf
        if nw == 0.0:
            return 0.0
        rho, v = nw, w / nw
    return float(rho)


def _select_best(A, b, info):
    # Try every candidate, keep whichever gives the smallest estimated rho(T_GS).
    best = (np.inf, None, None, "none")
    tried, ratios = {}, {}
    for name in PORTFOLIO:
        if name == "none":
            cand = A
        else:
            p, _ = {"mc64": mc64_permutation, "minsum": minsum_permutation,
                    "bottleneck": bottleneck_permutation}[name](A)
            if p is None:
                continue
            cand = A[p, :].tocsr()
        ratios[name] = worst_row_ratio(cand)
        if np.any(np.abs(cand.diagonal()) < 1e-14):
            tried[name] = np.inf
            continue
        rho = _estimate_rho_gs(cand)
        tried[name] = rho
        if rho < best[0]:
            best = (rho, cand, None if name == "none" else p, name)

    rho, cand, p, name = best
    info.update(chosen=name, estimated_rho_gs=rho, candidates=tried, ratios=ratios,
                permuted=name != "none")
    if cand is None:
        info["note"] = "no candidate produced a usable diagonal"
        return A, b, info
    info["zero_diagonal_after"] = 0
    info["worst_ratio_after"] = worst_row_ratio(cand)
    return cand, (b if p is None or b is None else b[p]), info


def make_solver(base_solver, objective="bottleneck"):
    # Wrap a stationary solver so it selects its diagonal first.
    def solve(A, b):
        t0 = time.perf_counter()
        A2, b2, _ = select_diagonal(A, b, objective=objective)
        setup = time.perf_counter() - t0
        x, iters, converged, work = base_solver(A2, b2)
        # count the permutation as this method's own setup cost, not a free head start.
        if work is not None and hasattr(work, "setup"):
            work.setup += setup
        return x, iters, converged, work
    solve.__name__ = f"{base_solver.__name__}_{objective}"
    return solve
