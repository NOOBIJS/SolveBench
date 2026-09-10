"""Convergence-oriented diagonal selection.

A stationary method divides by ``a_ii``, so its behaviour depends entirely on which
entries end up on the diagonal -- and that is decided by the order the rows happened to
arrive in, which came from a mesh numbering or a netlist, not from any thought about
convergence. This module treats that as a choice to be optimised.

**Why permutation is the only lever.** For the Jacobi iteration matrix
``T_J = I - D^-1 A``:

* Row scaling ``A' = RA`` gives ``D' = RD`` and ``T_J' = I - (RD)^-1 RA = T_J``. The
  spectrum is *unchanged*, exactly.
* Column scaling ``A' = AC`` gives ``T_J' = C^-1 (I - D^-1 A) C``, a similarity
  transform. The spectrum is unchanged again.
* A symmetric permutation ``PAP^T`` leaves the spectrum of ``T_J`` invariant, and on
  real matrices moves ``rho(T_GS)`` only in the fifth decimal (measured with both RCM
  and random orderings).

So scaling and symmetric reordering cannot help. Only a **row permutation** can, because
only it changes which entries make up ``D``.

**Why not just use MC64.** MC64 (Duff & Koster) also produces a row permutation, but it
maximises the *product* of the diagonal magnitudes -- an objective designed for pivot
stability in a direct factorization. Stationary convergence needs something else: a small
row ratio ``sum_{j != i} |a_ij| / |a_ii|``, since all such ratios below 1 means strict
diagonal dominance and guarantees convergence. Optimising the right objective gives a
different permutation and a different outcome: on ``odepa400`` MC64 leaves the matrix
untouched at rho(T_GS) = 1.0001, while the dominance objective reaches 0.495 and
Gauss-Seidel converges.

**The solution is untouched.** ``A' x = PAx = Pb = b'``, so a row permutation changes the
splitting without changing what solves the system. Nothing needs un-permuting afterwards.

Everything here works on the sparse pattern: the ratio matrix has exactly the nonzeros of
A, so cost is O(nnz), not O(n^2).
"""
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

OBJECTIVES = ("bottleneck", "minsum", "mc64", "best", "none")

#: Candidates the "best" objective chooses among. No single one dominates: on
#: ``odepa400`` bottleneck reaches rho(T_GS) = 0.495 where min-sum gives 1.265 and MC64
#: leaves it at 1.0001; on ``d_ss`` bottleneck is the worst of the three at 29.6 against
#: MC64's 1.83. Since each permutation costs milliseconds, the honest method computes
#: all of them and picks by direct spectral estimate.
PORTFOLIO = ("none", "mc64", "minsum", "bottleneck")

#: Every assignment is solved densely up to this size. Set above the corpus maximum of
#: n = 10,000 deliberately, because the earlier cap of 5,000 was the bug rather than the
#: safeguard: it left the sparse routines in play for the largest matrices, and rw5151
#: (n = 5,151, 151 past the cap) then hung the bottleneck search.
#:
#: Measured cost of the dense route at n = 10,000, an 800 MB array:
#:   min-cost assignment (MC64, min-sum)   7.9 s, once per matrix
#:   0/1 feasibility test (bottleneck)     0.5 s, about ten times per matrix
#: Kaggle offers roughly 30 GB, so the array is affordable and the time is not the
#: bottleneck. Past this size the sparse fallback returns, with the risk that implies.
DENSE_ASSIGNMENT_CAP = 12000


def row_ratios(A):
    """Sparse R with A's pattern: ``R[i,j]`` is what row i's Jacobi ratio becomes if
    ``a_ij`` is chosen as its diagonal, i.e. ``(sum_k |a_ik| - |a_ij|) / |a_ij|``.

    A value below 1 means that row would be diagonally dominant under this choice.
    """
    M = abs(A).tocsr()
    rowsum = np.asarray(M.sum(axis=1)).ravel()
    R = M.copy().astype(np.float64)
    counts = np.diff(R.indptr)
    R.data = (np.repeat(rowsum, counts) - M.data) / M.data
    return R


def worst_row_ratio(A):
    """``max_i (sum_{j != i} |a_ij|) / |a_ii|`` for the diagonal ``A`` currently has.

    Below 1 is strict diagonal dominance, which guarantees BOTH Jacobi and Gauss-Seidel
    converge -- so this one number says whether a permutation bought a convergence
    guarantee or merely a better-looking diagonal. Infinite when any diagonal entry is
    zero, which is the case the stationary methods are undefined on.

    O(nnz), read from the stored diagonal directly. It must NOT go through
    ``row_ratios(A).diagonal()``: a structurally absent a_ii has no entry there, so that
    route reports a ratio of 0 -- perfect dominance -- for exactly the matrices that have
    no usable diagonal at all.
    """
    M = abs(A).tocsr()
    rowsum = np.asarray(M.sum(axis=1)).ravel()
    d = np.abs(A.diagonal())
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (rowsum - d) / d
    r[d == 0] = np.inf
    return float(np.max(r)) if r.size else np.inf


def _perm_from_matching(match_rows_to_cols, n):
    """Row r assigned column c must sit at position c, so ``p[c] = r`` and ``A[p, :]``
    puts the chosen entry on the diagonal."""
    p = np.empty(n, dtype=int)
    p[match_rows_to_cols] = np.arange(n)
    return p


def bottleneck_permutation(A):
    """Minimise the WORST row ratio: ``min_pi max_i R[i, pi(i)]``.

    Bottleneck assignment, solved by binary search on the threshold -- keep only edges
    with ratio <= t and ask whether a perfect matching still exists. Returns
    ``(perm, worst_ratio)``, or ``(None, inf)`` if no perfect matching exists at all
    (then no permutation can give a nonzero diagonal).
    """
    n = A.shape[0]
    if n > DENSE_ASSIGNMENT_CAP:
        return None, np.inf          # see DENSE_ASSIGNMENT_CAP; the portfolio copes

    R = row_ratios(A)
    if R.nnz == 0:
        return None, np.inf

    candidates = np.unique(R.data)

    # Two prunings, both needed at scale. Each binary-search step runs a matching over
    # the whole pattern, so on TSC_OPF_1047 (n = 8,140, nnz = 2.0M) the naive search
    # over every distinct ratio took 103 seconds.
    #
    # First: MC64 is cheap and always yields a valid permutation, so the worst ratio it
    # achieves is an upper bound on the optimum. Everything above it can be discarded
    # before the search starts.
    mc_perm, _ = mc64_permutation(A)
    if mc_perm is not None:
        upper = worst_row_ratio(A[mc_perm, :])
        if np.isfinite(upper):
            kept = candidates[candidates <= upper]
            if kept.size:
                candidates = kept

    # Second: cap the search to a bounded number of steps. Above this many distinct
    # ratios the thresholds are taken as quantiles, so the result is the optimum to
    # within one quantile rather than exactly -- a permutation good enough to rank,
    # which is all the selector needs.
    if candidates.size > 1024:
        candidates = np.unique(np.quantile(candidates, np.linspace(0, 1, 1024)))

    dense = R.toarray()
    pattern = abs(A).toarray() > 0

    def feasible(threshold):
        """Does a perfect matching exist using only entries at or below `threshold`?

        Always dense. maximum_bipartite_matching was the third scipy routine in this
        module to misbehave on real inputs: on bcsstk19 -- n = 817, 6,853 nonzeros -- a
        single call took 7.2 seconds on a 3,764-edge subgraph, and the search makes ten
        such calls. It is slowest precisely when a matching does exist, which is the case
        the search spends most of its time in. The dense form answers the same question
        by assignment with 0/1 costs: a total of zero means every row found an allowed
        column.
        """
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
    """Minimise the TOTAL of the chosen row ratios: ``min_pi sum_i R[i, pi(i)]``.

    **This is MC64 under another name.** Since ``1 + R[i,j] = rowsum_i / |a_ij|``,

        sum_i log(1 + R[i,p(i)])  =  sum_i log(rowsum_i)  -  sum_i log|a_{i,p(i)}|

    and the first term does not depend on the permutation, so minimising the left side is
    exactly maximising ``sum log|a_ii|`` -- MC64's objective. Verified on 193 of 193
    random matrices, both reaching an identical objective value.

    It is kept because the full run reports it separately, and because the equivalence is
    worth stating: it was introduced as "the natural contrast to bottleneck" and is not a
    contrast at all. The portfolio has two distinct objectives, not three.
    """
    n = A.shape[0]
    if n > DENSE_ASSIGNMENT_CAP:
        return None, np.inf          # see DENSE_ASSIGNMENT_CAP; the portfolio copes

    R = row_ratios(A)
    if R.nnz == 0:
        return None, np.inf

    # Minimise sum(log(1 + ratio)), i.e. the PRODUCT of (1 + ratio) rather than the raw
    # sum. A raw sum is dominated by whichever single row has the largest ratio -- on
    # nnc261 those span 0 to 3.8e10 -- which would make "min-sum" nearly the bottleneck
    # objective it exists to contrast with.
    mask = abs(A).toarray() > 0
    return _dense_assignment(np.log1p(R.toarray()), mask, n)


def _dense_assignment(weights, mask, n):
    """Solve a minimum-cost perfect assignment densely.

    ``weights`` holds the cost of every real edge and ``mask`` says which entries are
    real. Absent edges get a penalty larger than any complete assignment of real ones, so
    a chosen penalty edge means no perfect matching exists over the real entries.

    Dense rather than ``min_weight_full_bipartite_matching`` because that routine is not
    dependable here. It hung outright on nnc261 under raw ratios and on west0067 -- 67x67,
    294 nonzeros -- under a log transform, and merely crawled elsewhere: 9.7 seconds on
    oscil_dcop_23 at n = 430, which the dense solver finishes in 6 milliseconds, a factor
    of 1,600. One of those hangs cost a twelve-hour Kaggle session that completed 30
    matrices of 930. A hang cannot be interrupted from Python, so the fix has to be
    avoiding the routine rather than detecting the hang.
    """
    W = np.full((n, n), np.inf)
    W[mask] = weights[mask]
    finite = W[np.isfinite(W)]
    if finite.size == 0:
        return None, np.inf
    penalty = (finite.max() + 1.0) * n + 1.0
    W[~np.isfinite(W)] = penalty
    rows, cols = linear_sum_assignment(W)
    if np.any(W[rows, cols] >= penalty):
        return None, np.inf          # no perfect matching over the real entries
    return _perm_from_matching(cols[np.argsort(rows)], n), float(W[rows, cols].sum())


def mc64_permutation(A):
    """Baseline: maximise the product of |diagonal| entries, as MC64 does.

    Equivalent to minimising ``sum -log|a_ij|``. This is the established tool, built for
    pivot stability rather than for convergence, and it is the comparison the method has
    to beat.
    """
    n = A.shape[0]
    M = abs(A).tocsr().astype(np.float64)
    if M.nnz == 0:
        return None, np.inf

    if n <= DENSE_ASSIGNMENT_CAP:
        d = M.toarray()
        mask = d > 0
        with np.errstate(divide="ignore"):
            C = -np.log(d, out=np.full_like(d, -np.inf), where=mask)
        C[mask] -= C[mask].min() - 1.0          # strictly positive weights
        return _dense_assignment(C, mask, n)

    return None, np.inf              # see DENSE_ASSIGNMENT_CAP; the portfolio copes


def apply_permutation(A, b, p):
    """``A' = P A``, ``b' = P b``. The solution x is unchanged, so no un-permuting."""
    return A[p, :].tocsr(), b[p]


def select_diagonal(A, b=None, objective="bottleneck"):
    """Choose the diagonal, returning ``(A', b', info)``.

    With ``objective="none"`` the matrix is returned untouched, which is the control
    condition for the ablation.
    """
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
    """Cheap power-iteration estimate of rho(T_GS), for ranking candidates only.

    Deliberately not the exact eigendecomposition: this runs once per candidate inside
    the solver, so it has to cost far less than solving. Ranking needs the ordering to be
    right, not the value.
    """
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
    """Try every candidate permutation and keep the one with the smallest estimated
    rho(T_GS). Strictly at least as good as any fixed choice, because "no permutation"
    is itself one of the candidates."""
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
        # Every candidate's worst row ratio, recorded whether or not it wins. This is
        # what lets the write-up compare the objectives on the quantity the convergence
        # guarantee is stated in, without paying for a fourth condition in the sweep --
        # the permutations are already computed here.
        ratios[name] = worst_row_ratio(cand)
        if np.any(np.abs(cand.diagonal()) < 1e-14):
            tried[name] = np.inf                 # still undefined, unusable
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
    """Wrap any stationary solver so it selects its diagonal first.

    The returned callable keeps the project's ``(x, iterations, converged, work)``
    contract, so it drops straight into the benchmark harness alongside every other
    method.
    """
    def solve(A, b):
        t0 = time.perf_counter()
        A2, b2, _ = select_diagonal(A, b, objective=objective)
        setup = time.perf_counter() - t0
        x, iters, converged, work = base_solver(A2, b2)
        # The permutation is part of this method's cost, not a free gift from outside.
        # A wrapper that hides its own setup wins on cost by bookkeeping.
        if work is not None and hasattr(work, "setup"):
            work.setup += setup
        return x, iters, converged, work
    solve.__name__ = f"{base_solver.__name__}_{objective}"
    return solve
