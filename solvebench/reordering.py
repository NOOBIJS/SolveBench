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
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import maximum_bipartite_matching, min_weight_full_bipartite_matching

OBJECTIVES = ("bottleneck", "minsum", "mc64", "best", "none")

#: Candidates the "best" objective chooses among. No single one dominates: on
#: ``odepa400`` bottleneck reaches rho(T_GS) = 0.495 where min-sum gives 1.265 and MC64
#: leaves it at 1.0001; on ``d_ss`` bottleneck is the worst of the three at 29.6 against
#: MC64's 1.83. Since each permutation costs milliseconds, the honest method computes
#: all of them and picks by direct spectral estimate.
PORTFOLIO = ("none", "mc64", "minsum", "bottleneck")


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
    R = row_ratios(A)
    n = A.shape[0]
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
        upper = float(np.max(np.diag(row_ratios(A[mc_perm, :]).toarray()))) if n <= 2000 \
            else float(np.max(row_ratios(A[mc_perm, :]).diagonal()))
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

    lo, hi, best = 0, candidates.size - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        G = _mask(R, R.data <= candidates[mid])
        m = maximum_bipartite_matching(G, perm_type="column")
        if (m >= 0).all():
            best = (m.copy(), candidates[mid])
            hi = mid - 1
        else:
            lo = mid + 1
    if best is None:
        return None, np.inf
    m, worst = best
    return _perm_from_matching(m, n), float(worst)


def _mask(R, keep):
    """R restricted to the entries `keep` selects, as a boolean pattern."""
    out = sp.csr_matrix((keep.astype(bool), R.indices.copy(), R.indptr.copy()),
                        shape=R.shape)
    out.eliminate_zeros()
    return out


def minsum_permutation(A):
    """Minimise the TOTAL of the chosen row ratios: ``min_pi sum_i R[i, pi(i)]``.

    The natural alternative to the bottleneck objective. Bottleneck protects the worst
    row and can wreck the rest -- on ``d_ss`` it drove rho(T_GS) from 1.884 to 29.605 --
    whereas min-sum trades the worst row away to improve the average. Which is better is
    an open question this benchmark is meant to answer.
    """
    R = row_ratios(A).tocsr()
    if R.nnz == 0:
        return None, np.inf

    # Minimise sum(log(1 + ratio)), i.e. the PRODUCT of (1 + ratio), not the raw sum.
    # Two reasons, and both matter. Statistically, a raw sum is dominated by whichever
    # single row has the largest ratio -- on nnc261 the ratios span 0 to 3.8e10 -- so
    # "min-sum" would barely differ from the bottleneck objective it is supposed to
    # contrast with. Numerically, min_weight_full_bipartite_matching runs a shortest-path
    # augmentation that degrades badly over that dynamic range: on nnc261 it had not
    # returned after 240 seconds with raw ratios, and finishes immediately under log.
    R.data = np.log1p(R.data) + 1e-300
    try:
        rows, cols = min_weight_full_bipartite_matching(R)
    except ValueError:
        return None, np.inf          # no perfect matching exists
    order = np.argsort(rows)
    return _perm_from_matching(cols[order], A.shape[0]), float(R[rows, cols].sum())


def mc64_permutation(A):
    """Baseline: maximise the product of |diagonal| entries, as MC64 does.

    Equivalent to minimising ``sum -log|a_ij|``. This is the established tool, built for
    pivot stability rather than for convergence, and it is the comparison the method has
    to beat.
    """
    M = abs(A).tocsr().astype(np.float64)
    if M.nnz == 0:
        return None, np.inf
    C = M.copy()
    with np.errstate(divide="ignore"):
        C.data = -np.log(M.data)
    C.data = C.data - C.data.min() + 1e-300     # keep weights positive and finite
    try:
        rows, cols = min_weight_full_bipartite_matching(C)
    except ValueError:
        return None, np.inf
    order = np.argsort(rows)
    return _perm_from_matching(cols[order], A.shape[0]), float(C[rows, cols].sum())


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
            "zero_diagonal_before": int((np.abs(A.diagonal()) < 1e-14).sum())}

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
                worst_ratio_after=float(np.max(np.diag(row_ratios(A2).toarray()))
                                        if n <= 2000 else np.nan))
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
    tried = {}
    for name in PORTFOLIO:
        if name == "none":
            cand = A
        else:
            p, _ = {"mc64": mc64_permutation, "minsum": minsum_permutation,
                    "bottleneck": bottleneck_permutation}[name](A)
            if p is None:
                continue
            cand = A[p, :].tocsr()
        if np.any(np.abs(cand.diagonal()) < 1e-14):
            tried[name] = np.inf                 # still undefined, unusable
            continue
        rho = _estimate_rho_gs(cand)
        tried[name] = rho
        if rho < best[0]:
            best = (rho, cand, None if name == "none" else p, name)

    rho, cand, p, name = best
    info.update(chosen=name, estimated_rho_gs=rho, candidates=tried,
                permuted=name != "none")
    if cand is None:
        info["note"] = "no candidate produced a usable diagonal"
        return A, b, info
    info["zero_diagonal_after"] = 0
    return cand, (b if p is None or b is None else b[p]), info


def make_solver(base_solver, objective="bottleneck"):
    """Wrap any stationary solver so it selects its diagonal first.

    The returned callable keeps the project's ``(x, iterations, converged, work)``
    contract, so it drops straight into the benchmark harness alongside every other
    method.
    """
    def solve(A, b):
        A2, b2, _ = select_diagonal(A, b, objective=objective)
        return base_solver(A2, b2)
    solve.__name__ = f"{base_solver.__name__}_{objective}"
    return solve
