"""Iterative solvers for Ax = b on sparse matrices.

Every solver returns ``(x, iterations, reported_converged, work)``. The third
value is the solver's own opinion and is recorded but never trusted -- success
is decided afterwards by :mod:`solvebench.metrics` from the residual we measure
ourselves. The fourth is a :class:`Work` tally of the operations that actually
cost something, which is the cost metric the comparison uses: wall-clock on a
shared Kaggle CPU is not reproducible, and "iterations" are not comparable
across methods because one preconditioned step does far more work than one
Jacobi step.

Two implementation notes matter for the fairness of the comparison:

* Gauss-Seidel and SOR are written in *delta form*, ``(D + wL) d = w r`` with
  ``x <- x + d``. The previous version solved ``(D+L) x = b - Ux`` by calling
  ``spsolve_triangular`` inside the loop, which re-analysed the triangular
  structure on every one of up to 10,000 iterations, and then computed the
  residual with a second matrix-vector product. Delta form factors the
  triangular matrix once and gets the residual for free, so what the timer sees
  is the algorithm rather than repeated setup.
* Jacobi is written the same way for the same reason: ``x <- x + r/d``.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import config
from .direct_solvers import SolverNotApplicable


class Work:
    """Counts the operations a solve actually performed.

    Wall-clock timing on shared hardware is not reproducible and iteration
    counts are not comparable between methods, so this is the primary cost
    metric. A preconditioner application is counted separately from a plain
    matrix-vector product because it is typically far more expensive.
    """

    __slots__ = ("matvecs", "tri_solves", "precond", "setup")

    def __init__(self):
        self.matvecs = 0
        self.tri_solves = 0
        self.precond = 0
        self.setup = 0.0      # seconds spent building a factorization, if any

    def as_dict(self):
        return {"matvecs": self.matvecs, "tri_solves": self.tri_solves,
                "precond_applies": self.precond, "setup_sec": self.setup}


def _counting_operator(A, work):
    """Wrap A so that every product a library solver takes gets counted."""
    def matvec(v):
        work.matvecs += 1
        return A @ v
    return spla.LinearOperator(A.shape, matvec=matvec, dtype=np.float64)


def _counting_preconditioner(apply_fn, shape, work):
    def matvec(v):
        work.precond += 1
        return apply_fn(v)
    return spla.LinearOperator(shape, matvec=matvec, dtype=np.float64)


def _diagonal_or_raise(A, what):
    d = A.diagonal()
    if np.any(np.abs(d) < 1e-14):
        raise SolverNotApplicable(f"{what} needs a nonzero diagonal "
                                  f"({int((np.abs(d) < 1e-14).sum())} zero entries)")
    return d


def _prefactor_lower(M):
    """Factor a lower-triangular iteration matrix once, for repeated solves.

    ``permc_spec="NATURAL"`` and ``diag_pivot_thresh=0`` keep SuperLU from
    reordering or pivoting, so the factorization of an already-triangular
    matrix is essentially free and every later solve is a plain substitution.
    """
    return spla.splu(M.tocsc(), permc_spec="NATURAL", diag_pivot_thresh=0.0)


# --------------------------------------------------------------- stationary


def jacobi(A, b, max_iter=config.MAX_ITERATIONS, tol=config.TOLERANCE):
    """x <- x + D^-1 (b - Ax). One matrix-vector product per iteration, and the
    residual needed for the stopping test falls out of it at no extra cost."""
    work = Work()
    d = _diagonal_or_raise(A, "Jacobi")
    b_norm = np.linalg.norm(b) or 1.0
    x = np.zeros(A.shape[0])
    best = np.inf

    for k in range(1, max_iter + 1):
        r = b - A @ x
        work.matvecs += 1
        rr = np.linalg.norm(r) / b_norm
        if not np.isfinite(rr):
            return x, k, False, work
        if rr <= tol:
            return x, k, True, work
        if rr > best * config.DIVERGENCE_GROWTH:
            return x, k, False, work
        best = min(best, rr)
        x = x + r / d

    return x, max_iter, False, work


def _sor_core(A, b, omega, max_iter, tol, label):
    """Shared engine for Gauss-Seidel (omega = 1) and SOR.

    Delta form: (D + omega*L) delta = omega * r, then x <- x + delta. Writing
    both methods through one routine guarantees they are treated identically;
    the only difference between them is omega.
    """
    work = Work()
    _diagonal_or_raise(A, label)
    n = A.shape[0]
    b_norm = np.linalg.norm(b) or 1.0

    M = (sp.tril(A, k=-1, format="csr") * omega + sp.diags(A.diagonal())).tocsc()
    try:
        lu = _prefactor_lower(M)
    except RuntimeError as e:                      # singular triangular part
        raise SolverNotApplicable(f"{label}: {e}") from e

    x = np.zeros(n)
    best = np.inf

    for k in range(1, max_iter + 1):
        r = b - A @ x
        work.matvecs += 1
        rr = np.linalg.norm(r) / b_norm
        if not np.isfinite(rr):
            return x, k, False, work
        if rr <= tol:
            return x, k, True, work
        if rr > best * config.DIVERGENCE_GROWTH:
            return x, k, False, work
        best = min(best, rr)
        x = x + lu.solve(omega * r)
        work.tri_solves += 1

    return x, max_iter, False, work


def gauss_seidel(A, b, max_iter=config.MAX_ITERATIONS, tol=config.TOLERANCE):
    return _sor_core(A, b, 1.0, max_iter, tol, "Gauss-Seidel")


def sor(A, b, omega=config.SOR_OMEGA, max_iter=config.MAX_ITERATIONS, tol=config.TOLERANCE):
    return _sor_core(A, b, omega, max_iter, tol, "SOR")


# --------------------------------------------------------------- Krylov


def _is_symmetric(A, tol=1e-8):
    diff = abs(A - A.T)
    return diff.nnz == 0 or diff.max() <= tol * max(abs(A).max(), 1.0)


def conjugate_gradient(A, b, max_iter=config.MAX_ITERATIONS, tol=config.TOLERANCE):
    """Unpreconditioned CG. Requires symmetry; indefiniteness shows up as a
    non-positive curvature term and is reported rather than silently ignored."""
    if not _is_symmetric(A):
        raise SolverNotApplicable("Conjugate Gradient needs a symmetric matrix")
    work = Work()
    b_norm = np.linalg.norm(b) or 1.0
    x = np.zeros(A.shape[0])
    r = b.copy()
    p = r.copy()
    rs = r @ r

    for k in range(1, max_iter + 1):
        Ap = A @ p
        work.matvecs += 1
        pAp = p @ Ap
        if pAp <= 0:
            raise SolverNotApplicable("Conjugate Gradient needs positive definiteness "
                                      f"(p'Ap = {pAp:.3e} at iteration {k})")
        alpha = rs / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rr = np.linalg.norm(r) / b_norm
        if not np.isfinite(rr):
            return x, k, False, work
        if rr <= tol:
            return x, k, True, work
        rs_new = r @ r
        p = r + (rs_new / rs) * p
        rs = rs_new

    return x, max_iter, False, work


def bicgstab(A, b, M=None, max_iter=config.MAX_ITERATIONS, tol=config.TOLERANCE):
    """BiCGSTAB via scipy, with A and the preconditioner wrapped so their
    applications are counted. Unlike the hand-written stationary methods this is
    a library implementation, and the results table labels it as such."""
    work = Work()
    op = _counting_operator(A, work)
    pre = None if M is None else _counting_preconditioner(M, A.shape, work)
    x, info = spla.bicgstab(op, b, rtol=tol, atol=0.0, maxiter=max_iter, M=pre)
    return x, work.matvecs, info == 0, work


def pcg(A, b, M=None, max_iter=config.MAX_ITERATIONS, tol=config.TOLERANCE):
    work = Work()
    op = _counting_operator(A, work)
    pre = None if M is None else _counting_preconditioner(M, A.shape, work)
    x, info = spla.cg(op, b, rtol=tol, atol=0.0, maxiter=max_iter, M=pre)
    return x, work.matvecs, info == 0, work


def gmres(A, b, M=None, restart=30, max_iter=config.MAX_ITERATIONS, tol=config.TOLERANCE):
    """Restarted GMRES -- the default Krylov method in PETSc, and the baseline
    whose absence made the dispatched solver look better than it is."""
    work = Work()
    op = _counting_operator(A, work)
    pre = None if M is None else _counting_preconditioner(M, A.shape, work)
    x, info = spla.gmres(op, b, rtol=tol, atol=0.0, restart=restart,
                         maxiter=max_iter // restart or 1, M=pre)
    return x, work.matvecs, info == 0, work


# --------------------------------------------------------------- preconditioning


def build_ilu(A, work=None):
    """Incomplete LU with the same relax-and-retry ladder the earlier run used.

    Returns an apply function, or None if no usable factorization was found.
    Note for the write-up: SuperLU's spilu applies a column permutation and
    partial pivoting, so the resulting factor is *not* symmetric even when A is.
    Feeding it to CG as a preconditioner violates CG's requirement that the
    preconditioner be symmetric positive definite, which is why the dispatched
    solver's symmetric branch is reported as a known defect rather than a
    feature.
    """
    import time
    t0 = time.perf_counter()
    for drop, fill in ((config.ILU_DROP_TOL, config.ILU_FILL_FACTOR),
                       (config.ILU_DROP_TOL * 10, config.ILU_FILL_FACTOR * 2),
                       (1e-2, 20)):
        try:
            ilu = spla.spilu(A.tocsc(), drop_tol=drop, fill_factor=fill)
        except (RuntimeError, ValueError, MemoryError):
            continue
        if work is not None:
            work.setup += time.perf_counter() - t0
        return ilu.solve

    d = A.diagonal()                          # last resort: diagonal scaling
    if np.any(np.abs(d) < 1e-14):
        return None
    if work is not None:
        work.setup += time.perf_counter() - t0
    return lambda v: v / d
