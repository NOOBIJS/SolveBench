# Iterative solvers for Ax = b; cost is tracked by Work, not wall-clock or iteration count.
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import config
from .direct_solvers import SolverNotApplicable


class Work:
    # Counts the operations a solve actually performed -- the real cost metric here.

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
    # Wrap A so that every product a library solver takes gets counted.
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
    # Factor a lower-triangular matrix once (no reordering/pivoting) for repeated solves.
    return spla.splu(M.tocsc(), permc_spec="NATURAL", diag_pivot_thresh=0.0)


# --------------------------------------------------------------- stationary


def jacobi(A, b, max_iter=config.MAX_ITERATIONS, tol=config.TOLERANCE):
    # x <- x + D^-1 (b - Ax), one matvec per iteration.
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
    # Shared engine for Gauss-Seidel (omega=1) and SOR: (D+wL)delta=w*r, x<-x+delta.
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

def estimate_rho_jacobi(A, iters=None, seed=0, work=None):
    # Power-iteration estimate of rho(T_J); matvecs charged to `work`, NaN if diagonal has a zero.
    iters = config.POWER_ITERS_OMEGA if iters is None else iters
    d = A.diagonal()
    if np.any(np.abs(d) < 1e-14):
        return float("nan")
    rng = np.random.default_rng(seed)
    v = rng.normal(size=A.shape[0])
    nv = np.linalg.norm(v)
    if nv == 0:
        return float("nan")
    v /= nv
    lam = 0.0
    for _ in range(iters):
        w = v - (A @ v) / d
        if work is not None:
            work.matvecs += 1
        nw = np.linalg.norm(w)
        if nw < 1e-300:
            return 0.0
        lam = nw
        v = w / nw
    return float(lam)


def optimal_omega(rho):
    # Young (1950) formula; falls back to 1.0 (plain Gauss-Seidel) when rho isn't usable.
    if not np.isfinite(rho) or rho >= 1.0:
        return 1.0
    return float(np.clip(2.0 / (1.0 + np.sqrt(max(1.0 - rho * rho, 0.0))), 1.0, 1.95))


def sor_adaptive(A, b, max_iter=config.MAX_ITERATIONS, tol=config.TOLERANCE):
    # SOR with omega estimated per matrix instead of fixed at one value for the corpus.
    work = Work()
    rho = estimate_rho_jacobi(A, work=work)
    omega = optimal_omega(rho)
    x, iters, converged, w2 = _sor_core(A, b, omega, max_iter, tol, "SOR (adaptive)")
    w2.matvecs += work.matvecs          # the estimate is part of this method's cost
    return x, iters, converged, w2

# --------------------------------------------------------------- Krylov


def _is_symmetric(A, tol=1e-8):
    diff = abs(A - A.T)
    return diff.nnz == 0 or diff.max() <= tol * max(abs(A).max(), 1.0)


def conjugate_gradient(A, b, max_iter=config.MAX_ITERATIONS, tol=config.TOLERANCE):
    # Unpreconditioned CG; needs symmetry, indefiniteness reported as p'Ap <= 0.
    if not _is_symmetric(A):
        raise SolverNotApplicable("Conjugate Gradient needs a symmetric matrix")
    work = Work()
    b_norm = np.linalg.norm(b) or 1.0
    x = np.zeros(A.shape[0])
    r = b.copy()
    p = r.copy()
    rs = r @ r

    if np.sqrt(rs) <= tol * b_norm:     # x=0 already solves it, else p'Ap=0 looks indefinite
        return x, 0, True, work

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
    # BiCGSTAB via scipy, A and the preconditioner wrapped so their calls get counted.
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
    # Restarted GMRES, PETSc's default Krylov method.
    work = Work()
    op = _counting_operator(A, work)
    pre = None if M is None else _counting_preconditioner(M, A.shape, work)
    x, info = spla.gmres(op, b, rtol=tol, atol=0.0, restart=restart,
                         maxiter=max_iter // restart or 1, M=pre)
    return x, work.matvecs, info == 0, work


# --------------------------------------------------------------- preconditioning


def build_ilu(A, work=None):
    # Incomplete LU with a relax-and-retry ladder; returns an apply function or None.
    import time
    t0 = time.perf_counter()
    Acsc = A.tocsc()
    probe = np.random.default_rng(0).standard_normal(A.shape[0])
    for drop, fill in config.ILU_LADDER:
        try:
            ilu = spla.spilu(Acsc, drop_tol=drop, fill_factor=fill)
        except (RuntimeError, ValueError, MemoryError):
            continue
        try:
            y = ilu.solve(probe)   # a factorization can still be unusable, tiny pivot -> inf
        except Exception:
            continue
        if not np.all(np.isfinite(y)):
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
