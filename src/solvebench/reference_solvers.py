# Library solvers (SuperLU, ILU+Krylov) -- what an engineer would actually reach for.
import time

import numpy as np
import scipy.sparse.linalg as spla

from .direct_solvers import SolverNotApplicable
from .iterative_solvers import Work, build_ilu, bicgstab, gmres, pcg, _is_symmetric


def sparse_lu(A, b):
    # SuperLU factorization then a single triangular solve pair, counted as 1 iteration.
    work = Work()
    t0 = time.perf_counter()
    try:
        lu = spla.splu(A.tocsc())
    except RuntimeError as e:
        raise SolverNotApplicable(f"SuperLU could not factor this matrix: {e}") from e
    work.setup = time.perf_counter() - t0
    x = lu.solve(b)
    work.tri_solves += 2
    return x, 1, True, work


def sparse_spsolve(A, b):
    # The one-line answer: ``spsolve(A, b)``. What a practitioner types.
    work = Work()
    t0 = time.perf_counter()
    try:
        x = spla.spsolve(A.tocsc(), b)
    except RuntimeError as e:
        raise SolverNotApplicable(f"spsolve failed: {e}") from e
    work.setup = time.perf_counter() - t0
    work.tri_solves += 2
    return x, 1, True, work


def ilu_only(A, b):
    # Zero-iteration control: apply the ILU factorization to b and stop, no Krylov on top.
    work = Work()
    apply_ilu = build_ilu(A, work)
    if apply_ilu is None:
        raise SolverNotApplicable("no usable ILU factorization for this matrix")
    x = apply_ilu(b)
    work.precond += 1
    return x, 0, True, work


def ilu_bicgstab(A, b):
    # ILU-preconditioned BiCGSTAB, run unconditionally -- no dispatch.
    work = Work()
    apply_ilu = build_ilu(A, work)
    if apply_ilu is None:
        raise SolverNotApplicable("no usable ILU factorization for this matrix")
    x, its, conv, w = bicgstab(A, b, M=apply_ilu)
    w.setup = work.setup
    return x, its, conv, w


def ilu_gmres(A, b, restart=30):
    # ILU-preconditioned restarted GMRES: PETSc's default configuration.
    work = Work()
    apply_ilu = build_ilu(A, work)
    if apply_ilu is None:
        raise SolverNotApplicable("no usable ILU factorization for this matrix")
    x, its, conv, w = gmres(A, b, M=apply_ilu, restart=restart)
    w.setup = work.setup
    return x, its, conv, w


def ilu_krylov_dispatched(A, b):
    # PCG if A is symmetric else BiCGSTAB -- this is APK, kept as an honest baseline, not a novelty.
    work = Work()
    apply_ilu = build_ilu(A, work)
    if apply_ilu is None:
        raise SolverNotApplicable("no usable ILU factorization for this matrix")
    if _is_symmetric(A):
        x, its, conv, w = pcg(A, b, M=apply_ilu)
        if conv and np.all(np.isfinite(x)):
            w.setup = work.setup
            return x, its, conv, w
    x, its, conv, w = bicgstab(A, b, M=apply_ilu)     # fallback and default path
    w.setup = work.setup
    return x, its, conv, w
