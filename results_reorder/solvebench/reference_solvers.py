"""Library solvers: the ones an engineer with this problem would actually reach for.

Their absence from the first sweep is what made the proposed method look strong.
A comparative study of solvers for sparse systems that never runs
``scipy.sparse.linalg.spsolve`` has not compared against the state of practice,
it has compared against four textbook methods from the 1950s.

These are sparse direct factorizations, so unlike the hand-written dense direct
solvers in :mod:`solvebench.direct_solvers` they carry no size cap -- SuperLU on
a sparse matrix costs far less than O(n^3), which is exactly the point being
tested.

Every solver here returns the same ``(x, iterations, reported_converged, work)``
tuple as the iterative ones so the harness can score them identically.
"""
import time

import numpy as np
import scipy.sparse.linalg as spla

from .direct_solvers import SolverNotApplicable
from .iterative_solvers import Work, build_ilu, bicgstab, gmres, pcg, _is_symmetric


def sparse_lu(A, b):
    """SuperLU factorization then a single triangular solve pair.

    Reported as one "iteration" because a direct method does a fixed amount of
    work: there is no convergence loop to count.
    """
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
    """The one-line answer: ``spsolve(A, b)``. What a practitioner types."""
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
    """Zero-iteration control: apply the ILU factorization to b and stop.

    This is the baseline that says how much of the proposed method's result came
    from the preconditioner alone, with no Krylov iteration on top. Without it
    there is no way to attribute the gain.
    """
    work = Work()
    apply_ilu = build_ilu(A, work)
    if apply_ilu is None:
        raise SolverNotApplicable("no usable ILU factorization for this matrix")
    x = apply_ilu(b)
    work.precond += 1
    return x, 0, True, work


def ilu_bicgstab(A, b):
    """ILU-preconditioned BiCGSTAB, run unconditionally -- no dispatch."""
    work = Work()
    apply_ilu = build_ilu(A, work)
    if apply_ilu is None:
        raise SolverNotApplicable("no usable ILU factorization for this matrix")
    x, its, conv, w = bicgstab(A, b, M=apply_ilu)
    w.setup = work.setup
    return x, its, conv, w


def ilu_gmres(A, b, restart=30):
    """ILU-preconditioned restarted GMRES: PETSc's default configuration."""
    work = Work()
    apply_ilu = build_ilu(A, work)
    if apply_ilu is None:
        raise SolverNotApplicable("no usable ILU factorization for this matrix")
    x, its, conv, w = gmres(A, b, M=apply_ilu, restart=restart)
    w.setup = work.setup
    return x, its, conv, w


def ilu_krylov_dispatched(A, b):
    """Symmetry-dispatched ILU-preconditioned Krylov: PCG if A is symmetric,
    BiCGSTAB otherwise.

    This is the method the proposal called APK and claimed as novel. It is the
    decision rule in Barrett et al., *Templates* (SIAM, 1994), and the default
    behaviour of PETSc's KSP and MATLAB's backslash dispatch. It is kept in the
    benchmark as a baseline so the dispatch rule itself can be measured against
    ``ilu_bicgstab`` and ``ilu_gmres`` running unconditionally -- an earlier
    probe put dispatch at 51/70 against always-BiCGSTAB's 51/70, i.e. no
    measurable contribution, and that comparison belongs in the results rather
    than in a claim.

    Known defect, reported rather than hidden: spilu pivots and permutes, so on
    a symmetric A the factor is not symmetric and the PCG branch violates CG's
    preconditioner requirement.
    """
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
