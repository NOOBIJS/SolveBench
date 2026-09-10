"""The benchmark harness.

For every matrix in the corpus and every method defined on it, run the solve and
record what happened. Three rules hold throughout, and each exists because the
first sweep broke it:

1. **One judge.** No solver decides its own outcome. Every result goes through
   :func:`solvebench.metrics.score`, which measures the residual from A, x and b
   and applies one tolerance to all methods.
2. **Every matrix is accounted for.** A matrix that is skipped, unloadable or
   structurally singular still produces a row, with a reason code, for every
   method. Denominators reconcile to the corpus size by construction rather than
   by hoping nothing was dropped.
3. **Refinement is a factor, not a feature.** Every method is run at each
   refinement pass count, so no method's accuracy is credited to a wrapper the
   others did not get.
"""
import time
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import config, direct_solvers as direct, iterative_solvers as it
from . import reordering
from . import metrics, reference_solvers as ref, spectral
from .direct_solvers import SolverNotApplicable
from .io_utils import (cancellation_ratio, discover_matrices, load_matrix,
                       make_ground_truth, structural_singularity)
from .refinement import refine


class Method:
    """One benchmarked method and the facts the harness needs about it."""

    def __init__(self, name, fn, family, dense=False, capped=False):
        self.name = name
        self.fn = fn
        self.family = family
        self.dense = dense      # takes a dense array rather than a sparse matrix
        self.capped = capped    # subject to DIRECT_SIZE_CAP

    def __repr__(self):
        return f"<Method {self.name}>"


def _wrap_dense(fn):
    """Adapt a hand-written direct solver to the common 4-tuple contract."""
    def call(A_dense, b):
        x, steps = fn(A_dense, b)
        return x, steps, None, it.Work()      # None: it makes no convergence claim
    return call


#: Every method in the benchmark. The hand-written direct solvers are labelled
#: "direct-handwritten" and are pedagogical implementations validated against
#: LAPACK, not performance competitors -- they are pure-Python O(n^3) and about
#: 50-100x slower than a library call, which is why they alone carry a size cap.
METHODS = [
    Method("Gauss elimination", _wrap_dense(direct.gauss_elimination), "direct-handwritten", dense=True, capped=True),
    Method("Gauss-Jordan", _wrap_dense(direct.gauss_jordan), "direct-handwritten", dense=True, capped=True),
    Method("LU", _wrap_dense(direct.lu_solve), "direct-handwritten", dense=True, capped=True),
    Method("Cholesky", _wrap_dense(direct.cholesky_solve), "direct-handwritten", dense=True, capped=True),

    Method("spsolve (SuperLU)", ref.sparse_spsolve, "direct-library"),
    Method("splu (SuperLU)", ref.sparse_lu, "direct-library"),

    Method("Jacobi", it.jacobi, "stationary"),
    Method("Gauss-Seidel", it.gauss_seidel, "stationary"),
    Method("SOR", it.sor, "stationary"),

    Method("Conjugate Gradient", it.conjugate_gradient, "krylov"),
    Method("BiCGSTAB", it.bicgstab, "krylov"),
    Method("GMRES(30)", lambda A, b: it.gmres(A, b, restart=30), "krylov"),

    Method("ILU only", ref.ilu_only, "preconditioned"),
    Method("ILU-BiCGSTAB", ref.ilu_bicgstab, "preconditioned"),
    Method("ILU-GMRES(30)", ref.ilu_gmres, "preconditioned"),
    Method("ILU-Krylov (dispatched)", ref.ilu_krylov_dispatched, "preconditioned"),
]

#: The proposed pipeline, and the arms needed to attribute its effect.
#:
#: Two preprocessing steps sit in front of unmodified solvers:
#:
#:   1. choose the diagonal by assignment  -- makes D invertible, and makes an
#:      incomplete factorization constructible, on matrices where neither was true
#:   2. choose omega from an estimate of rho(T_J) -- replaces the fixed 1.25
#:
#: Both are measured separately as well as together, because "the pipeline helps" is
#: not a result: which step helps, and by how much, is. Baselines for every row here
#: already exist in the main sweep, so this list is run on its own rather than by
#: repeating all sixteen methods.
#:
#: Step 1 is not offered to the direct or unpreconditioned-Krylov families: they never
#: divide by a diagonal entry, so a row permutation changes their arithmetic without
#: addressing anything they were failing on.
PIPELINE_METHODS = [
    # step 2 alone -- isolates the relaxation factor from the reordering
    Method("SOR (adaptive w)", it.sor_adaptive, "pipeline"),

    # step 1 alone, on the methods that need a usable diagonal to exist at all
    Method("Jacobi + reordered", reordering.make_solver(it.jacobi, "best"), "pipeline"),
    Method("Gauss-Seidel + reordered",
           reordering.make_solver(it.gauss_seidel, "best"), "pipeline"),
    Method("SOR + reordered", reordering.make_solver(it.sor, "best"), "pipeline"),

    # both steps
    Method("SOR + pipeline", reordering.make_solver(it.sor_adaptive, "best"), "pipeline"),

    # step 1 in front of the preconditioned family: ILU fails to build on 166 matrices
    # and a zero on the diagonal is what closes its last fallback
    Method("ILU-BiCGSTAB + reordered",
           reordering.make_solver(ref.ilu_bicgstab, "best"), "pipeline"),
    Method("ILU-Krylov + reordered",
           reordering.make_solver(ref.ilu_krylov_dispatched, "best"), "pipeline"),
]

PIPELINE_NAMES = [m.name for m in PIPELINE_METHODS]

METHOD_NAMES = [m.name for m in METHODS]


def condition_number(A, A_dense, n):
    """Exact via SVD where affordable, a 1-norm estimate otherwise.

    The route is returned alongside the value: an exact condition number and an
    estimate are not the same measurement and must not be pooled silently.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if A_dense is not None and n <= config.COND_EXACT_CAP:
            try:
                return float(np.linalg.cond(A_dense)), "exact_svd"
            except (np.linalg.LinAlgError, MemoryError):
                pass
        try:
            lu = spla.splu(A.tocsc())
            inv_norm = spla.onenormest(spla.LinearOperator(
                A.shape, matvec=lu.solve, rmatvec=lambda v: lu.solve(v, "T")))
            return float(spla.onenormest(A) * inv_norm), "estimate_1norm"
        except Exception:
            return float("inf"), "failed"


def _row(base, method, passes, **extra):
    return {**base, "method": method.name, "family": method.family,
            "refinement_passes": passes, **extra}


def run_one_matrix(domain, name, mtx_path, refinement_passes=(0, 1),
                   with_spectral=True, seed=None, verbose=True, methods=None):
    """Benchmark a set of methods on one matrix. Returns (result_rows, spectral_row).

    ``methods`` defaults to the sixteen baseline methods. Passing ``PIPELINE_METHODS``
    runs the proposed pipeline instead, on the same corpus and through the same scoring,
    so the two are directly comparable without repeating the baselines that are already
    measured.
    """
    methods = METHODS if methods is None else methods
    try:
        A = load_matrix(mtx_path)
    except Exception as e:
        base = {"domain": domain, "matrix": name, "n": np.nan, "nnz": np.nan}
        rows = [_row(base, m, p, **metrics.blank("load_failed", f"{type(e).__name__}: {e}"))
                for m in methods for p in refinement_passes]
        return rows, {"domain": domain, "matrix": name, "status": "load_failed"}

    n = A.shape[0]
    base = {"domain": domain, "matrix": name, "n": n, "nnz": A.nnz,
            "density": A.nnz / (n * n)}

    zero_rows, zero_cols = structural_singularity(A)
    if zero_rows or zero_cols:
        if verbose:
            print(f"    structurally singular ({zero_rows} zero rows, {zero_cols} zero cols)"
                  " -- no unique solution, all methods skipped")
        note = f"{zero_rows} zero rows, {zero_cols} zero cols"
        rows = [_row(base, m, p, **metrics.blank(metrics.STATUS_SINGULAR, note))
                for m in methods for p in refinement_passes]
        return rows, {**base, "status": metrics.STATUS_SINGULAR}

    x_true, b = make_ground_truth(A, seed=seed)
    b_norm = np.linalg.norm(b) or 1.0
    xt_norm = np.linalg.norm(x_true) or 1.0

    needs_dense = n <= max(config.DIRECT_SIZE_CAP, config.COND_EXACT_CAP)
    A_dense = A.toarray() if needs_dense else None

    cond, cond_how = condition_number(A, A_dense, n)
    base.update(condition_number=cond, condition_method=cond_how,
                ill_conditioned=bool(cond > config.ILL_CONDITIONED),
                cancellation_ratio=cancellation_ratio(A, x_true))
    if verbose:
        print(f"    n={n:,} nnz={A.nnz:,} cond={cond:.2e} ({cond_how})")

    rows = []
    for m in methods:
        operand = A_dense if m.dense else A
        for passes in refinement_passes:
            if m.capped and n > config.DIRECT_SIZE_CAP:
                rows.append(_row(base, m, passes,
                                 **metrics.blank(metrics.STATUS_SKIPPED,
                                                 f"n={n} > cap {config.DIRECT_SIZE_CAP}")))
                continue
            if m.dense and A_dense is None:
                rows.append(_row(base, m, passes,
                                 **metrics.blank(metrics.STATUS_SKIPPED, "dense form not built")))
                continue
            try:
                t0 = time.perf_counter()
                if passes:
                    x, iters, conv, work = refine(m.fn, operand, b, passes=passes)
                else:
                    x, iters, conv, work = m.fn(operand, b)
                elapsed = time.perf_counter() - t0
                scored = metrics.score(A, x, b, x_true, b_norm, xt_norm, conv)
                rows.append(_row(base, m, passes, runtime_sec=elapsed, iterations=iters,
                                 **work.as_dict(), **scored))
            except SolverNotApplicable as e:
                rows.append(_row(base, m, passes,
                                 **metrics.blank(metrics.STATUS_NOT_APPLICABLE, str(e))))
            except (MemoryError, RuntimeError, ValueError, ZeroDivisionError) as e:
                rows.append(_row(base, m, passes,
                                 **metrics.blank(metrics.STATUS_ERROR, f"{type(e).__name__}: {e}")))
        if verbose:
            last = rows[-1]
            print(f"      {m.name:<24s} {last['status']}")

    spec = {**base, "status": "analysed"}
    if with_spectral:
        try:
            rho_j, how_j = spectral.spectral_radius(A, "jacobi")
            rho_g, how_g = spectral.spectral_radius(A, "gauss_seidel")
            spec.update(spectral.classify(A, rho_j, rho_g),
                        rho_jacobi_method=how_j, rho_gs_method=how_g)
        except (MemoryError, ValueError) as e:
            spec.update(status=f"spectral_failed: {type(e).__name__}")

    return rows, spec


def run_full_benchmark(dataset_root, results_csv, spectral_csv=None,
                       refinement_passes=(0, 1), with_spectral=True,
                       seed=None, verbose=True):
    """Sweep the whole corpus, checkpointing after every matrix."""
    entries = discover_matrices(dataset_root)
    all_rows, all_spec = [], []
    t0 = time.perf_counter()

    for i, entry in enumerate(entries, 1):
        if verbose:
            print(f"\n[{i}/{len(entries)}] {entry['domain']}/{entry['name']}"
                  f"   ({(time.perf_counter() - t0) / 60:.1f} min elapsed)")
        rows, spec = run_one_matrix(entry["domain"], entry["name"], entry["path"],
                                    refinement_passes=refinement_passes,
                                    with_spectral=with_spectral, seed=seed, verbose=verbose)
        all_rows.extend(rows)
        all_spec.append(spec)
        pd.DataFrame(all_rows).to_csv(results_csv, index=False)     # checkpoint
        if spectral_csv:
            pd.DataFrame(all_spec).to_csv(spectral_csv, index=False)

    results = pd.DataFrame(all_rows)
    if verbose:
        expected = len(entries) * len(METHODS) * len(refinement_passes)
        print(f"\nSweep complete in {(time.perf_counter() - t0) / 60:.1f} min")
        print(f"  rows {len(results):,} of {expected:,} expected"
              f"  |  matrices {results['matrix'].nunique()} of {len(entries)}")
        print(results["status"].value_counts().to_string())
    return results, pd.DataFrame(all_spec)


def summarise(results, passes=0):
    """Applicability and conditional success per method, as separate columns.

    These are never multiplied into a single rate: doing that is what reported
    Conjugate Gradient at 10.4% when its conditional success is 83.9%, and
    Cholesky at 10.1% when it solves every system it applies to.
    """
    subset = results[results["refinement_passes"] == passes]
    return pd.DataFrame([metrics.rates(subset, m) for m in METHOD_NAMES
                         if m in set(subset["method"])])
