"""The benchmark harness: for every (matrix, method) pair, run the solve and
record runtime, iterations, residual, error, and condition number -- exactly
the five columns described in linear.tex's methodology diagram.
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import direct_solvers as direct
from . import iterative_solvers as iterative
from .direct_solvers import SolverNotApplicable
from .io_utils import discover_matrices, load_matrix, make_ground_truth

DIRECT_METHODS = {
    "Gauss elimination": direct.gauss_elimination,
    "Gauss-Jordan": direct.gauss_jordan,
    "LU": direct.lu_solve,
    "Cholesky": direct.cholesky_solve,
}

ITERATIVE_METHODS = {
    "Jacobi": iterative.jacobi,
    "Gauss-Seidel": iterative.gauss_seidel,
    "SOR": iterative.sor,
    "Conjugate Gradient": iterative.conjugate_gradient,
}

# Our direct solvers are genuinely hand-written (no LAPACK), which makes them
# roughly 50-100x slower than numpy.linalg.solve at this scale -- confirmed by
# timing n=3000 at ~35s. Past this size the O(n^3) cost makes them impractical
# to actually run (tens of minutes per matrix, per method). Iterative methods
# are sparse-based and stay fast regardless of size, so only direct methods
# are capped here.
DIRECT_METHOD_SIZE_CAP = 3000


def _condition_number(A_dense: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(np.linalg.cond(A_dense))
        except np.linalg.LinAlgError:
            return float("nan")


def run_one_matrix(domain: str, name: str, mtx_path: Path) -> list[dict]:
    A_sparse = load_matrix(mtx_path)
    n = A_sparse.shape[0]
    x_true, b = make_ground_truth(A_sparse)
    b_norm = np.linalg.norm(b) or 1.0
    x_true_norm = np.linalg.norm(x_true) or 1.0
    A_dense = A_sparse.toarray()
    cond = _condition_number(A_dense)

    rows = []

    for method_name, solver in DIRECT_METHODS.items():
        row = {"domain": domain, "matrix": name, "n": n, "nnz": A_sparse.nnz,
               "method": method_name, "type": "direct", "condition_number": cond}
        if n > DIRECT_METHOD_SIZE_CAP:
            row.update({"status": f"skipped_too_large_for_hand_written_direct_solver (n={n} > {DIRECT_METHOD_SIZE_CAP})",
                        "runtime_sec": None, "iterations": None, "residual_abs": None,
                        "error_abs": None, "residual_rel": None, "error_rel": None})
            rows.append(row)
            continue
        try:
            t0 = time.perf_counter()
            x, steps = solver(A_dense, b)
            elapsed = time.perf_counter() - t0
            residual_abs = float(np.linalg.norm(A_sparse @ x - b))
            error_abs = float(np.linalg.norm(x - x_true))
            row.update({
                "status": "ok",
                "runtime_sec": elapsed,
                "iterations": steps,
                "residual_abs": residual_abs,
                "error_abs": error_abs,
                "residual_rel": residual_abs / b_norm,
                "error_rel": error_abs / x_true_norm,
            })
        except SolverNotApplicable as e:
            row.update({"status": f"not_applicable: {e}", "runtime_sec": None, "iterations": None,
                        "residual_abs": None, "error_abs": None, "residual_rel": None, "error_rel": None})
        rows.append(row)

    for method_name, solver in ITERATIVE_METHODS.items():
        row = {"domain": domain, "matrix": name, "n": n, "nnz": A_sparse.nnz,
               "method": method_name, "type": "iterative", "condition_number": cond}
        try:
            t0 = time.perf_counter()
            x, its, converged = solver(A_sparse, b)
            elapsed = time.perf_counter() - t0
            residual_abs = float(np.linalg.norm(A_sparse @ x - b))
            error_abs = float(np.linalg.norm(x - x_true))
            row.update({
                "status": "ok" if converged else "did_not_converge",
                "runtime_sec": elapsed,
                "iterations": its,
                "residual_abs": residual_abs,
                "error_abs": error_abs,
                "residual_rel": residual_abs / b_norm,
                "error_rel": error_abs / x_true_norm,
            })
        except SolverNotApplicable as e:
            row.update({"status": f"not_applicable: {e}", "runtime_sec": None, "iterations": None,
                        "residual_abs": None, "error_abs": None, "residual_rel": None, "error_rel": None})
        rows.append(row)

    return rows


def run_full_benchmark(dataset_root: Path, output_csv: Path, verbose: bool = True) -> pd.DataFrame:
    matrices = discover_matrices(dataset_root)
    all_rows = []
    for i, entry in enumerate(matrices, 1):
        if verbose:
            print(f"[{i}/{len(matrices)}] {entry['domain']}/{entry['name']} ...", flush=True)
        try:
            all_rows.extend(run_one_matrix(entry["domain"], entry["name"], entry["path"]))
        except Exception as e:  # matrix-level failure (bad file, out of memory, etc.)
            if verbose:
                print(f"    FAILED to process matrix: {e}")
            all_rows.append({"domain": entry["domain"], "matrix": entry["name"], "n": None,
                              "nnz": None, "method": "ALL", "type": "ALL",
                              "condition_number": None, "status": f"matrix_load_failed: {e}",
                              "runtime_sec": None, "iterations": None, "residual_abs": None,
                              "error_abs": None, "residual_rel": None, "error_rel": None})

    df = pd.DataFrame(all_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    if verbose:
        print(f"\nWrote {len(df)} rows to {output_csv}")
    return df
