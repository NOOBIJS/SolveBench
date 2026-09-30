# Loading matrices and building the ground-truth (x_true, b) pair.
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp


def load_matrix(mtx_path):
    # Load a Matrix Market file as a real-valued square CSR matrix.
    A = scipy.io.mmread(str(mtx_path))
    A = sp.csr_matrix(A, dtype=np.float64)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"{mtx_path} is not square: {A.shape}")
    return A


def make_ground_truth(A, seed=None):
    # Return (x_true, b) with b = A @ x_true; seed draws x_true random instead of ones.
    n = A.shape[0]
    if seed is None:
        x_true = np.ones(n, dtype=np.float64)
    else:
        x_true = np.random.default_rng(seed).standard_normal(n)
    return x_true, A @ x_true


def cancellation_ratio(A, x_true):
    # ||A x|| / |||A| |x|||: near machine epsilon, b is just rounding noise.
    num = float(np.linalg.norm(A @ x_true))
    den = float(np.linalg.norm(abs(A) @ np.abs(x_true)))
    return num / den if den > 0 else np.nan


def structural_singularity(A):
    # Count all-zero rows/columns -- no unique solution, and spilu OS-kills on these on Kaggle.
    zero_rows = int((np.diff(A.indptr) == 0).sum())
    zero_cols = int((np.diff(A.tocsc().indptr) == 0).sum())
    return zero_rows, zero_cols


def structural_rank_deficit(A):
    # n - structural_rank(A): stronger than structural_singularity, catches M10PI_n-style cases.
    from scipy.sparse.csgraph import structural_rank
    n = A.shape[0]
    try:
        return int(n - structural_rank(A.tocsr()))
    except Exception:
        return 0            # never let the screen itself be what fails the run


def discover_matrices(dataset_root):
    # Find every .mtx under the corpus root, flat or nested, as one entry each.
    root = Path(dataset_root)
    entries = []
    for domain_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for mtx_path in sorted(domain_dir.rglob("*.mtx")):
            entries.append({"domain": domain_dir.name,
                            "name": mtx_path.stem,
                            "path": mtx_path})
    return entries
