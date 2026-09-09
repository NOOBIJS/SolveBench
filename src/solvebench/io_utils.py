"""Loading matrices and building the ground-truth (x_true, b) pair.

The corpus is laid out flat as ``<domain>/<name>.mtx``. An earlier version of
this module expected the nested ``<domain>/<name>/<name>.mtx`` shape the pilot
dataset used, so it silently found nothing in the real corpus -- one of the ways
the library and the notebook that produced the results had drifted apart.
Both layouts are accepted now.
"""
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp


def load_matrix(mtx_path):
    """Load a Matrix Market file as a real-valued square CSR matrix."""
    A = scipy.io.mmread(str(mtx_path))
    A = sp.csr_matrix(A, dtype=np.float64)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"{mtx_path} is not square: {A.shape}")
    return A


def make_ground_truth(A, seed=None):
    """Return (x_true, b) with b = A @ x_true, so 'correct' is unambiguous.

    With `seed` unset, x_true is the vector of ones -- deterministic and easy to
    reason about, but a special vector: it is in the null space of any matrix
    whose rows sum to zero, and it makes b a plain row-sum, which for a handful
    of matrices in this corpus is almost entirely cancellation (the worst has
    ||A.1|| / |||A|.1|| = 3.5e-17, meaning b is essentially rounding noise).
    Passing a seed draws x_true from N(0,1) instead, which is how the replicate
    runs check that no result depends on that choice.
    """
    n = A.shape[0]
    if seed is None:
        x_true = np.ones(n, dtype=np.float64)
    else:
        x_true = np.random.default_rng(seed).standard_normal(n)
    return x_true, A @ x_true


def cancellation_ratio(A, x_true):
    """||A x|| / |||A| |x|||: how much of b survives cancellation.

    Near machine epsilon, b carries almost no information about A and no solver
    can be judged on it. Reported per matrix so those systems can be excluded
    from accuracy statistics rather than quietly distorting them.
    """
    num = float(np.linalg.norm(A @ x_true))
    den = float(np.linalg.norm(abs(A) @ np.abs(x_true)))
    return num / den if den > 0 else np.nan


def structural_singularity(A):
    """Count all-zero rows and columns.

    Such a matrix has no unique solution, so every solver result on it is
    meaningless. Screening is also a hard robustness requirement: SuperLU's
    spilu raises a catchable error on these under scipy 1.17 but exhausts memory
    and gets the process OS-killed under Kaggle's scipy 1.16.3. Two full runs
    died this way, both at M40PI_n1. 19 of 930 matrices are affected.
    """
    zero_rows = int((np.diff(A.indptr) == 0).sum())
    zero_cols = int((np.diff(A.tocsc().indptr) == 0).sum())
    return zero_rows, zero_cols


def discover_matrices(dataset_root):
    """Find every .mtx under the corpus root, flat or nested, as one entry each."""
    root = Path(dataset_root)
    entries = []
    for domain_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for mtx_path in sorted(domain_dir.rglob("*.mtx")):
            entries.append({"domain": domain_dir.name,
                            "name": mtx_path.stem,
                            "path": mtx_path})
    return entries
