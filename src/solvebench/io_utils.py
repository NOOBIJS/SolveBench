"""Load SuiteSparse .mtx matrices and build the ground-truth (x_true, b) pair used
by every solver in the benchmark: x_true = ones(n), b = A @ x_true.
"""
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp


def load_matrix(mtx_path: Path) -> sp.csr_matrix:
    """Load a Matrix Market file as a sparse CSR matrix (real-valued, square)."""
    A = scipy.io.mmread(str(mtx_path))
    A = sp.csr_matrix(A, dtype=np.float64)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"{mtx_path} is not square: {A.shape}")
    return A


def make_ground_truth(A: sp.csr_matrix, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """x_true = vector of ones; b = A @ x_true. Deterministic, no ambiguity in
    what 'correct' means -- exactly the ground-truth trick described in the guide.
    """
    n = A.shape[0]
    x_true = np.ones(n, dtype=np.float64)
    b = A @ x_true
    return x_true, b


def discover_matrices(dataset_root: Path) -> list[dict]:
    """Walk dataset/<domain>/<name>/<name>.mtx and return one entry per matrix."""
    entries = []
    for domain_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        domain = domain_dir.name
        for mat_dir in sorted(p for p in domain_dir.iterdir() if p.is_dir()):
            mtx_path = mat_dir / f"{mat_dir.name}.mtx"
            if mtx_path.exists():
                entries.append({"domain": domain, "name": mat_dir.name, "path": mtx_path})
    return entries
