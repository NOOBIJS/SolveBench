"""Four direct solvers for Ax = b, implemented from scratch on dense arrays
with vectorized NumPy row operations (not element-by-element Python loops --
see the guide's runtime section for why that distinction matters at this scale).

Each solver raises SolverNotApplicable if the matrix doesn't meet its
requirements (e.g. Cholesky needs symmetric positive-definite).
"""
import numpy as np


class SolverNotApplicable(Exception):
    """Raised when a method's mathematical requirements aren't met by this matrix."""


def _partial_pivot(A: np.ndarray, b: np.ndarray, row: int) -> None:
    """In-place: swap row `row` with the row below it that has the largest
    pivot-column magnitude, to keep elimination numerically stable.
    """
    n = A.shape[0]
    pivot_row = row + int(np.argmax(np.abs(A[row:, row])))
    if abs(A[pivot_row, row]) < 1e-14:
        raise SolverNotApplicable("matrix is singular (zero pivot even after partial pivoting)")
    if pivot_row != row:
        A[[row, pivot_row]] = A[[pivot_row, row]]
        b[[row, pivot_row]] = b[[pivot_row, row]]


def gauss_elimination(A_dense: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, int]:
    """Forward elimination with partial pivoting, then back-substitution.

    Returns (x, steps), where steps is the number of elimination stages
    actually performed (n-1 for a full-size solve) -- a direct method's
    deterministic analogue of "iterations": fixed by matrix size, not by
    how close the current estimate is to converged.
    """
    n = A_dense.shape[0]
    A = A_dense.copy()
    b = b.copy()
    steps = 0

    for k in range(n - 1):
        _partial_pivot(A, b, k)
        factors = A[k + 1 :, k] / A[k, k]
        A[k + 1 :, k:] -= np.outer(factors, A[k, k:])
        b[k + 1 :] -= factors * b[k]
        steps += 1

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1 :] @ x[i + 1 :]) / A[i, i]
    return x, steps


def gauss_jordan(A_dense: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, int]:
    """Reduce all the way to the identity matrix (no separate back-substitution).

    Returns (x, steps): one elimination stage per row, so steps = n.
    """
    n = A_dense.shape[0]
    A = A_dense.copy()
    b = b.copy()
    steps = 0

    for k in range(n):
        _partial_pivot(A, b, k)
        pivot = A[k, k]
        A[k, :] /= pivot
        b[k] /= pivot
        pivot_col = A[:, k].copy()
        pivot_col[k] = 0.0
        A -= np.outer(pivot_col, A[k, :])
        b -= pivot_col * b[k]
        steps += 1

    return b, steps


def lu_solve(A_dense: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, int]:
    """Doolittle LU decomposition with partial pivoting (A = P^-1 L U), then
    solve via forward substitution (Ly=b) followed by back substitution (Ux=y).

    Returns (x, steps): steps = n-1 factorization stages (same structure as
    Gauss elimination, since LU is elimination with the multipliers kept).
    """
    n = A_dense.shape[0]
    U = A_dense.copy()
    L = np.eye(n)
    perm_b = b.copy()
    steps = 0

    for k in range(n - 1):
        pivot_row = k + int(np.argmax(np.abs(U[k:, k])))
        if abs(U[pivot_row, k]) < 1e-14:
            raise SolverNotApplicable("matrix is singular (zero pivot even after partial pivoting)")
        if pivot_row != k:
            U[[k, pivot_row]] = U[[pivot_row, k]]
            perm_b[[k, pivot_row]] = perm_b[[pivot_row, k]]
            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]
        factors = U[k + 1 :, k] / U[k, k]
        L[k + 1 :, k] = factors
        U[k + 1 :, k:] -= np.outer(factors, U[k, k:])
        steps += 1

    # Forward substitution: L y = perm_b
    y = np.zeros(n)
    for i in range(n):
        y[i] = perm_b[i] - L[i, :i] @ y[:i]

    # Back substitution: U x = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1 :] @ x[i + 1 :]) / U[i, i]
    return x, steps


def cholesky_solve(A_dense: np.ndarray, b: np.ndarray, sym_tol: float = 1e-8) -> tuple[np.ndarray, int]:
    """Cholesky decomposition (A = R^T R), valid only for symmetric
    positive-definite matrices. Raises SolverNotApplicable otherwise.

    Returns (x, steps): one factorization stage per row, so steps = n.
    """
    n = A_dense.shape[0]
    if not np.allclose(A_dense, A_dense.T, atol=sym_tol, rtol=sym_tol):
        raise SolverNotApplicable("matrix is not symmetric")

    R = np.zeros((n, n))
    steps = 0
    for i in range(n):
        diag_term = A_dense[i, i] - R[:i, i] @ R[:i, i]
        if diag_term <= 0:
            raise SolverNotApplicable("matrix is not positive-definite")
        R[i, i] = np.sqrt(diag_term)
        if i + 1 < n:
            R[i, i + 1 :] = (A_dense[i, i + 1 :] - R[:i, i] @ R[:i, i + 1 :]) / R[i, i]
        steps += 1

    # Solve R^T y = b (forward), then R x = y (back)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - R[:i, i] @ y[:i]
        y[i] /= R[i, i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - R[i, i + 1 :] @ x[i + 1 :]) / R[i, i]
    return x, steps
