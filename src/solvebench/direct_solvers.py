# Four direct solvers for Ax = b, from scratch on dense arrays with vectorized NumPy rows.
import numpy as np


class SolverNotApplicable(Exception):
    pass  # matrix doesn't meet this method's requirements (e.g. Cholesky needs SPD)


def _partial_pivot(A: np.ndarray, b: np.ndarray, row: int) -> None:
    # In-place: swap in the row below with the largest pivot-column magnitude.
    n = A.shape[0]
    pivot_row = row + int(np.argmax(np.abs(A[row:, row])))
    if abs(A[pivot_row, row]) < 1e-14:
        raise SolverNotApplicable("matrix is singular (zero pivot even after partial pivoting)")
    if pivot_row != row:
        A[[row, pivot_row]] = A[[pivot_row, row]]
        b[[row, pivot_row]] = b[[pivot_row, row]]


def gauss_elimination(A_dense: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, int]:
    # Forward elimination with partial pivoting, then back-substitution.
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
    # Reduce all the way to the identity matrix, no separate back-substitution.
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
    # Doolittle LU with partial pivoting, then forward then back substitution.
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
    # Cholesky (A = R^T R), valid only for symmetric positive-definite matrices.
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
