# Spectral radii of the iteration matrices, and which classical hypothesis class applies.
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import config


def _diagonal(A):
    d = A.diagonal()
    return None if np.any(np.abs(d) < 1e-14) else d


def jacobi_operator(A):
    # T_J = I - D^-1 A, applied without forming it.
    d = _diagonal(A)
    if d is None:
        return None
    return spla.LinearOperator(A.shape, matvec=lambda v: v - (A @ v) / d, dtype=np.float64)


def gauss_seidel_operator(A):
    # T_GS = -(D + L)^-1 U, with the triangular factor built once.
    if _diagonal(A) is None:
        return None
    M = sp.tril(A, format="csc")
    U = sp.triu(A, k=1, format="csr")
    try:
        lu = spla.splu(M, permc_spec="NATURAL", diag_pivot_thresh=0.0)
    except RuntimeError:
        return None
    return spla.LinearOperator(A.shape, matvec=lambda v: -lu.solve(U @ v), dtype=np.float64)


def _power_iteration(op, n, rng=None):
    # Estimate the dominant eigenvalue magnitude by repeated application. Returns (rho, converged).
    rng = rng or np.random.default_rng(0)
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v) or 1.0
    rho = np.nan
    for _ in range(config.POWER_ITER_MAX):
        w = op @ v
        nw = np.linalg.norm(w)
        if not np.isfinite(nw):
            return np.inf, False
        if nw == 0.0:
            return 0.0, True
        prev, rho = rho, nw
        v = w / nw
        if np.isfinite(prev) and abs(rho - prev) <= config.POWER_ITER_TOL * max(rho, 1e-30):
            return float(rho), True
    return float(rho), False


def dense_iteration_matrix(A, kind="jacobi"):
    # Build the iteration matrix densely, all columns at once rather than one by one.
    d = _diagonal(A)
    if d is None:
        return None
    n = A.shape[0]
    if kind == "jacobi":
        return np.eye(n) - A.toarray() / d[:, None]

    M = sp.tril(A, format="csc")
    U = sp.triu(A, k=1, format="csr")
    try:
        lu = spla.splu(M, permc_spec="NATURAL", diag_pivot_thresh=0.0)
    except RuntimeError:
        return None
    return -lu.solve(U.toarray())          # splu.solve takes all columns at once


def spectral_radius(A, kind="jacobi", exact_cap=None):
    # rho of an iteration matrix. Returns (rho, how), how = exact_eig/arpack/power_iteration.
    exact_cap = config.SPECTRAL_EXACT_CAP if exact_cap is None else exact_cap
    n = A.shape[0]
    op = jacobi_operator(A) if kind == "jacobi" else gauss_seidel_operator(A)
    if op is None:
        return np.nan, "not_applicable"

    if n <= exact_cap:
        T = dense_iteration_matrix(A, kind)
        try:
            if T is not None:
                return float(np.max(np.abs(np.linalg.eigvals(T)))), "exact_eig"
        except (np.linalg.LinAlgError, MemoryError):
            pass

    try:
        vals = spla.eigs(op, k=1, which="LM", return_eigenvectors=False,
                         maxiter=config.POWER_ITER_MAX * 10, tol=config.POWER_ITER_TOL)
        return float(np.abs(vals[0])), "arpack"
    except Exception:
        rho, ok = _power_iteration(op, n)
        return rho, "power_iteration" if ok else "power_iteration_unconverged"


def iterations_to_tolerance(rho, tol=None):
    # How many iterations rho^k <= tol needs (asymptotic rate only); infinite when rho >= 1.
    tol = config.TOLERANCE if tol is None else tol
    if not np.isfinite(rho) or rho >= 1.0:
        return np.inf
    if rho <= 0.0:
        return 1.0
    return float(np.log(tol) / np.log(rho))


def convergence_verdict(rho, tol=None, budget=None):
    # Four-way verdict: not_applicable (zero diagonal), diverges, too_slow, or converges.
    budget = config.MAX_ITERATIONS if budget is None else budget
    if rho is None or (isinstance(rho, float) and np.isnan(rho)):
        return "not_applicable"
    if not np.isfinite(rho) or rho >= 1.0:      # +inf is genuine divergence
        return "diverges"
    return "converges" if iterations_to_tolerance(rho, tol) <= budget else "too_slow"


def comparison_matrix(A):
    # M(A): |a_ii| on the diagonal, -|a_ij| off it -- A is H-matrix iff M(A) is M-matrix.
    M = -abs(A).tolil()
    M.setdiag(np.abs(A.diagonal()))
    return M.tocsr()


def is_strictly_diagonally_dominant(A):
    d = np.abs(A.diagonal())
    off = np.asarray(abs(A).sum(axis=1)).ravel() - d
    return bool(np.all(d > off))


def is_weakly_diagonally_dominant(A):
    d = np.abs(A.diagonal())
    off = np.asarray(abs(A).sum(axis=1)).ravel() - d
    return bool(np.all(d >= off) and np.any(d > off))


def is_l_matrix(A):
    d = A.diagonal()
    if np.any(d <= 0):
        return False
    off = A - sp.diags(d)
    return bool(off.nnz == 0 or off.max() <= 0)


def is_symmetric(A, tol=1e-8):
    diff = abs(A - A.T)
    return bool(diff.nnz == 0 or diff.max() <= tol * max(abs(A).max(), 1.0))


def is_spd(A):
    # Symmetric with a successful sparse Cholesky-equivalent factorization.
    if not is_symmetric(A):
        return False
    try:
        vals = spla.eigsh(A.astype(np.float64), k=1, which="SA",
                          return_eigenvectors=False, maxiter=5000)
        return bool(vals[0] > 0)
    except Exception:
        try:
            np.linalg.cholesky(A.toarray())
            return True
        except (np.linalg.LinAlgError, MemoryError, ValueError):
            return False


def classify(A, rho_jacobi=None, rho_gs=None):
    # Label a matrix with every hypothesis class it satisfies, plus one primary label.
    flags = {
        "strict_diag_dominant": is_strictly_diagonally_dominant(A),
        "weak_diag_dominant": is_weakly_diagonally_dominant(A),
        "l_matrix": is_l_matrix(A),
        "symmetric": is_symmetric(A),
        "spd": is_spd(A),
    }

    if rho_jacobi is None:
        rho_jacobi, _ = spectral_radius(A, "jacobi")
    if rho_gs is None:
        rho_gs, _ = spectral_radius(A, "gauss_seidel")

    rho_comparison, _ = spectral_radius(comparison_matrix(A), "jacobi")
    flags["h_matrix"] = bool(np.isfinite(rho_comparison) and rho_comparison < 1.0)

    flags["m_matrix"] = bool(flags["l_matrix"] and np.isfinite(rho_jacobi) and rho_jacobi < 1.0)

    # property A proxy: Young's rho(T_GS) = rho(T_J)^2 holding numerically
    flags["property_a_proxy"] = bool(
        np.isfinite(rho_jacobi) and np.isfinite(rho_gs) and rho_jacobi > 0
        and abs(rho_gs - rho_jacobi ** 2) <= 1e-3 * max(rho_jacobi ** 2, 1e-12)
    )

    for label in ("m_matrix", "spd", "strict_diag_dominant", "h_matrix",
                  "l_matrix", "weak_diag_dominant", "property_a_proxy"):
        if flags[label]:
            primary = label
            break
    else:
        primary = "none"

    return {**flags,
            "rho_jacobi": rho_jacobi,
            "rho_gauss_seidel": rho_gs,
            "rho_comparison": rho_comparison,
            "hypothesis_class": primary,
            "jacobi_converges_predicted": bool(np.isfinite(rho_jacobi) and rho_jacobi < 1.0),
            "gs_converges_predicted": bool(np.isfinite(rho_gs) and rho_gs < 1.0),
            "jacobi_verdict": convergence_verdict(rho_jacobi),
            "gs_verdict": convergence_verdict(rho_gs),
            "jacobi_iterations_needed": iterations_to_tolerance(rho_jacobi),
            "gs_iterations_needed": iterations_to_tolerance(rho_gs),
            "optimal_omega": (2.0 / (1.0 + np.sqrt(1.0 - rho_jacobi ** 2))
                              if np.isfinite(rho_jacobi) and rho_jacobi < 1.0 else np.nan)}
