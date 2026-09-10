"""Spectral radii of the iteration matrices, and hypothesis-class labels.

This is the quantity the base paper exists to characterise, and the first sweep
never computed it. Khrapov and Volkov derive exactly where rho(T) < 1 holds for
2 and 3 unknowns; an extension of their work that measures convergence
empirically but never looks at rho has no way to say *why* a method converged.

Two things are computed here.

**Spectral radius.** For the Jacobi iteration matrix T_J = I - D^-1 A and the
Gauss-Seidel iteration matrix T_GS = -(D+L)^-1 U. A stationary method converges
from any starting vector if and only if rho(T) < 1, so this predicts the
outcome the benchmark measures, and the two can be compared.

**Hypothesis class.** Which classical theorem, if any, covers the matrix:

* *Strict diagonal dominance* -- both Jacobi and Gauss-Seidel converge.
* *L-matrix* (positive diagonal, non-positive off-diagonal). With rho(T_J) < 1
  this is an M-matrix, and Stein and Rosenberg (1948) then give
  rho(T_GS) < rho(T_J) < 1: Gauss-Seidel converges whenever Jacobi does, and
  strictly faster. "Jacobi converges but Gauss-Seidel does not" is impossible
  here -- which is why finding zero such cases in real data is not a discovery.
* *H-matrix* -- rho of the comparison matrix's Jacobi iteration is below 1;
  both methods converge.
* *SPD* -- Householder (1958) and John: Gauss-Seidel always converges.
* *Property A / consistently ordered* -- Young (1950) gives
  rho(T_GS) = rho(T_J)^2 and an optimal relaxation factor
  omega* = 2 / (1 + sqrt(1 - rho(T_J)^2)). Detected here by the numerical
  identity rather than by inspecting the ordering, so it is a proxy.

Coverage of these classes across the corpus is the measurement that explains the
benchmark's own headline result, and it is what any referee would ask for first.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import config


def _diagonal(A):
    d = A.diagonal()
    return None if np.any(np.abs(d) < 1e-14) else d


def jacobi_operator(A):
    """T_J = I - D^-1 A, applied without forming it."""
    d = _diagonal(A)
    if d is None:
        return None
    return spla.LinearOperator(A.shape, matvec=lambda v: v - (A @ v) / d, dtype=np.float64)


def gauss_seidel_operator(A):
    """T_GS = -(D + L)^-1 U, with the triangular factor built once."""
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
    """Estimate the dominant eigenvalue magnitude by repeated application.

    Returns (rho, converged). Reported as an estimate, never as an exact value:
    with a complex dominant pair the ratio oscillates and settles on the modulus
    more slowly than for a real dominant eigenvalue.
    """
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
    """Build the iteration matrix densely, in one shot rather than column by column.

    Applying the operator to n basis vectors in a Python loop costs the same
    arithmetic but pays n rounds of interpreter and dispatch overhead, which is
    the difference between a usable sweep over 611 matrices and an unusable one.
    Both forms below hand the whole right-hand side over at once instead.
    """
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
    """rho of an iteration matrix. Returns (rho, how).

    `how` records which route produced the number -- "exact_eig" from a dense
    eigendecomposition, "arpack" or "power_iteration" otherwise -- so results
    computed by different routes are never silently pooled.
    """
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
    """How many iterations rho^k <= tol needs. Infinite when rho >= 1.

    The asymptotic rate only, ignoring the transient, so it is a lower bound on
    what a real run costs rather than a prediction of it.
    """
    tol = config.TOLERANCE if tol is None else tol
    if not np.isfinite(rho) or rho >= 1.0:
        return np.inf
    if rho <= 0.0:
        return 1.0
    return float(np.log(tol) / np.log(rho))


def convergence_verdict(rho, tol=None, budget=None):
    """Three-way prediction, where the textbook criterion gives only two.

    Convergence theory asks a yes/no question -- is rho < 1 -- but a matrix can
    satisfy it and still be useless. nos7 in this corpus has rho(T_J) =
    0.999999984536822: strictly below 1, so Jacobi provably converges, and it
    would take about 1.19e9 iterations to reach 1e-8 against an iteration cap of
    10,000. Reporting that as a prediction failure would be wrong, and reporting
    it as "converges" would be misleading. It is a third case.

    A fourth case has to be kept apart from these three. When the diagonal carries a
    zero the iteration matrix does not exist, so rho is NaN -- and "the method is
    undefined here" is not the same statement as "it would diverge". Conflating them put
    491 systems into the diverging column for Jacobi, which is most of that column.

    Returns "not_applicable", "diverges", "too_slow", or "converges".
    """
    budget = config.MAX_ITERATIONS if budget is None else budget
    if rho is None or (isinstance(rho, float) and np.isnan(rho)):
        return "not_applicable"
    if not np.isfinite(rho) or rho >= 1.0:      # +inf is genuine divergence
        return "diverges"
    return "converges" if iterations_to_tolerance(rho, tol) <= budget else "too_slow"


def comparison_matrix(A):
    """M(A): |a_ii| on the diagonal, -|a_ij| off it. A is an H-matrix exactly
    when M(A) is an M-matrix, which is what the H-matrix test below checks."""
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
    """Symmetric with a successful sparse Cholesky-equivalent factorization."""
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
    """Label one matrix with every hypothesis class it satisfies.

    Returns the individual flags plus `hypothesis_class`, a single label chosen
    by the precedence below for reporting convenience. The flags are the real
    output -- a matrix can belong to several classes at once, and collapsing
    that to one label loses information.
    """
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

    # H-matrix: rho of the comparison matrix's Jacobi iteration below 1.
    rho_comparison, _ = spectral_radius(comparison_matrix(A), "jacobi")
    flags["h_matrix"] = bool(np.isfinite(rho_comparison) and rho_comparison < 1.0)

    # M-matrix: an L-matrix whose Jacobi iteration converges. This is the class
    # in which Stein-Rosenberg forbids "Jacobi converges, Gauss-Seidel does not".
    flags["m_matrix"] = bool(flags["l_matrix"] and np.isfinite(rho_jacobi) and rho_jacobi < 1.0)

    # Property A proxy: Young's rho(T_GS) = rho(T_J)^2 holding numerically.
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
            # Theory's binary answer, kept because it is what the classical
            # theorems actually state ...
            "jacobi_converges_predicted": bool(np.isfinite(rho_jacobi) and rho_jacobi < 1.0),
            "gs_converges_predicted": bool(np.isfinite(rho_gs) and rho_gs < 1.0),
            # ... and the three-way verdict, which is what the benchmark can
            # observe within a finite iteration budget.
            "jacobi_verdict": convergence_verdict(rho_jacobi),
            "gs_verdict": convergence_verdict(rho_gs),
            "jacobi_iterations_needed": iterations_to_tolerance(rho_jacobi),
            "gs_iterations_needed": iterations_to_tolerance(rho_gs),
            "optimal_omega": (2.0 / (1.0 + np.sqrt(1.0 - rho_jacobi ** 2))
                              if np.isfinite(rho_jacobi) and rho_jacobi < 1.0 else np.nan)}
