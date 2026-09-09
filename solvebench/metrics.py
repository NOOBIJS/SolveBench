"""The single place where a solve is judged correct or not.

Every method -- direct or iterative, hand-written or library -- is scored here
and nowhere else. This matters more than it looks. The earlier harness used two
different definitions of success: direct methods were marked "ok" whenever the
call did not raise, and iterative methods were marked "ok" whenever the solver
returned its own convergence flag. Neither one looked at the answer. That let
Gauss-Jordan report success on 57 systems whose relative residual reached 9.7e12.

The rule here is deliberately blunt: a solve succeeded if the residual we
measure ourselves, afterwards, from A, x and b, is small. A solver's opinion of
its own convergence is recorded (see `reported_converged`) but never decides
the outcome.
"""
import numpy as np

# Relative residual ||Ax - b|| / ||b|| below which a solve counts as successful.
# Matches the tolerance the iterative solvers are asked to reach, so a converged
# iterative solve and a good direct solve are held to the same standard.
SUCCESS_TOL = 1e-8

# Relative forward error ||x - x_true|| / ||x_true|| reported alongside, but NOT
# used to gate success: on ill-conditioned systems a perfectly good solver can
# have a large forward error through no fault of its own. It is a separate
# column so both views can be reported, never a hidden second criterion.
ACCURATE_TOL = 1e-6

STATUS_SOLVED = "solved"                    # residual verified below SUCCESS_TOL
STATUS_INACCURATE = "inaccurate"            # solver CLAIMED success; the residual says no
STATUS_NO_CONVERGE = "did_not_converge"     # solver admitted failure, residual agrees
STATUS_DIVERGED = "diverged"                # iterate blew up or went non-finite
STATUS_NOT_APPLICABLE = "not_applicable"    # method undefined on this matrix
STATUS_SINGULAR = "structurally_singular"   # matrix has no unique solution
STATUS_SKIPPED = "skipped_too_large"        # deliberately not attempted at this size
STATUS_ERROR = "error"                      # unexpected failure, see note

#: Statuses that mean "the method was defined on this matrix and we let it try".
#: The denominator for a conditional success rate is exactly this set.
APPLICABLE_STATUSES = frozenset({STATUS_SOLVED, STATUS_INACCURATE,
                                 STATUS_NO_CONVERGE, STATUS_DIVERGED})


def score(A, x, b, x_true, b_norm=None, x_true_norm=None, reported_converged=None):
    """Measure one solve and decide whether it succeeded.

    Returns the metric columns plus a status. `reported_converged` is whatever
    the solver claimed about itself; it is stored for comparison but has no
    influence on the verdict.
    """
    b_norm = b_norm if b_norm else (np.linalg.norm(b) or 1.0)
    x_true_norm = x_true_norm if x_true_norm else (np.linalg.norm(x_true) or 1.0)

    if x is None or not np.all(np.isfinite(x)):
        return {"status": STATUS_DIVERGED, "residual_abs": np.nan, "error_abs": np.nan,
                "residual_rel": np.nan, "error_rel": np.nan,
                "reported_converged": reported_converged, "accurate": False}

    residual_abs = float(np.linalg.norm(A @ x - b))
    error_abs = float(np.linalg.norm(x - x_true))
    residual_rel = residual_abs / b_norm
    error_rel = error_abs / x_true_norm

    if residual_rel <= SUCCESS_TOL:
        status = STATUS_SOLVED
    elif reported_converged is False:
        status = STATUS_NO_CONVERGE     # the solver said so itself, and it was right
    else:
        # The solver either claimed convergence or, like the hand-written direct
        # methods, simply returned without complaint -- and the answer is wrong.
        # This is the category that was invisible before: 57 Gauss-Jordan solves
        # were counted as successes here, one of them at a relative residual of
        # 9.71e+12.
        status = STATUS_INACCURATE

    return {
        "status": status,
        "residual_abs": residual_abs,
        "error_abs": error_abs,
        "residual_rel": residual_rel,
        "error_rel": error_rel,
        "reported_converged": reported_converged,
        "accurate": bool(error_rel <= ACCURATE_TOL),
    }


def blank(status, note=None):
    """A metric row for a solve that never happened (skipped, singular, N/A)."""
    row = {"status": status, "residual_abs": np.nan, "error_abs": np.nan,
           "residual_rel": np.nan, "error_rel": np.nan,
           "reported_converged": None, "accurate": False}
    if note is not None:
        row["note"] = note
    return row


def rates(df, method):
    """Applicability and conditional success for one method, as separate numbers.

    These must never be multiplied together into a single "success rate": doing
    that is what produced the earlier 10.4% figure for Conjugate Gradient, whose
    real conditional success is 83.9%. Report both, always with the denominator.
    """
    rows = df[df["method"] == method]
    total = len(rows)
    applicable = rows["status"].isin(APPLICABLE_STATUSES).sum()
    solved = (rows["status"] == STATUS_SOLVED).sum()
    return {
        "method": method,
        "corpus": total,
        "applicable": int(applicable),
        "solved": int(solved),
        "applicability": applicable / total if total else float("nan"),
        "conditional_success": solved / applicable if applicable else float("nan"),
    }
