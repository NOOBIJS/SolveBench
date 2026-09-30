# The single place a solve gets judged correct or not -- from the residual, not the solver's own claim.
import numpy as np

# relative residual ||Ax - b|| / ||b|| below which a solve counts as successful
SUCCESS_TOL = 1e-8

# relative forward error, reported alongside but never used to gate success (ill-conditioning)
ACCURATE_TOL = 1e-6

STATUS_SOLVED = "solved"                    # residual verified below SUCCESS_TOL
STATUS_INACCURATE = "inaccurate"            # solver CLAIMED success; the residual says no
STATUS_NO_CONVERGE = "did_not_converge"     # solver admitted failure, residual agrees
STATUS_DIVERGED = "diverged"                # iterate blew up or went non-finite
STATUS_NOT_APPLICABLE = "not_applicable"    # method undefined on this matrix
STATUS_SINGULAR = "structurally_singular"   # matrix has no unique solution
STATUS_SKIPPED = "skipped_too_large"        # deliberately not attempted at this size
STATUS_ERROR = "error"                      # unexpected failure, see note

# the method was defined on this matrix and got to try -- denominator for conditional success
APPLICABLE_STATUSES = frozenset({STATUS_SOLVED, STATUS_INACCURATE,
                                 STATUS_NO_CONVERGE, STATUS_DIVERGED})


def score(A, x, b, x_true, b_norm=None, x_true_norm=None, reported_converged=None):
    # Measure one solve and decide status; reported_converged is stored, never trusted.
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
        status = STATUS_INACCURATE      # solver claimed success (or stayed silent) and is wrong

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
    # A metric row for a solve that never happened (skipped, singular, N/A).
    row = {"status": status, "residual_abs": np.nan, "error_abs": np.nan,
           "residual_rel": np.nan, "error_rel": np.nan,
           "reported_converged": None, "accurate": False}
    if note is not None:
        row["note"] = note
    return row


def rates(df, method):
    # Applicability and conditional success, kept separate -- never multiply them together.
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
