# Iterative refinement (Wilkinson 1963), applied the same way to every solver.
import numpy as np

from .iterative_solvers import Work


def refine(solver, A, b, passes=1):
    # Run `solver` on (A, b), then apply `passes` correction steps, cost summed in.
    x, iterations, converged, work = solver(A, b)
    if passes <= 0 or x is None or not np.all(np.isfinite(x)):
        return x, iterations, converged, work

    total = Work()
    total.matvecs = work.matvecs
    total.tri_solves = work.tri_solves
    total.precond = work.precond
    total.setup = work.setup

    b_norm = np.linalg.norm(b) or 1.0
    for _ in range(passes):
        r = b - A @ x
        total.matvecs += 1
        if not np.all(np.isfinite(r)):
            break
        # nothing left to correct -- a zero residual would make CG report a false failure
        if np.linalg.norm(r) <= np.finfo(float).eps * b_norm:
            break
        d, its, conv, w = solver(A, r)
        total.matvecs += w.matvecs
        total.tri_solves += w.tri_solves
        total.precond += w.precond
        total.setup += w.setup
        iterations += its
        if d is None or not np.all(np.isfinite(d)):
            break
        x_next = x + d
        if not np.all(np.isfinite(x_next)):
            break
        x = x_next
        converged = converged or conv

    return x, iterations, converged, total
