"""Iterative refinement as an orthogonal factor, available to every solver.

In the first sweep refinement was applied only to the proposed method, and that
single asymmetry produced its headline accuracy result: refined, it reached a
median error of 1.45e-14 against unrefined baselines. The same wrapper placed
around Gauss-Seidel reaches 5.44e-16 -- about 27 times more accurate. The
accuracy ranking was measuring the wrapper, not the solver.

So refinement lives here, outside any solver, and the harness runs every method
at each pass count in ``config.REFINEMENT_PASSES``. The extra cost is real and
is counted: each pass is a full additional solve of the residual equation.

The procedure is the classical one (Wilkinson 1963): solve, form the residual,
solve the residual equation with the same method, correct. Working in the same
precision throughout, it cannot repair a badly conditioned system, but it does
recover the accuracy an early-stopped iterative solve leaves on the table.
"""
import numpy as np

from .iterative_solvers import Work


def refine(solver, A, b, passes=1):
    """Run `solver` on (A, b), then apply `passes` correction steps.

    `solver` is any callable with the project's ``(x, iterations, converged,
    work)`` contract. Returns the same tuple, with iterations and work summed
    across the initial solve and every correction, so a refined run is never
    credited with the cost of an unrefined one.
    """
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
        # Nothing left to correct. Handing a zero right-hand side to the solver
        # is not merely wasteful: CG's first search direction is then the zero
        # vector, p'Ap is exactly 0, and it reports the matrix as indefinite --
        # so an exact solve would be recorded as a solver failure.
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
