"""Every tunable number in the benchmark, in one place.

The notebook used to carry its own copy of these values, which is how the
released library ended up capping direct methods at n=3000 while the run that
produced the results capped them at n=2000. Nothing outside this module is
allowed to hard-code a limit or a tolerance.
"""

# --- iterative solvers -------------------------------------------------------
MAX_ITERATIONS = 10_000     # cap before a stationary/Krylov method is called non-convergent
TOLERANCE = 1e-8            # relative residual an iterative method is asked to reach
SOR_OMEGA = 1.25            # fixed relaxation factor (1.0 would reduce SOR to Gauss-Seidel)

# A run whose residual climbs past this multiple of its own best is abandoned
# early: it is diverging, and letting it burn the full iteration cap teaches us
# nothing. 2,022,191 of the iterations in the previous sweep were spent this way.
DIVERGENCE_GROWTH = 1e4

# --- iterative refinement ----------------------------------------------------
# Refinement is an ORTHOGONAL factor, not a property of one solver. The previous
# run applied it only to APK, which is why APK appeared to be the most accurate
# method: the same wrapper around Gauss-Seidel reaches 5.44e-16, about 27x more
# accurate than refined APK's 1.45e-14. Every solver is now measured at each of
# these pass counts so the comparison is like-for-like.
REFINEMENT_PASSES = (0, 1, 2)

# --- ILU preconditioner ------------------------------------------------------
ILU_DROP_TOL = 1e-3
ILU_FILL_FACTOR = 5

# --- direct solvers ----------------------------------------------------------
# The direct solvers are hand-written rather than LAPACK calls, so they cost
# O(n^3) in Python: roughly 50-100x slower than numpy.linalg.solve, measured at
# ~35s for n=3000. They are pedagogical implementations validated against LAPACK,
# not performance competitors, and are capped so a sweep stays tractable.
DIRECT_SIZE_CAP = 2000

# --- condition number --------------------------------------------------------
COND_EXACT_CAP = 2000       # above this, fall back to a 1-norm estimate
# Beyond this the matrix is numerically singular in double precision and any
# forward error reported for it says more about the matrix than about the solver.
# 160 of 930 matrices in this collection are past it, and they are reported as a
# separate stratum rather than pooled into the headline error statistics.
ILL_CONDITIONED = 1e15

# --- spectral analysis -------------------------------------------------------
# Above this size we estimate rho(T) by power iteration instead of forming the
# iteration matrix; 611 of 930 matrices are small enough for the exact route.
SPECTRAL_EXACT_CAP = 2000
POWER_ITER_MAX = 200
POWER_ITER_TOL = 1e-6
