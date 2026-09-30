# Every tunable number in the benchmark, kept in one place.

# --- iterative solvers -------------------------------------------------------
MAX_ITERATIONS = 10_000     # cap before a stationary/Krylov method is called non-convergent
TOLERANCE = 1e-8            # relative residual an iterative method is asked to reach
SOR_OMEGA = 1.25            # fixed relaxation factor (1.0 would reduce SOR to Gauss-Seidel)

POWER_ITERS_OMEGA = 60      # power iterations to estimate rho(T_J) before picking omega

# raised from 1e4 -- that value was killing real convergences mid-transient, e.g. cdde6
DIVERGENCE_GROWTH = 1e12

# --- iterative refinement ----------------------------------------------------
# orthogonal to the solver -- giving it to one method only is what made APK look best before
REFINEMENT_PASSES = (0, 1)
REFINEMENT_STUDY_PASSES = (0, 1, 2)     # third pass barely moves the needle, checked on fs_541_2

# --- ILU preconditioner ------------------------------------------------------
ILU_DROP_TOL = 1e-3
ILU_FILL_FACTOR = 5

# retry ladder, pushes toward robustness each step -- see tools/audit_ilu_hole.py
ILU_LADDER = ((ILU_DROP_TOL, ILU_FILL_FACTOR),
              (1e-4, 10),
              (1e-6, 20),
              (0.0, 50))

# --- direct solvers ----------------------------------------------------------
DIRECT_SIZE_CAP = 2000      # hand-written, O(n^3) in python, ~35s at n=3000

# --- condition number --------------------------------------------------------
COND_EXACT_CAP = 2000       # above this, fall back to a 1-norm estimate
ILL_CONDITIONED = 1e15      # numerically singular past here, reported as its own stratum

# --- spectral analysis -------------------------------------------------------
SPECTRAL_EXACT_CAP = 2000   # above this estimate rho(T) by power iteration instead
POWER_ITER_MAX = 200
POWER_ITER_TOL = 1e-6
