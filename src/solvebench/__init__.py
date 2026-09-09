"""SolveBench: direct and iterative linear solvers benchmarked on real sparse matrices.

This package is the single source of truth for what the benchmark runs. The
Kaggle notebook is generated from it by ``tools/build_notebook.py`` rather than
carrying its own copy of the solvers, which is how the notebook and this library
previously drifted apart -- different size caps, different solver sets, and a
dataset layout the library could not even find.

Solver contract: every method returns ``(x, iterations, reported_converged, work)``.
The solver's own convergence claim is recorded but never decides the outcome;
:func:`solvebench.metrics.score` does that, from the residual it measures itself.
"""
from . import (benchmark, config, direct_solvers, io_utils, iterative_solvers,
               metrics, reference_solvers, refinement, spectral)
from .benchmark import METHODS, METHOD_NAMES, run_full_benchmark, run_one_matrix, summarise
from .direct_solvers import SolverNotApplicable
from .io_utils import discover_matrices, load_matrix, make_ground_truth
from .metrics import score
from .refinement import refine

__all__ = [
    "benchmark", "config", "direct_solvers", "io_utils", "iterative_solvers",
    "metrics", "reference_solvers", "refinement", "spectral",
    "METHODS", "METHOD_NAMES", "run_full_benchmark", "run_one_matrix", "summarise",
    "SolverNotApplicable", "discover_matrices", "load_matrix", "make_ground_truth",
    "score", "refine",
]
