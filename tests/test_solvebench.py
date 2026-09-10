"""Local checks to run before anything is pushed to Kaggle.

Kaggle runs a different scipy than most local setups (1.16.3 there against
1.17.x here), and that difference has already killed two full sweeps: spilu on a
structurally singular matrix raises a catchable error locally but exhausts
memory and gets the process OS-killed on Kaggle. These tests will not catch an
environment difference on their own, but they do catch the class of bug that
wasted the most time -- a harness that reports success without checking, or a
solver whose rewrite quietly changed its answer.

    python -m pytest tests/ -q
"""
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from solvebench import benchmark, direct_solvers as direct, io_utils
from solvebench import iterative_solvers as it, metrics, reference_solvers as ref, spectral
from solvebench.direct_solvers import SolverNotApplicable
from solvebench.refinement import refine


@pytest.fixture
def tiny():
    """A hand-checkable diagonally dominant 3x3 with solution [1, 1, 1]."""
    A = sp.csr_matrix(np.array([[10.0, 1.0, 1.0],
                                [1.0, 10.0, 1.0],
                                [1.0, 1.0, 10.0]]))
    x_true = np.ones(3)
    return A, A @ x_true, x_true


@pytest.fixture
def unsymmetric():
    """Jacobi diverges here, Gauss-Seidel converges -- the interesting case."""
    rng = np.random.default_rng(7)
    n = 60
    A = sp.random(n, n, density=0.1, random_state=3, format="csr")
    A = (A + sp.diags(np.abs(A).sum(axis=1).A1 * 0.55 + 0.1)).tocsr()
    x_true = rng.standard_normal(n)
    return A, A @ x_true, x_true


# ---------------------------------------------------------------- solvers


@pytest.mark.parametrize("solver", [it.jacobi, it.gauss_seidel, it.sor,
                                    it.conjugate_gradient, it.bicgstab])
def test_iterative_solves_tiny_system(tiny, solver):
    A, b, x_true = tiny
    x, _, _, _ = solver(A, b)
    assert np.allclose(x, x_true, atol=1e-6)


@pytest.mark.parametrize("solver", [direct.gauss_elimination, direct.gauss_jordan,
                                    direct.lu_solve, direct.cholesky_solve])
def test_direct_matches_lapack(tiny, solver):
    A, b, _ = tiny
    x, _ = solver(A.toarray(), b)
    assert np.allclose(x, np.linalg.solve(A.toarray(), b), atol=1e-10)


@pytest.mark.parametrize("solver", [ref.sparse_spsolve, ref.sparse_lu,
                                    ref.ilu_bicgstab, ref.ilu_gmres,
                                    ref.ilu_krylov_dispatched])
def test_reference_solvers_solve(unsymmetric, solver):
    A, b, x_true = unsymmetric
    x, _, _, _ = solver(A, b)
    assert metrics.score(A, x, b, x_true)["status"] == metrics.STATUS_SOLVED


def test_gauss_seidel_delta_form_matches_reference(unsymmetric):
    """The rewrite must not have changed the answer, only the cost.

    The previous implementation solved (D+L)x = b - Ux with spsolve_triangular
    inside the loop; this reproduces it and checks the delta form agrees.
    """
    A, b, _ = unsymmetric
    L, U = sp.tril(A, format="csr"), sp.triu(A, k=1, format="csr")
    bn = np.linalg.norm(b)
    x_ref = np.zeros(A.shape[0])
    for _ in range(5000):
        x_ref = spla.spsolve_triangular(L, b - U @ x_ref, lower=True)
        if np.linalg.norm(A @ x_ref - b) / bn < 1e-8:
            break
    x_new, _, converged, _ = it.gauss_seidel(A, b)
    assert converged
    assert np.allclose(x_ref, x_new, rtol=1e-6, atol=1e-9)


def test_cholesky_rejects_unsymmetric():
    A = np.array([[4.0, 1.0], [3.0, 5.0]])
    with pytest.raises(SolverNotApplicable):
        direct.cholesky_solve(A, np.array([5.0, 8.0]))


def test_zero_diagonal_is_not_applicable_not_a_crash():
    A = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    for solver in (it.jacobi, it.gauss_seidel, it.sor):
        with pytest.raises(SolverNotApplicable):
            solver(A, np.array([1.0, 1.0]))


# ---------------------------------------------------------------- scoring


def test_success_requires_a_correct_answer(tiny):
    """The bug that mattered most: 'ok' used to mean 'did not raise'."""
    A, b, x_true = tiny
    garbage = np.array([1e6, -1e6, 1e6])
    assert metrics.score(A, garbage, b, x_true)["status"] != metrics.STATUS_SOLVED


def test_solver_claiming_convergence_wrongly_is_flagged_inaccurate(tiny):
    A, b, x_true = tiny
    scored = metrics.score(A, np.zeros(3), b, x_true, reported_converged=True)
    assert scored["status"] == metrics.STATUS_INACCURATE


def test_honest_failure_is_distinguished_from_a_false_claim(tiny):
    A, b, x_true = tiny
    scored = metrics.score(A, np.zeros(3), b, x_true, reported_converged=False)
    assert scored["status"] == metrics.STATUS_NO_CONVERGE


def test_non_finite_is_diverged(tiny):
    A, b, x_true = tiny
    assert metrics.score(A, np.full(3, np.nan), b, x_true)["status"] == metrics.STATUS_DIVERGED


def test_rates_keep_applicability_and_success_separate():
    """A method defined on half the corpus and perfect there is 100% conditional,
    not 50%. Collapsing the two is what understated Cholesky as 10.1%."""
    import pandas as pd
    df = pd.DataFrame({
        "method": ["Cholesky"] * 4,
        "status": [metrics.STATUS_SOLVED, metrics.STATUS_SOLVED,
                   metrics.STATUS_NOT_APPLICABLE, metrics.STATUS_NOT_APPLICABLE],
    })
    r = metrics.rates(df, "Cholesky")
    assert r["applicability"] == 0.5
    assert r["conditional_success"] == 1.0


# ---------------------------------------------------------------- refinement


def test_refinement_improves_accuracy_and_costs_more(unsymmetric):
    A, b, x_true = unsymmetric
    x0, _, _, w0 = refine(it.gauss_seidel, A, b, passes=0)
    x1, _, _, w1 = refine(it.gauss_seidel, A, b, passes=1)
    e0 = metrics.score(A, x0, b, x_true)["residual_rel"]
    e1 = metrics.score(A, x1, b, x_true)["residual_rel"]
    assert e1 < e0
    assert w1.matvecs > w0.matvecs, "refinement must be charged for the work it does"


def test_refinement_is_available_to_every_method(tiny):
    A, b, _ = tiny
    for m in benchmark.METHODS:
        if m.dense or m.family == "preconditioned":
            continue
        x, _, _, _ = refine(m.fn, A, b, passes=1)
        assert np.all(np.isfinite(x)), m.name


# ---------------------------------------------------------------- spectral


def test_spectral_radius_predicts_convergence(tiny, unsymmetric):
    for A, b, x_true in (tiny, unsymmetric):
        rho_j, _ = spectral.spectral_radius(A, "jacobi")
        rho_g, _ = spectral.spectral_radius(A, "gauss_seidel")
        for rho, solver in ((rho_j, it.jacobi), (rho_g, it.gauss_seidel)):
            x, _, _, _ = solver(A, b, max_iter=20000)
            solved = metrics.score(A, x, b, x_true)["status"] == metrics.STATUS_SOLVED
            verdict = spectral.convergence_verdict(rho, budget=20000)
            assert solved == (verdict == "converges"), f"rho={rho:.6f}, verdict={verdict}"


def test_convergence_verdict_separates_slow_from_convergent():
    """rho < 1 is not the same as usable. nos7 in this corpus has
    rho(T_J) = 0.999999984536822 -- provably convergent, and about 1.19e9
    iterations away from a 1e-8 residual."""
    assert spectral.convergence_verdict(1.5) == "diverges"
    assert spectral.convergence_verdict(0.5) == "converges"
    assert spectral.convergence_verdict(0.999999984536822) == "too_slow"
    assert spectral.iterations_to_tolerance(0.999999984536822) > 1e9
    assert spectral.iterations_to_tolerance(1.5) == np.inf


def test_batched_iteration_matrix_matches_operator(unsymmetric):
    A, _, _ = unsymmetric
    for kind in ("jacobi", "gauss_seidel"):
        op = (spectral.jacobi_operator(A) if kind == "jacobi"
              else spectral.gauss_seidel_operator(A))
        loop = np.column_stack([op @ e for e in np.eye(A.shape[0])])
        assert np.allclose(loop, spectral.dense_iteration_matrix(A, kind))


def test_diagonally_dominant_matrix_is_classified(tiny):
    A, _, _ = tiny
    c = spectral.classify(A)
    assert c["strict_diag_dominant"] and c["symmetric"] and c["spd"]
    assert c["rho_jacobi"] < 1.0 and c["hypothesis_class"] != "none"


# ---------------------------------------------------------------- harness


def test_every_matrix_produces_a_full_row_set(tmp_path, tiny):
    """Denominators must reconcile by construction, including for skips."""
    A, _, _ = tiny
    d = tmp_path / "toy_domain"
    d.mkdir()
    import scipy.io
    scipy.io.mmwrite(str(d / "tiny.mtx"), A)

    rows, spec = benchmark.run_one_matrix("toy_domain", "tiny", d / "tiny.mtx",
                                          refinement_passes=(0, 1), verbose=False)
    assert len(rows) == len(benchmark.METHODS) * 2
    assert {r["method"] for r in rows} == set(benchmark.METHOD_NAMES)
    assert all("status" in r for r in rows)
    assert spec["rho_jacobi"] < 1.0


def test_structurally_singular_matrix_is_screened_not_solved(tmp_path):
    """19 of 930 matrices have a zero row or column, and spilu on one of them
    got two Kaggle kernels OS-killed."""
    A = sp.csr_matrix(np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]))
    assert io_utils.structural_singularity(A) == (1, 1)

    d = tmp_path / "sing"
    d.mkdir()
    import scipy.io
    scipy.io.mmwrite(str(d / "s.mtx"), A)
    rows, _ = benchmark.run_one_matrix("sing", "s", d / "s.mtx",
                                       refinement_passes=(0,), verbose=False)
    assert {r["status"] for r in rows} == {metrics.STATUS_SINGULAR}


def test_cancellation_ratio_detects_a_noise_rhs():
    """b = A @ ones is nearly pure cancellation when rows sum to ~0. Six real
    matrices hit this, the worst at 3.5e-17."""
    A = sp.csr_matrix(np.array([[1.0, -1.0], [1.0, -1.0]]))
    assert io_utils.cancellation_ratio(A, np.ones(2)) < 1e-15


# ---------------------------------------------------------------- reordering


def test_permutation_restores_a_shuffled_dominant_matrix():
    """The worked example: a diagonally dominant matrix whose rows arrived in the
    wrong order. rho(T_GS) is 99.19 as given and 0.0316 once the diagonal is chosen."""
    from solvebench import reordering as ro
    A = sp.csr_matrix(np.array([[1.0, 10.0, 1.0],
                                [10.0, 1.0, 1.0],
                                [1.0, 1.0, 10.0]]))
    b = A @ np.ones(3)
    assert spectral.spectral_radius(A, "gauss_seidel")[0] > 1.0

    for objective in ("bottleneck", "minsum", "mc64", "best"):
        A2, b2, _ = ro.select_diagonal(A, b, objective=objective)
        assert spectral.spectral_radius(A2, "gauss_seidel")[0] < 0.1, objective
        x, _, conv, _ = it.gauss_seidel(A2, b2)
        assert metrics.score(A2, x, b2, np.ones(3), reported_converged=conv)["status"] \
            == metrics.STATUS_SOLVED, objective


def test_row_permutation_preserves_the_solution():
    """A'x = PAx = Pb = b', so nothing needs un-permuting afterwards."""
    from solvebench import reordering as ro
    rng = np.random.default_rng(11)
    A = sp.csr_matrix(rng.standard_normal((6, 6)))
    x_true = rng.standard_normal(6)
    b = A @ x_true
    A2, b2, _ = ro.select_diagonal(A, b, objective="bottleneck")
    assert np.allclose(A2 @ x_true, b2)


def test_scaling_cannot_change_the_spectral_radius():
    """Row and column scaling leave rho(T_J) invariant -- the fact that rules most
    preprocessing out and leaves row permutation as the only lever."""
    rng = np.random.default_rng(5)
    A = sp.csr_matrix(rng.standard_normal((30, 30)) + 6 * sp.eye(30))
    base = spectral.spectral_radius(A, "jacobi")[0]
    s = np.abs(rng.standard_normal(30)) + 0.5
    assert np.isclose(spectral.spectral_radius(sp.diags(s) @ A, "jacobi")[0], base)
    assert np.isclose(spectral.spectral_radius(A @ sp.diags(s), "jacobi")[0], base)


def test_best_is_never_worse_than_doing_nothing():
    """"no permutation" is one of the candidates, so the portfolio cannot lose."""
    from solvebench import reordering as ro
    rng = np.random.default_rng(3)
    A = sp.csr_matrix(rng.standard_normal((25, 25)) + 8 * sp.eye(25))
    before = spectral.spectral_radius(A, "gauss_seidel")[0]
    A2, _, info = ro.select_diagonal(A, None, objective="best")
    assert spectral.spectral_radius(A2, "gauss_seidel")[0] <= before + 1e-9
    assert info["chosen"] in ro.PORTFOLIO


def test_wide_ratio_range_does_not_hang():
    """Raw ratios spanning ten orders of magnitude made the min-sum matching run
    past 240 s on nnc261; log1p compresses them and it returns immediately."""
    from solvebench import reordering as ro
    A = sp.csr_matrix(np.array([[1e-10, 1.0, 0.0],
                                [1.0, 1e10, 1.0],
                                [0.0, 1.0, 1.0]]))
    p, cost = ro.minsum_permutation(A)
    assert p is not None and np.isfinite(cost)


# ---------------------------------------------------------------- corpus


def test_duplicate_matrices_are_removed_once():
    """bcsstk07, bcsstk12 and t2dal_a are byte-identical copies of other matrices in
    the corpus, so every result for them was counted twice."""
    import pandas as pd
    from solvebench import corpus
    df = pd.DataFrame({"matrix": ["bcsstk06", "bcsstk07", "t2dal", "t2dal_a", "nos7"],
                       "domain": ["structural_problem"] * 5})
    out = corpus.apply(df)
    assert set(out.matrix) == {"bcsstk06", "t2dal", "nos7"}
    assert "t2dal_bci" not in corpus.DUPLICATES, "same shape, different matrix -- keep it"
    for kept in corpus.KEPT_DESPITE_DUPLICATE_KIND:
        assert kept not in corpus.DUPLICATES


def test_split_cfd_domain_is_merged():
    """SuiteSparse spells the Goodwin group's kind without a 'problem' suffix, so its
    four matrices landed in a folder of their own beside the other 122."""
    import pandas as pd
    from solvebench import corpus
    df = pd.DataFrame({"matrix": ["Goodwin_010", "ex11"],
                       "domain": ["computational_fluid_dynamics",
                                  "computational_fluid_dynamics_problem"]})
    assert corpus.apply(df).domain.nunique() == 1


def test_correction_leaves_other_domains_alone():
    import pandas as pd
    from solvebench import corpus
    df = pd.DataFrame({"matrix": ["a", "b"],
                       "domain": ["structural_problem", "circuit_simulation_problem"]})
    assert list(corpus.apply(df).domain) == ["structural_problem", "circuit_simulation_problem"]


def test_divergence_guard_survives_a_transient_hump():
    """rho(T) is the asymptotic rate. With a non-normal iteration matrix ||T^k|| can
    climb a long way before it decays, and an over-eager guard kills the run on the way
    up. cdde6 has rho(T_J) = 0.717, peaks at 44,764x its starting residual by iteration
    73, and converges at 178; the old 1e4 threshold aborted it at 61."""
    from solvebench import config
    assert config.DIVERGENCE_GROWTH > 4.5e4, "must clear the measured transient peak"

    # A small non-normal system whose Jacobi iteration grows before it settles.
    n = 40
    A = sp.diags([np.full(n, 4.0), np.full(n - 1, -3.9), np.full(n - 1, 0.5)],
                 [0, 1, -1], format="csr")
    b = A @ np.ones(n)
    rho, _ = spectral.spectral_radius(A, "jacobi")
    assert rho < 1.0
    x, _, converged, _ = it.jacobi(A, b, max_iter=20000)
    assert converged, f"rho={rho:.4f} < 1 but the guard stopped the run"


def test_worst_row_ratio_reports_infinity_for_a_missing_diagonal():
    """A structurally absent a_ii must read as infinite, not as zero.

    The tempting implementation, ``row_ratios(A).diagonal()``, has no entry where the
    diagonal is not stored and so returns 0.0 -- a perfect ratio -- for precisely the
    matrices on which the stationary methods are undefined. That would have reported the
    491 zero-diagonal systems as ideally conditioned.
    """
    from solvebench import reordering as ro
    A = sp.csr_matrix(np.array([[0.0, 2.0], [1.0, 4.0]]))
    assert np.isinf(ro.worst_row_ratio(A))

    B = sp.csr_matrix(np.array([[4.0, 1.0], [1.0, 4.0]]))
    assert ro.worst_row_ratio(B) == pytest.approx(0.25)


def test_bottleneck_attains_the_minimum_worst_ratio():
    """The claim the method rests on: bottleneck assignment does not merely improve the
    worst row ratio, it *minimises* it. So a permutation making the matrix strictly
    diagonally dominant -- ratio < 1, which guarantees both Jacobi and Gauss-Seidel
    converge -- is found whenever one exists at all.

    Checked against brute force over every permutation on matrices small enough to
    enumerate, half of them built by shuffling the rows of a dominant matrix so that a
    dominant order is known to exist.
    """
    from solvebench import reordering as ro
    import itertools

    rng = np.random.default_rng(7)
    dominant_cases = 0
    for trial in range(120):
        n = int(rng.integers(3, 6))
        if trial % 2 == 0:
            M = rng.normal(size=(n, n)) * 0.1
            M[np.arange(n), np.arange(n)] = 1.0 + rng.random(n)
            M = M[rng.permutation(n), :]
        else:
            M = rng.normal(size=(n, n)) * (rng.random((n, n)) < 0.7)
        A = sp.csr_matrix(M)
        if A.nnz == 0:
            continue

        brute = min(ro.worst_row_ratio(A[list(p), :].tocsr())
                    for p in itertools.permutations(range(n)))
        p, _ = ro.bottleneck_permutation(A)
        got = np.inf if p is None else ro.worst_row_ratio(A[p, :].tocsr())

        assert np.isclose(got, brute, rtol=1e-9) or (np.isinf(got) and np.isinf(brute)), \
            f"bottleneck gave {got}, brute force reaches {brute}"
        if brute < 1:
            dominant_cases += 1
            assert got < 1, "a dominant permutation exists but was not found"

    assert dominant_cases > 20, "test lost its coverage of the dominant case"
