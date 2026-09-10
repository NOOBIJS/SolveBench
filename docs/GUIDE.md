# SolveBench — a guide for the group

Read this before the code. It explains what the project is, what every solver family
does, what each file produces, and how to read every figure. By the end you should be
able to open any chart in `results/figures/` and say what it means without asking.

CSE 402 (Numerical Analysis Lab), BUET, Section A2.

---

## Contents

1. [The problem, and the paper we are extending](#1-the-problem)
2. [The corpus](#2-the-corpus)
3. [The four solver families, explained](#3-the-four-solver-families)
4. [How we decide a solve succeeded](#4-how-we-decide-a-solve-succeeded)
5. [The four metrics, and why they are never mixed](#5-the-four-metrics)
6. [Our contribution: the pipeline](#6-our-contribution-the-pipeline)
7. [Every file, and what it does](#7-every-file)
8. [Every table, and what is in it](#8-every-table)
9. [Every figure, and how to read it](#9-every-figure)
10. [The results, in one place](#10-the-results)
11. [How to run it yourself](#11-how-to-run-it-yourself)
12. [Mistakes we made, so nobody repeats them](#12-mistakes-we-made)

---

## 1. The problem

We want to solve `Ax = b`, where `A` is a large sparse matrix — mostly zeros. This is the
single most common computation in engineering: circuit simulation, structural analysis,
fluid dynamics and power networks all reduce to it, often thousands of times per run.

There are two ways to do it.

**Direct methods** factor the matrix — `A = LU` — and then solve two easy triangular
systems. They give the answer in a fixed number of steps, and the answer is as accurate
as the arithmetic allows. The cost is that factoring can need far more memory than the
matrix itself, because `L` and `U` fill in where `A` had zeros.

**Iterative methods** start from a guess and improve it: `x₀ → x₁ → x₂ → …`, stopping
when the residual `‖Ax − b‖` is small enough. Each step is cheap. But they only work if
the iteration *converges*, and whether it does depends on the matrix.

### The base paper

> P. Khrapov and N. Volkov, *Comparative Analysis of Jacobi and Gauss-Seidel Iterative
> Methods*, International Journal of Open Information Technologies, 12(2), 2024.

It works out exactly which 2×2 and 3×3 matrices make Jacobi and Gauss-Seidel converge,
then checks the result statistically on **random** matrices up to 5 unknowns.

It never touches real data. That is the gap.

### Our question

**Do those conclusions survive on matrices people actually solve?** And once you put the
solvers an engineer would really reach for beside them, where do Jacobi and Gauss-Seidel
actually stand?

---

## 2. The corpus

```
SuiteSparse Matrix Collection
927 unique matrices · 26 domains · n = 5 to 10,000 · 53,167,175 nonzeros
```

SuiteSparse is the standard public collection of real sparse matrices, contributed by
engineers from actual simulations. Every matrix carries the domain it came from, which
is how we can ask "does Gauss-Seidel work in circuit simulation?" rather than only "does
it work on average".

**Selection rule:** every square real matrix at n ≤ 10,000. The cap is a memory limit —
computing an exact spectral radius needs a dense `n × n` array, and 10,000² doubles is
already 800 MB.

Some bookkeeping that matters when you read the numbers:

* 930 files, but **3 are exact duplicates** of other matrices under a second name, so
  927 unique.
* **6 files were truncated downloads** — their headers declared more nonzeros than the
  files contained — so they never loaded. The usable corpus is **924**.
* 100 of the 930 are **random matrices we generated ourselves**, to compare directly
  against the base paper's own validation method.
* SuiteSparse spells one domain two ways, splitting it in two; we merge them.

`dataset_full/` holds a larger corpus (1,278 matrices) downloaded later. **It is not used
for any result in this project** — every number here was measured on the 924, and mixing
them would silently invalidate all of it.

---

## 3. The four solver families

Sixteen methods, in five families. Here is what each family actually does.

### Direct, hand-written (4 methods)

Written from scratch on dense arrays, because this is a numerical analysis course and
implementing them is part of the point.

| method | what it does |
|---|---|
| **Gauss elimination** | eliminate below the diagonal, then back-substitute |
| **Gauss-Jordan** | eliminate above *and* below, so no back-substitution is needed |
| **LU** | the same elimination, but stored as `A = LU` so many right-hand sides reuse it |
| **Cholesky** | `A = LLᵀ`, half the work — but only valid for symmetric positive definite matrices |

All four need a dense `n × n` array, so they are capped at **n ≤ 2,000**. Above that they
are recorded as `skipped_too_large`, not as failures.

### Direct, library (2 methods)

| method | what it does |
|---|---|
| **spsolve (SuperLU)** | sparse LU with fill-reducing ordering and partial pivoting |
| **splu (SuperLU)** | the same factorization, kept for reuse |

These are what an engineer would actually run. They are the ceiling everything else is
measured against.

### Stationary (3 methods) — the base paper's subject

Split the matrix into `A = D + L + U`: the diagonal, everything below it, everything
above it. Then iterate.

| method | the update |
|---|---|
| **Jacobi** | `x_new = D⁻¹(b − (L+U)x_old)` — every component uses only old values |
| **Gauss-Seidel** | `x_new = (D+L)⁻¹(b − U x_old)` — uses new values as soon as they exist |
| **SOR** | Gauss-Seidel, then overshoot by a factor `ω` (we use 1.25) |

**All three divide by `D`.** If any `a_ii = 0` then `D⁻¹` does not exist and the method is
not *wrong* — it is **undefined**. On this corpus that happens on **491 of 927 matrices**,
which is the single most important fact in the whole project.

Convergence is governed by the **spectral radius** `ρ(T)` of the iteration matrix:

```
ρ(T) < 1   the error shrinks every step, eventually     → converges
ρ(T) ≥ 1   the error grows                              → diverges
```

`ρ` also sets the *speed*: reaching tolerance `ε` takes roughly `log(ε)/log(ρ)` steps.
`ρ = 0.5` needs about 27 steps; `ρ = 0.999` needs about 18,000.

### Krylov (3 methods)

Instead of a fixed update rule, build a subspace from `b, Ab, A²b, …` and pick the best
answer inside it at each step. Much stronger than stationary methods.

| method | what it does |
|---|---|
| **Conjugate Gradient (CG)** | optimal for symmetric positive definite matrices; **only** valid there |
| **BiCGSTAB** | works on nonsymmetric matrices, no symmetry requirement |
| **GMRES(30)** | minimises the residual over the subspace; restarted every 30 steps to bound memory |

### Preconditioned (4 methods)

A **preconditioner** `M ≈ A` that is cheap to invert. Solving `M⁻¹Ax = M⁻¹b` instead
converges far faster when `M⁻¹A` is closer to the identity than `A` is.

**ILU** (Incomplete LU) is the standard choice: a factorization that drops the small
fill-in entries, keeping it sparse.

| method | what it does |
|---|---|
| **ILU only** | apply the preconditioner once, no iteration — a control |
| **ILU-BiCGSTAB** | ILU + BiCGSTAB |
| **ILU-GMRES(30)** | ILU + GMRES |
| **ILU-Krylov (dispatched)** | pick BiCGSTAB or GMRES per matrix |

That last one was the group's **original** proposed contribution. We measured it
honestly: against always choosing BiCGSTAB it is worth **one matrix in 736**. It is kept
as a baseline. This is why the project needed a real contribution instead.

---

## 4. How we decide a solve succeeded

**This is the most important rule in the codebase.**

Every solver returns `(x, iterations, reported_converged, work)`. The third value is the
solver's *own opinion*, and we record it and then **ignore it**. Success is decided
afterwards, by us, from the residual we compute ourselves:

```
relative residual  =  ‖Ax − b‖ / ‖b‖        solved if ≤ 1e-8
```

Why this matters: in the first version of this project, direct methods were marked `ok`
whenever the call did not raise an exception. That passed **57 solves whose relative
residual reached 9.71e+12** — answers wrong by twelve orders of magnitude, recorded as
successes.

Eight possible outcomes, and the distinctions are deliberate:

| status | meaning |
|---|---|
| `solved` | residual verified below tolerance |
| `inaccurate` | **the solver claimed success and the residual says otherwise** |
| `did_not_converge` | the solver admitted failure, and the residual agrees |
| `diverged` | the iterate blew up or went non-finite |
| `not_applicable` | the method is undefined here (zero diagonal, or CG on a non-SPD matrix) |
| `structurally_singular` | the matrix has no unique solution — nobody's fault |
| `skipped_too_large` | deliberately not attempted (dense methods above n = 2,000) |
| `error` | an unexpected failure, with the exception recorded |

`not_applicable` and `did_not_converge` are **completely different** and merging them is
how a method gets slandered. Jacobi is not "failing" on 491 matrices — it does not exist
on them.

---

## 5. The four metrics

| metric | definition |
|---|---|
| **applicability** | of the whole corpus, how much is this method even *defined* on |
| **conditional success** | of that, how much does it actually solve |
| **accuracy** | relative error against the known solution |
| **cost** | matrix-vector products, triangular solves, preconditioner applications |

### The rule: applicability and conditional success are never multiplied

Conjugate Gradient is defined on **11.7%** of the corpus (SPD only) and solves **84.3%**
of those. Multiply them and you report "CG solves 10.4%", which sounds useless. It is
not useless — it is *specialised*. Where it applies it is excellent.

Every rate in every figure divides by what the method is applicable to, never by the
corpus.

### Cost is counted, not timed

We count **matrix-vector products**, not seconds. Two reasons:

* Kaggle runs on shared hardware; wall-clock time is not reproducible.
* Iteration counts are not comparable across methods — one ILU-preconditioned step does
  the work of many Jacobi steps.

Runtime is still recorded, and used only where a common unit is unavoidable (the
performance profile, figure 05).

### Ground truth

We do not use a right-hand side that came with the matrix. We pick `x_true` ourselves and
compute `b = A x_true`. That way the exact answer is known, and `error_rel` is a real
error rather than an estimate.

---

## 6. Our contribution: the pipeline

### The observation

A stationary method divides by `a_ii`. So its entire behaviour is decided by **which
entries sit on the diagonal** — and in a real matrix, that was decided by the order the
rows happened to be written in. Nobody chose it with convergence in mind.

Swapping rows does not change the solution:

```
A' = PA,  b' = Pb        A'x = PAx = Pb = b'
```

Same `x`. Nothing has to be undone afterwards. But it completely changes `D`.

### A worked example

```
A = [ 0  5 ]     a₁₁ = 0  →  D⁻¹ does not exist  →  Jacobi, GS, SOR and ILU all fail
    [ 3  1 ]

swap the rows:

A'= [ 3  1 ]     a₁₁ = 3  →  everything works
    [ 0  5 ]
```

### What can and cannot be changed — proved, not assumed

Before searching, we established that row permutation is the *only* lever:

| transformation | effect on `ρ(T_J)` |
|---|---|
| row scaling | `T_J' = I − (RD)⁻¹RA = T_J` — **identical**, exactly |
| column scaling | a similarity transform — spectrum unchanged |
| symmetric permutation | measured on real matrices: moves `ρ(T_GS)` in the 5th decimal |
| **row permutation** | **changes which entries form `D`** |

### The pipeline

```
A, b
  │
  ▼  STEP 1 — choose the diagonal
  │   an assignment problem: pick one entry per row, one per column,
  │   optimising some objective. Four compete:
  │
  │     none        change nothing            the control
  │     mc64        max Π|a_ii|               the established tool (Duff & Koster)
  │     minsum      min Σ log(1 + ratio)      proved here to BE mc64
  │     bottleneck  min max row-ratio         ours
  │
  │   the selector runs all four and keeps whichever gives the
  │   smallest estimated ρ(T_GS)
  │
  ▼  STEP 2 — choose ω   (SOR only)
  │   estimate ρ(T_J) by power iteration, then Young's 1950 formula
  │     ω* = 2 / (1 + √(1 − ρ²))
  │   if ρ ≥ 1, fall back to ω = 1 — over-relaxing a divergent
  │   iteration only diverges it faster
  │
  ▼  STEP 3 — run the solver, completely unmodified
```

### The row ratio, which is what "bottleneck" minimises

For row `i` with `a_ii` on the diagonal:

```
ratio_i  =  ( Σ_{j≠i} |a_ij| )  /  |a_ii|
```

**`ratio < 1` for every row means strict diagonal dominance**, which *guarantees* both
Jacobi and Gauss-Seidel converge. So the ratio is not an arbitrary score — it is the
quantity the classical convergence theorem is stated in.

MC64 maximises `Π|a_ii|`, which looks only at how *big* the diagonal entry is. A value of
1000 looks excellent until you notice the rest of its row sums to 50,000. **That is the
difference between the two objectives.**

### Because "none" is a candidate, the pipeline cannot lose

The selector chooses among four options and one of them is "change nothing". So the
portfolio is **guaranteed** never to score below the control — that is a property of the
construction, not a lucky measurement.

### The result

| method | as given | with the pipeline | |
|---|---|---|---|
| Jacobi | 76 | **124** | **+63%** |
| Gauss-Seidel | 122 | **208** | **+70%** |
| SOR | 119 | **184** | **+55%** |

**199 systems recovered, none lost.**

---

## 7. Every file

### `src/solvebench/` — the library, and the only source of truth

| file | what it does |
|---|---|
| `config.py` | every tolerance, cap and constant, in one place. Nothing anywhere else may hard-code a limit |
| `metrics.py` | the single success predicate. Defines the 8 statuses and `score()` |
| `io_utils.py` | loads a `.mtx` file, builds the ground-truth `(x_true, b)` pair, detects structural singularity |
| `direct_solvers.py` | the four hand-written direct methods |
| `iterative_solvers.py` | Jacobi, Gauss-Seidel, SOR, CG, BiCGSTAB, GMRES, `build_ilu`, and the adaptive-ω pieces |
| `reference_solvers.py` | the SuperLU and ILU-Krylov baselines |
| `refinement.py` | iterative refinement, usable by *any* solver |
| `reordering.py` | **our method** — the four objectives, the selector, `make_solver` |
| `spectral.py` | `ρ(T_J)`, `ρ(T_GS)`, the convergence verdict, and the classical matrix classes |
| `benchmark.py` | the harness: the 16 baseline methods, the 7 pipeline arms, `run_one_matrix` |
| `corpus.py` | corrections applied at analysis time — merges the split domain, drops the 3 duplicates |

### `tools/` — scripts you run

| file | what it does | what it produces |
|---|---|---|
| `build_notebooks.py` | embeds the library into Kaggle notebooks | `notebooks/*/*.ipynb` |
| `test_notebooks.py` | executes every generated cell locally on 4 matrices, and parses each against Kaggle's Python version | pass/fail — **nothing goes to Kaggle without this** |
| `make_figures.py` | draws every figure from the CSVs | `results/figures/*.png` + `index.md` |
| `download_corpus.py` | fetches missing SuiteSparse matrices, verifying each against its own header | `dataset_full/` |
| `preflight_reordering.py` | runs every permutation objective over the whole corpus locally | catches hangs before they cost a Kaggle session |
| `probe_ilu_reorder.py` | tests whether reordering makes ILU buildable | `results/tables/ilu_reorder_probe.csv` |
| `probe_adaptive_sor.py` | tests whether a per-matrix ω beats the fixed 1.25 | `results/tables/adaptive_sor_probe.csv` |
| `rhs_cancellation.py` | measures how much of `b` survives floating-point cancellation | diagnostic |

### `notebooks/` — what runs on Kaggle

Five notebooks, each **generated** from the library. **Never edit a notebook by hand** —
that is exactly how the library and the notebook drifted apart once already.

| notebook | what it runs | how long |
|---|---|---|
| `solvebench-main-sweep` | 927 × 16 methods | 5.19 h |
| `solvebench-spectral` | `ρ` and matrix classes, no solving | 0.95 h |
| `solvebench-refinement-study` | 0/1/2 refinement passes for every method | 3.30 h |
| `solvebench-reordering-study` | 3 conditions × 3 stationary methods | 1.91 h |
| `solvebench-pipeline` | the 7 pipeline arms | ~2 h |

### Documents

| file | what it is |
|---|---|
| `README.md` | the front door: the result, then how to run it |
| `SCOPE.md` | what the project covers, what it deliberately does not, what it does not claim |
| `RESULTS.md` | **every measured number, with its denominator** |
| `docs/GUIDE.md` | this file |
| `docs/PROJECT_STATE.md` | where the work stands, what was decided, what is still open |
| `docs/legacy_first_run/` | the first sweep, superseded — kept for comparison |

---

## 8. Every table

All in `results/tables/`.

| table | rows | what is in it |
|---|---|---|
| `benchmark_results.csv` | 14,880 | one row per (matrix, method, refinement pass) — the main result |
| `method_summary.csv` | 16 | applicability and conditional success per method |
| `per_domain.csv` | — | the same, split by domain, **with denominators** |
| `spectral.csv` | 930 | `ρ(T_J)`, `ρ(T_GS)`, verdicts, and 7 matrix-class flags — one row per *file*, so the 3 duplicates are still in it; `corpus.apply` drops them at analysis time |
| `hypothesis_coverage.csv` | — | how many matrices fall in each classical class |
| `refinement_study.csv` | — | every method at 0, 1 and 2 refinement passes |
| `reordering_study.csv` | 8,170 | 3 conditions × 3 methods, plus the four objectives' ratios |
| `reordering_pivot.csv` | — | solved / not solved, pivoted for the ablation |
| `ilu_reorder_probe.csv` | 166 | whether reordering makes ILU buildable, and whether it then solves |
| `adaptive_sor_probe.csv` | 400 | fixed ω against adaptive ω |

### The columns you will actually use

```
domain, matrix, n, nnz, density        what the matrix is
condition_number, ill_conditioned      how hard it is
method, family                         what was run
status                                 the outcome — one of the 8
residual_rel, error_rel                accuracy
matvecs, tri_solves, precond_applies   cost
iterations, runtime_sec, setup_sec     effort
reported_converged                     the solver's own claim — recorded, not trusted
```

---

## 9. Every figure

Seventeen figures, numbered in the order the story is told. Contribution first, then the
benchmark, then the theory, then supporting studies.

### The contribution (01–03)

**`01_pipeline_scoreboard.png`** — Two bars per method: as given, and with the pipeline.
The percentage above each pair is the gain. This is the headline: 76→124, 122→208,
119→184. The solvers are unchanged; only the row order and, for SOR, ω.

**`02_ilu_hole.png`** — Two panels.
*Left:* on the 166 matrices where no ILU can be built, Gauss-Seidel solves 0, GMRES 11,
BiCGSTAB 12 — while sparse direct solves 148. **That 148 is the point**: these are not
hard systems, they are systems the iterative toolkit cannot reach.
*Right:* choosing the diagonal first makes 150 of the 166 constructible, and 17 solve.
The gap between 150 and 17 is deliberate — a preconditioner you can build is not a solve.

**`03_dominance.png`** — Two panels.
*Left:* a scatter, MC64's row ratio against ours, one point per matrix, log-log with the
diagonal drawn. **Every point is on or below the line** — 367 below, 0 above, 506 on it.
Our objective never loses on the quantity it optimises.
*Right:* the same as a distribution, with the `ratio = 1` line marked. That line is
strict diagonal dominance, where convergence is guaranteed. 43 matrices are already
below it and **no permutation moves a single further matrix under it** — so the win on
the left lands in a regime where it does not decide convergence.

### The benchmark (04–10)

**`04_method_scoreboard.png`** — Applicability against conditional success, all 16
methods. The two are separate bars, never multiplied.

**`05_performance_profile.png`** — The standard solver comparison (Dolan–Moré). For each
matrix, divide every solver's time by the fastest time any solver achieved on it. The
curve is the fraction handled within a factor τ of the best.
*Read it two ways:* height at τ = 1 is how often that method is **outright fastest**; the
right-hand plateau is how much of the corpus it **solves at all**. Sparse direct 99%,
ILU-Krylov 70%, dense direct 67%, Krylov 39%, stationary 14%.

**`06_outcomes.png`** — The 8 statuses per method. Watch for `inaccurate` — the category
that did not exist before: the solver returned happily and the residual says the answer
is wrong.

**`07_domain_heatmap.png`** — Method against domain. Grey cells mean the method is
defined on nothing in that domain. Rates are conditional; denominators are in
`per_domain.csv`.

**`08_cost_profile.png`** — Matrix-vector products. Direct methods are absent **by
construction** — they perform no matvecs at all; their cost is the factorization.

**`09_runtime_scaling.png`** — Median runtime against matrix size, restricted to the 113
matrices *all four families solve*, so the curves describe the same problems at every
size. Without that restriction the Krylov and stationary curves bend **downward**, which
would suggest they get faster on bigger matrices; they do not — at large sizes they only
succeed on the easy ones. Read the slope, not the height.

**`10_iteration_counts.png`** — How many iterations each method needs where it succeeds.
ILU-BiCGSTAB median 5, Gauss-Seidel 36, BiCGSTAB 568. Each curve is conditioned on that
method's own successes, so `n` is printed in the legend. Fewer steps ≠ less work — one
preconditioned step is expensive. Figure 08 counts the work.

### The theory (11–14)

**`11_spectral_radius.png`** — The distribution of `ρ(T_J)` and `ρ(T_GS)`. This is the
quantity the base paper exists to characterise, computed here for every matrix.

**`12_hypothesis_coverage.png`** — How many matrices fall in each classical class.
Stein–Rosenberg (1948) forbids "Jacobi converges but Gauss-Seidel does not" for
M-matrices; Householder–John (1958) guarantees Gauss-Seidel for SPD. How much of the
corpus those theorems actually cover is what explains the benchmark's own results.

**`13_prediction_vs_observation.png`** — Does theory predict the outcome? Includes
`too_slow`, the case the textbook criterion cannot express: `ρ < 1` so convergence is
guaranteed, but not within any usable number of iterations.

**`14_jacobi_vs_gauss_seidel.png`** — The base paper's own question, on real data. The
denominator is matrices where **both** are defined, not the corpus.

### Supporting studies (15–17)

**`15_refinement_effect.png`** — Iterative refinement given to *every* method. Jacobi
becomes the most accurate method in the benchmark — and pays about 20× the work for it,
which is why cost is plotted beside accuracy. In the first sweep only one method received
refinement, and that is exactly where its apparent accuracy advantage came from.

**`16_conditioning.png`** — Accuracy split by whether the matrix is ill-conditioned.
Pooling the two is what let *dataset* difficulty masquerade as *solver* inaccuracy.

**`17_dispatch_ablation.png`** — The group's original proposal, measured honestly. If the
dispatched bar matches always-BiCGSTAB, the dispatch rule adds nothing. It very nearly
does.

---

## 10. The results

### The corpus breaks the textbook methods before you start

```
Jacobi / Gauss-Seidel / SOR are UNDEFINED on 491 of 927 matrices  (53%)
```

Not slow. Not divergent. **Undefined** — a zero on the diagonal.

### A sparse direct solver wins outright

```
spsolve                874 / 927
best iterative         617        (ILU-Krylov)
best stationary        121        (Gauss-Seidel)
```

### On random matrices — the base paper's own test bed — stationary methods score zero

```
100 random matrices, n = 5 … 300

  Jacobi        0 / 100        spsolve       100 / 100
  Gauss-Seidel  0 / 100        ILU-BiCGSTAB  100 / 100
```

And `ρ(T_J)` grows roughly linearly with `n`: median 10 at n ≤ 10, 166 at n ≈ 300. The
base paper's validation works at 2–5 unknowns and stops working almost immediately after.

### Our pipeline

```
Jacobi         76 → 124   (+63%)
Gauss-Seidel  122 → 208   (+70%)
SOR           119 → 184   (+55%)
              199 recovered, 0 lost
```

### The ILU hole

```
166 matrices where no ILU can be built
  → step 1 makes 150 constructible, 17 solve
```

### Where our objective stands against MC64

On the row ratio it optimises: **367 wins, 0 losses, 506 ties.**
On systems solved: **+2**. The reordering *idea* carries the 199; our objective is a
small part of it. Both facts are in `RESULTS.md`.

---

## 11. How to run it yourself

```bash
pip install -r requirements.txt

python -m pytest tests/            # 47 tests — run this first
python tools/build_notebooks.py    # regenerate the notebooks from the library
python tools/test_notebooks.py 4   # execute every cell locally on 4 matrices
python tools/make_figures.py       # redraw every figure from results/tables/
```

The matrix corpus (~2.6 GB) is not in the repository. `tools/download_corpus.py` fetches
it from SuiteSparse.

**Order matters.** The library is the source of truth; notebooks are generated from it;
CSVs come from the notebooks; figures come from the CSVs. Editing anything downstream by
hand breaks the chain.

---

## 12. Mistakes we made

Every one of these was in the first version and changed a headline number when fixed.
They are here so nobody reintroduces them.

**Trusting the solver's own success flag.** 57 solves were recorded as successful with
relative residuals up to 9.71e+12. Success is now computed from the residual, by the
harness, for every method identically.

**Dividing by the corpus instead of by applicability.** Reported CG at 10.4% when its
conditional success is 84.3%.

**Giving refinement to one method only.** That method's accuracy advantage *was* the
refinement. It is now an orthogonal factor available to all.

**Re-analysing the triangular factor every iteration.** Gauss-Seidel and SOR were
factoring inside the loop, inflating their measured runtime 11–14×.

**A divergence guard that was too tight.** Aborting at 1e4 residual growth killed about
23 genuine convergences. `ρ(T)` is the *asymptotic* rate — a non-normal iteration matrix
can grow enormously before it decays. `cdde6` climbs to 44,764× its starting residual and
then converges at iteration 178.

**Two copies of the library.** The notebook carried its own copy, which drifted from the
real one — they disagreed about a size cap while producing results. Notebooks are now
generated, never edited.

**Keeping unverified downloads.** Six files were truncated and nobody noticed until they
broke a benchmark run months later. Downloads are now verified against their own headers
on arrival.

**Trusting library routines without measuring them.** Two `scipy` graph-matching routines
hang on real matrices — one ran 90+ seconds on a 817×817 matrix, another never returned
on a 67×67 one. They cost a full 12-hour Kaggle session before we replaced them with a
dense assignment. `tools/preflight_reordering.py` now tries every matrix locally first.
