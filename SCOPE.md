# SolveBench — Project Scope

CSE 402 (Numerical Analysis Lab), BUET, Section A2.
Last updated: **2026-09-10**, after the reordering study completed.

Base paper: P. Khrapov and N. Volkov, *Comparative Analysis of Jacobi and Gauss-Seidel
Iterative Methods*, International Journal of Open Information Technologies, vol. 12,
no. 2, pp. 23–34, 2024.

> This file states what the project covers and what it does not. Measured numbers live
> in [`RESULTS.md`](RESULTS.md); where the work stands and what is still open lives in
> [`PROJECT_STATE.md`](PROJECT_STATE.md).

---

## 1. The question

The base paper derives exact convergence regions for Jacobi and Gauss-Seidel at **2 and
3 unknowns** algebraically, then validates them statistically on **random matrices** up
to 5 unknowns. It never touches real data.

**This project asks whether those conclusions survive on matrices practitioners actually
solve**, at sizes up to n = 10,000, and what happens to the comparison once the solvers
people really use are put beside them.

Three questions follow from that:

1. On real matrices, do Jacobi and Gauss-Seidel behave as the theory says?
2. Where do they stand against direct, Krylov and preconditioned solvers?
3. Can the systems they fail on be made solvable by changing how the matrix is presented?

---

## 2. Data

```
SuiteSparse Matrix Collection
927 unique square matrices · 26 domains · n = 5 to 10,000 · 53,167,175 nonzeros
```

* 930 files downloaded; 3 are exact duplicates of others and are dropped
  (`bcsstk07`≡`bcsstk06`, `bcsstk12`≡`bcsstk11`, `t2dal_a`≡`t2dal`).
* 6 files are truncated downloads and never load: `cavity20`, `nemeth17`, `nemeth22`,
  `nemeth23`, `nemeth25`, `psmigr_2`. Usable corpus is therefore **924 files**.
* One SuiteSparse spelling split a single domain in two
  (`computational_fluid_dynamics` / `..._problem`); they are merged.

**Selection rule:** every square real matrix in the collection with n ≤ 10,000. The cap
is the memory limit for the dense spectral analysis, not a judgement about difficulty.
The corpus is a sample of that population, not yet a complete census — 99 qualifying
matrices remain undownloaded.

---

## 3. Methods — 16, in five families

| Family | Methods |
|---|---|
| direct, hand-written | Gauss elimination, Gauss-Jordan, LU, Cholesky |
| direct, library | `spsolve` (SuperLU), `splu` (SuperLU) |
| **stationary** | **Jacobi, Gauss-Seidel, SOR (ω = 1.25)** |
| Krylov | Conjugate Gradient, BiCGSTAB, GMRES(30) |
| preconditioned | ILU only, ILU-BiCGSTAB, ILU-GMRES(30), ILU-Krylov (dispatched) |

The three stationary methods are the base paper's subject. The other thirteen exist so
that a statement like "Jacobi is slow" has something to be slow *against* — without a
sparse direct baseline the comparison has no floor and no ceiling.

`ILU-Krylov (dispatched)` is the rule the group originally proposed as its contribution:
choose BiCGSTAB or GMRES per matrix. It is kept as a baseline and reported honestly — it
is worth **one matrix in 736** against always choosing BiCGSTAB.

---

## 4. What is measured — four metrics, kept separate

| Metric | Definition |
|---|---|
| **applicability** | fraction of the corpus on which the method is *defined at all* |
| **conditional success** | of those, the fraction actually solved |
| **accuracy** | relative error against the known solution |
| **cost** | matrix-vector products, triangular solves, preconditioner applications |

**Applicability and conditional success are never multiplied together.** Conjugate
Gradient is defined on 11.7% of the corpus (SPD only) and solves 84.3% of those.
Collapsing that into one number reports CG at 10.4% and is how the first analysis
concluded CG was useless.

**Cost is counted in matrix-vector products, not seconds.** Wall-clock time on shared
Kaggle hardware is not reproducible, and iteration counts are not comparable across
methods — one ILU-preconditioned step does the work of many Jacobi steps.

No single method wins every metric, which is the point of keeping them apart:

```
applicability        BiCGSTAB / GMRES     97.3%
systems solved       spsolve              874
conditional success  splu                 98.8%
accuracy             ILU only             9.4e-15   (but solves only 79)
cost                 direct family        0 matvecs
```

---

## 5. Studies — four runs

| Study | Shape | Runtime | Question |
|---|---|---|---|
| Main sweep | 927 × 16 | 5.19 h | what each method can do |
| Spectral analysis | ρ(T_J), ρ(T_GS), matrix classes | 0.95 h | does theory predict the outcome? |
| Refinement study | 0 / 1 / 2 passes × every method | 3.30 h | who gains when all are given the same help? |
| **Reordering study** | 927 × 3 conditions × 3 methods | 1.91 h | **the group's own method** |

All four run on Kaggle from a notebook generated out of `src/solvebench/`, so the
library and the run cannot drift apart.

---

## 6. Methodological ground rules

These are in scope because the first version of this project got them wrong, and
correcting them changed the headline results.

1. **Success is decided from the residual by the harness**, never from a solver's own
   return flag. 57 solves had been recorded as successful with relative residuals up to
   9.71e+12.
2. **Every rate divides by what the method is applicable to**, never by the corpus.
3. **Iterative refinement is an orthogonal factor available to all**, not one method's
   private advantage.
4. **One configuration file** holds every tolerance and cap, so no two runs can disagree
   about what they measured.
5. **The divergence guard must not abort a genuine convergence.** ρ(T) is asymptotic; a
   non-normal iteration matrix can grow enormously first (`cdde6` peaks at 44,764× and
   converges at iteration 178).

---

## 7. The contribution — a convergence-oriented preprocessing pipeline

**We built a preprocessing pipeline that raises the success rate of the stationary
solvers by 55-72%, measured across 927 real sparse matrices.**

| method | as given | with the pipeline | |
|---|---|---|---|
| Jacobi | 75 | **124** | **+65%** |
| Gauss-Seidel | 121 | **208** | **+72%** |
| SOR | 119 | **184** | **+55%** |

199 systems recovered. **None lost** — the pipeline cannot score below the control,
because "change nothing" is one of the candidates it chooses among.

### What the pipeline does

A stationary method divides by `a_ii`, so its behaviour is decided by which entries sit
on the diagonal — and on a real matrix that was decided by the order the rows happened
to arrive in, not by anything to do with convergence. Two steps run before the solver,
which is left untouched:

```
step 1   choose the diagonal      a row permutation, solved as an assignment problem
step 2   choose omega             from a power-iteration estimate of rho(T_J)
step 3   run the solver           Jacobi / Gauss-Seidel / SOR / ILU-Krylov, unmodified
```

Step 1 runs four objectives in competition and keeps whichever gives the smallest
estimated `rho(T_GS)`:

```
none        change nothing            the control, and the guarantee we cannot lose
mc64        max prod |a_ii|           the established tool (Duff & Koster)
minsum      min sum log(1 + ratio)    proved here to be MC64 under another name
bottleneck  min max row ratio         ours
```

Row permutation is the only lever available, and that is proved rather than assumed: row
scaling leaves `T_J` algebraically identical, column scaling is a similarity transform,
and symmetric permutation moves `rho(T_GS)` only in the fifth decimal on real matrices.

### What is ours

**1. The pipeline** — the two preprocessing steps, the portfolio of competing objectives,
the selector, and the evaluation of all of it on a full corpus. MC64 is a component it
uses, the way any numerical code uses BLAS.

**2. The diagnosis of the ILU hole** — 166 matrices on which no incomplete factorization
can be built, which takes the entire preconditioned family down at once: BiCGSTAB solves
12 of them, GMRES 11, Gauss-Seidel 0, against sparse direct's 148. The cause is that
`build_ilu` falls back to diagonal scaling, which a zero on the diagonal closes off.
Step 1 makes **150 of the 166 constructible** and 17 solve outright.

**3. The bottleneck objective** — minimising the worst row ratio rather than maximising
the product of the diagonal. It beats MC64 on the quantity it targets **367-0** with 506
ties, and it is solved exactly, verified against brute force over every permutation of
matrices small enough to enumerate, 400/400.

Row permutation is the only lever available, and that is proved rather than assumed: row
scaling leaves `T_J` algebraically identical, column scaling is a similarity transform,
and symmetric permutation moves `rho(T_GS)` only in the fifth decimal on real matrices.
That is what keeps the method principled rather than an arbitrary heuristic.

### Where the detail lives

Every number behind the claim above, with its denominator — the per-objective
comparison, what each step contributes on its own, and the cases where the pipeline
gains nothing — is in [`RESULTS.md`](RESULTS.md) section 8. This file states what the
project claims; that one states what was measured.

## 8. Out of scope

* Non-square, complex, or n > 10,000 matrices
* Multigrid, domain decomposition, or any hierarchical solver
* GPU, parallel, or distributed implementation
* Wall-clock performance engineering
* Inventing a new *iterative method* — considered, and deliberately not attempted; the
  contribution is a preprocessing objective, not a new iteration
* Optimising ω for SOR per matrix (fixed at 1.25 throughout)
* Any claim about which method is best *for a domain* where the corpus holds fewer than
  five matrices — eight domains are in that position and every table prints n

---

## 9. Deliverables

| | State |
|---|---|
| Library, `src/solvebench/` | done — 43 tests |
| Kaggle notebooks, generated from the library | done — 5 |
| `RESULTS.md`, every number with its denominator | done |
| Figures | done — 16 |
| Repository, <https://github.com/NOOBIJS/SolveBench> | public |
| Slides, `linear.tex` | Update pages appended |
| **Report** | **not started** — contents not yet specified by the course |

---

## 10. What this project does not claim

* That the convergence-oriented objective beats MC64 on systems solved. It does not.
* That any iterative method beats a sparse direct solver here. `spsolve` solves 874 of
  927; the best iterative configuration solves 617.
* That the corpus is a complete census of SuiteSparse at n ≤ 10,000. 99 qualifying
  matrices are still undownloaded.
* That domain-level rankings are reliable where n is small.
