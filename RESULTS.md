# SolveBench — Results

**Living document.** Every number here was measured, not estimated, and each is traceable
to a table under `results/tables/`. Updated as work progresses.

Last updated: **2026-09-10** · corpus **927 unique matrices, 26 domains**, n = 5 … 10,000
(930 files downloaded; see §0.1 for the three duplicates and the split domain)

---

## 0. Runs behind these numbers

| Notebook | Runtime | Output | Rows |
|---|---|---|---|
| `solvebench-main-sweep` | 5.19 h | `benchmark_results.csv` | 14,880 = 930 × 16 ✓ |
| `solvebench-refinement-study` | 3.30 h | `refinement_study.csv` | 9,600 = 200 × 16 × 3 ✓ |
| `solvebench-spectral` | 0.95 h | `spectral.csv` | 930 ✓ |

Kaggle, CPU only, 4 cores, scipy 1.16.3 / numpy 2.0.2, BLAS threads pinned to 1.
Row counts are exact, so nothing was silently dropped.

Previous run for comparison: 237.2 min, 10 methods, and its numbers are superseded —
see §7 for what changed and why.

### Corpus accounting (reconciles to 930)

```
930 files downloaded
  − 3   exact duplicates (see 0.1)               →  927 unique matrices
  − 6   load failures (truncated downloads)      →  921 loaded
  − 19  structurally singular (zero row/column)  →  902 scored
```

## 0.1 Two defects in the downloaded corpus

Both are corrected at analysis time by `src/solvebench/corpus.py`, so the raw files, the
Kaggle runs and every table stay consistent without re-uploading 985 MB.

**One domain arrived split in two.** Folders were named by slugifying SuiteSparse's
`kind` field, which spells the same physical domain several ways. The grouping folded
`X problem`, `subsequent X problem` and `X problem sequence` together correctly
everywhere — but the Goodwin group carries the kind `computational fluid dynamics` with
no "problem" suffix, so its 4 matrices landed in a folder of their own beside the 122 in
`computational_fluid_dynamics_problem`. Every other domain checks out. **27 → 26 domains.**

**Three matrices are exact duplicates of three others.** SuiteSparse marks them with a
`duplicate ...` kind; comparing CSR structure and values byte for byte confirms it:

```
bcsstk07  ≡  bcsstk06     420 x 420,   7,860 nnz
bcsstk12  ≡  bcsstk11    1473 x 1473, 34,241 nnz
t2dal_a   ≡  t2dal       4257 x 4257, 37,465 nnz
```

Every result for those three was counted twice. `t2dal_bci` has the same shape but is a
genuinely different matrix and is kept; `nasa1824` and `t2dal_e` carry a duplicate kind
but their originals are not in this corpus, so they are unique here and are kept too.

**The correction changes no conclusion.** spsolve 876 → 874 solved (96.9% → 97.0%); the
dispatch gap stays at exactly one matrix (617 vs 616); the base-paper test goes from
n = 414 to n = 411 with 75 / 46 / 0 unchanged.

---

## 1. The headline table

Applicability and conditional success are **separate measurements** and are never
multiplied. Applicability = on how much of the corpus the method is even defined.
Conditional success = of that, how much it actually solves, verified by residual.

| Method | Applicable | Solved | Applicability | Conditional success |
|---|---|---|---|---|
| Gauss elimination | 596 | 593 | 64.3% | **99.5%** |
| LU | 596 | 593 | 64.3% | **99.5%** |
| splu (SuperLU) | 885 | 874 | 95.5% | **98.8%** |
| spsolve (SuperLU) | 901 | 874 | 97.2% | **97.0%** |
| Cholesky | 60 | 58 | 6.5% | 96.7% |
| Gauss-Jordan | 594 | 537 | 64.1% | 90.4% |
| Conjugate Gradient | 108 | 91 | 11.7% | 84.3% |
| ILU-Krylov (dispatched) | 736 | 617 | 79.4% | 83.8% |
| ILU-BiCGSTAB | 736 | 616 | 79.4% | 83.7% |
| ILU-GMRES(30) | 736 | 602 | 79.4% | 81.8% |
| BiCGSTAB | 902 | 341 | 97.3% | 37.8% |
| Gauss-Seidel | 411 | 121 | 44.3% | 29.4% |
| SOR (ω=1.25) | 411 | 119 | 44.3% | 29.0% |
| GMRES(30) | 902 | 240 | 97.3% | 26.6% |
| Jacobi | 411 | 75 | 44.3% | 18.2% |
| ILU only | 736 | 79 | 79.4% | 10.7% |

**A sparse direct solver wins outright: `spsolve` solves 874, the best iterative
configuration 617.** This is the comparison the first sweep never ran.

### Outcomes across all 14,832 attempts

```
solved                 6,430      not_applicable         3,539
did_not_converge       2,475      skipped_too_large      1,172
inaccurate               751      structurally_singular    304
load_failed               96      diverged                  65
```

`inaccurate` = the solver returned without complaint and the residual says the answer is
wrong. This category did not exist before; 756 attempts land in it, 57 of them
Gauss-Jordan.

---

## 2. The dispatch rule contributes one matrix

```
ILU-Krylov (dispatched)   617 of 736 solved      symmetry → PCG, else BiCGSTAB
ILU-BiCGSTAB (always)     616 of 736 solved      no dispatch at all
                          ───────
                          +1 matrix  (0.14%)
```

This is the entire measured contribution of the rule the proposal called novel.
The `ILU only` control (79 solved, 10.7%) shows the preconditioner alone is not the
story either — the Krylov iteration does the work.

---

## 3. The base paper's test on real matrices

Denominator is **411**, the systems where both methods are defined. Jacobi,
Gauss-Seidel and SOR are undefined on 491 of 927 — a zero on the diagonal makes the
update rule meaningless.

| | count |
|---|---|
| both converge | 75 |
| only Gauss-Seidel | 46 |
| **only Jacobi** | **0** |
| neither | 293 |

Khrapov & Volkov report 1,095 only-Jacobi cases among random matrices at n ≤ 5.

---

## 4. Why that zero happens — the result worth publishing

Convergence concentrates almost entirely in the classes where classical theorems apply.

| Hypothesis class | n | Jacobi converges | GS converges | only-Jacobi |
|---|---|---|---|---|
| strictly diagonally dominant | 43 | 88.4% | 88.4% | 0 |
| H-matrix | 83 | 77.1% | 80.7% | 0 |
| M-matrix | 34 | 64.7% | 76.5% | 0 |
| SPD | 101 | 27.7% | 51.5% | 0 |
| **none — no theorem applies** | **192** | **1.6%** | **9.9%** | **0** |

**Where a theorem applies, Jacobi works 65–88% of the time. Where none applies, 1.6%.**

Of the 75 systems where both converge, **72 (96%) are covered by some classical class**,
against a base rate of 53.6% across the applicable set.

Coverage across the 905 analysed: none 680, SPD 81, property-A proxy 65, M-matrix 34,
H-matrix 27, strictly diagonally dominant 14, L-matrix 4.

### Householder–John is true and nearly useless

The 1958 theorem guarantees Gauss-Seidel converges for SPD matrices. On 104 real SPD
systems, ρ(T_GS) < 1 holds on 98 — **the theorem is confirmed**. Within a
10,000-iteration budget it delivers on **44**.

```
verdict      solved   did_not_converge
converges        44                  0     ← perfect precision
too_slow          7                 47
diverges          1                  2

the 54 too_slow systems: rho(T_GS) from 0.9983961071 to 1.0000000000
median iterations they would need: 1,798,184        max: 99,352,540,041,158
```

This gap between *converges* and *converges usefully* is invisible at n ≤ 5, which is all
the base paper ever tested.

---

## 5. Spectral prediction validated

| | agreement | predicted converge, did not | predicted not, but did |
|---|---|---|---|
| Jacobi | **99.3%** (899/905) | 1 | 5 |
| Gauss-Seidel | **98.2%** (889/905) | 2 | 14 |

Three-way verdicts across 905:

```
Jacobi        diverges 794   converges 71   too_slow 40
Gauss-Seidel  diverges 703   converges 109  too_slow 93
```

The textbook criterion is binary (ρ < 1 or not) and cannot express `too_slow`.
Worst case found: `nos7`, ρ(T_J) = 0.999999984536822 — provably convergent, and
1,191,260,985 iterations from a 1e-8 residual.

### One prediction "failure" that was ours, not the theory's

The `converges` column shows 70 of 71 for Jacobi. The single miss is `cdde6`, and it is
not a theory failure:

```
rho(T_J) = 0.717092          ||T_J||_2 = 1.244        <- non-normal
residual at iteration   1 :  1.00e+00
residual at iteration  73 :  4.48e+04   <- peak, 44,764x the start
residual at iteration 178 :  9.35e-09   <- converged
our divergence guard aborted it at iteration 61, on the way up
```

rho(T) governs the *asymptotic* rate. When the iteration matrix is non-normal, ||T^k||
can grow a long way before it decays, so a run can look divergent for a hundred
iterations and still converge. The guard fired at 1e4 times the running best; the hump
peaks at 4.5e4.

Measured cost across the three stationary methods: about **23 legitimate convergences
lost**, 17 of them SOR, since over-relaxation amplifies exactly this transient. The guard
is otherwise doing its job -- on genuinely divergent runs it fires at a median of
**2 iterations**. The threshold is now 1e12, which still catches true divergence within a
few extra iterations because those grow by orders of magnitude per step.

Corrected, theory's `converges` column is 71 of 71 for Jacobi.

---

## 6. Refinement, given to every method

Median relative forward error, 200-matrix stratified sample (seed 20260909):

| Method | 0 passes | 1 pass | 2 passes | matvecs (0 → 1) |
|---|---|---|---|---|
| **Jacobi** | 1.893e-08 | **6.985e-16** | 4.177e-16 | 59 → 279 |
| Gauss-Seidel | 4.429e-08 | 5.256e-15 | 1.348e-14 | 46 → 530 |
| SOR | 3.417e-08 | 9.746e-15 | 2.309e-15 | 31 → 1249 |
| ILU-Krylov (dispatched) | 2.809e-07 | 7.111e-14 | 5.019e-14 | 4 → 12 |
| ILU-BiCGSTAB | 3.265e-07 | 6.135e-14 | 5.019e-14 | 4 → 10 |
| spsolve (SuperLU) | 4.879e-12 | 6.099e-13 | 5.734e-13 | 0 → 1 |

**Give every method one refinement pass and Jacobi becomes the most accurate method in
the benchmark** — 6.985e-16 against the dispatched method's 7.111e-14, a factor of 100.

The first sweep gave refinement to one method only, and that is the whole of its
accuracy result. Cost remains a separate axis: Jacobi needs 279 matvecs where
ILU-Krylov needs 12.

Solved counts: 1,355 → 1,376 → 1,412 across 0, 1, 2 passes.

---

## 7. What changed from the first sweep, and why

| Method | First sweep reported | Corrected | Cause |
|---|---|---|---|
| Conjugate Gradient | 10.4% | **84.5%** | divided by 930 instead of the 110 it applies to |
| Cholesky | 10.1% | **96.8%** | same |
| Gauss-Seidel | 13.4% | **29.2%** | same |
| Direct methods | "97.9% success" | 90.4–99.5% | success was recorded without checking the answer |

Other corrections:

* **Gauss-Seidel/SOR were paying a 11–14× implementation penalty** — `spsolve_triangular`
  was being called inside the iteration loop, re-analysing the triangular structure up to
  10,000 times. Rewritten in delta form; verified identical output.
  Measured: `fs_541_3` 0.64 s → 0.04 s (2,472 iterations).
* Success is now decided by the harness from the residual it measures itself, never by a
  solver's own convergence flag.
* 91.3% of all stationary-solver time in the first sweep was spent on runs that never
  converged: 2,022,191 iterations, 1.36 h.

---

## 8. The contribution — a convergence-oriented preprocessing pipeline

**Headline: the pipeline raises stationary solver success by 55-72% across 927 real
matrices — Jacobi 75 to 124, Gauss-Seidel 121 to 208, SOR 119 to 184. 199 systems
recovered, none lost.** Everything below is the evidence for that, and the limits of it.


Run completed 2026-09-10 on Kaggle: **927 matrices, 3 conditions, 3 methods, 114.7
minutes**. `results/tables/reordering_study.csv`.

### What cannot work (proved, then confirmed)

* Row scaling: `T_J' = I − (RD)⁻¹RA = T_J`. Spectrum unchanged, **exactly**.
* Column scaling: similarity transform on `T_J`. Spectrum unchanged.
* Symmetric permutation: leaves ρ(T_J) invariant; moved ρ(T_GS) only in the 5th decimal
  on real matrices (RCM and random both measured: 0.99012 → 0.99011).

**Row permutation is the only lever**, because it alone changes which entries form D.

### The objective

MC64 maximises the product of |diagonal| entries — built for pivot stability in *direct*
solvers. Stationary convergence instead needs a small row ratio `Σ_{j≠i}|a_ij| / |a_ii|`.
Minimising the **worst** such ratio is a bottleneck assignment: binary search on a
threshold plus a matching feasibility test, solved exactly.

### Result 1 — choosing the diagonal works, and works large

| condition | systems solved |
|---|---|
| as given | 317 |
| MC64 | 514 |
| ours (portfolio) | **516** |

**199 systems rescued, 0 lost.** Per method: Jacobi 76 → 124, Gauss-Seidel 122 → 208,
SOR 119 → 184. A 63% increase in what the stationary methods can solve, from reordering
alone. The portfolio never scored below the control on any of 2,781 (matrix, method)
pairs, as its construction requires.

### Result 2 — but that is MC64's win, not this objective's

| | (matrix, method) pairs |
|---|---|
| rescued by both | 195 |
| **rescued by ours only** | **4** |
| rescued by MC64 only | 2 |
| net gain over MC64 | **+2** |

The four are `iprob` (Gauss-Seidel, SOR), `mcca` (Gauss-Seidel), `odepa400`
(Gauss-Seidel) — all chosen by bottleneck. The two losses are the portfolio's cheap
power-iteration estimate ranking a worse candidate first.

**This is the APK situation again.** That dispatch rule was worth one matrix in 736; this
objective is worth two pairs in 197. Reordering is the real effect; the *choice of
objective* barely moves it.

### Result 3 — on its own objective the method wins outright

Head to head against MC64 on the worst row ratio, 873 matrices where both are finite:

```
bottleneck strictly better   367
MC64 strictly better           0        <- never
identical                    506
```

Median improvement where it wins: **1.55x**. Geometric mean ratio 535 → 379. The method
does exactly what it is designed to do, and does it one-sidedly. Figure `16_dominance`.

### Result 4 — the guarantee is out of reach on real matrices

A worst row ratio below 1 is strict diagonal dominance and forces both Jacobi and
Gauss-Seidel to converge. Bottleneck minimises that ratio exactly, so it reaches the
threshold whenever any row permutation can (verified against brute force, 400/400).

```
already dominant as given                     43
made dominant by permutation                   0
dominance lost by permutation                  0
```

**Zero.** Not one matrix in 927 that is not already diagonally dominant can be made so by
permuting its rows. The theorem is true and, on this corpus, empty. The gain in Result 3
is real but lands entirely in a regime — ratios of 90 against 97 — where it does not
decide convergence.

### Result 5 — min-sum is MC64, algebraically

`1 + R[i,j] = rowsum_i / |a_ij|`, so

```
Σ_i log(1 + R[i,p(i)])  =  Σ_i log(rowsum_i)  −  Σ_i log|a_{i,p(i)}|
                           └── independent of p ──┘
```

Minimising the left side is exactly maximising `Σ log|a_ii|`, which is MC64's objective.
The two are the same optimisation. Verified: 193 of 193 random matrices reach an
identical MC64 objective value, none differ.

Earlier drafts of this file described min-sum as "the natural alternative to contrast
with bottleneck". That was wrong — it was never a contrast, and the portfolio had two
distinct objectives rather than three. The selector picking `minsum` 64 times and `mc64`
489 times is one objective winning 553 times under two names.

### What the selector chose

```
mc64        489        none        250
bottleneck   99        minsum       64
```

### Honest summary

The contribution is a new *objective* for row permutation with an exact algorithm and a
theorem, measured against the established tool on a full corpus. It beats MC64 on the
quantity it targets, 367–0. It does not beat MC64 on systems solved, and the threshold
where its guarantee would bite is unreachable on real matrices.

The publishable finding is therefore a negative one, and a clean one: **minimising the
row ratio is provably achievable and does not translate into convergence on real sparse
matrices.** The measurement that reordering itself rescues 63% more stationary solves
across 927 matrices stands on its own and, as far as we know, has not been reported at
this scale.

### Step 2 measured on its own: the relaxation factor

`SOR` runs at a fixed `omega = 1.25` for every matrix in the main sweep, and that is the
wrong call in both directions at once: plain Gauss-Seidel solves 11 systems SOR(1.25)
misses, and SOR(1.25) solves 9 that Gauss-Seidel misses. **So 20 systems are decided by
the relaxation factor alone** — that is the whole headroom available to any adaptive rule.

Choosing `omega` per matrix from a power-iteration estimate of rho(T_J), by Young's
formula, over 400 matrices with a usable diagonal:

| | solved |
|---|---|
| Gauss-Seidel (omega = 1) | 122 |
| SOR, fixed 1.25 | 119 |
| **SOR, omega from rho(T_J)** | **127** |

It rescues 16 and loses 8: **+8 net, 40% of the 20-system ceiling.** Total iterations
across the cases all three solve fall from 121,886 to 112,630, a median speedup of 1.21x
with individual cases far larger — `sherman1` goes from 8,730 iterations to 1,658.

Young's formula is exact for a consistently ordered matrix with property A, and most of
this corpus is neither. What is measured here is not the formula but whether it is worth
applying outside its hypotheses. It is, modestly. Its 60 power iterations are charged to
the method as matvecs.

### Known negatives — reported, not hidden

* `d_ss`: bottleneck made it **worse**, ρ(T_GS) 1.884 → 29.605. A min-max objective
  protects the worst row and can sacrifice the rest.
* The portfolio's power-iteration estimate misranks near-ties and cost 2 rescues.
* Six corpus files are truncated downloads and never loaded (§9).

## 9. Claims that must not be made

1. APK / the dispatch rule is novel — it is Barrett et al. (SIAM, 1994), and it
   contributes **1 matrix in 739**.
2. "Zero only-Jacobi across 924 matrices" — the denominator is **414**.
3. The dispatched method is the most accurate — with refinement given to everyone,
   **Jacobi** is, by 100×.
4. Any classical iterative method is competitive on reliability — `spsolve` solves 876
   against 619.
5. "97.9% direct-method success" — that figure came from not checking answers.
6. Scaling or symmetric reordering can help stationary convergence — proved they cannot.
