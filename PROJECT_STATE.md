# SolveBench — Working State

**What this file is for.** `RESULTS.md` records what was *measured*. This one records where
the work stands, what was decided and why, and what is still open — the things a fresh
reader (or a fresh session) would otherwise have to rediscover.

Last updated: **2026-09-10**

---

## 1. Where the project stands

The course project is complete end to end except for the novel method's headline number.
Publication is a later, separate goal and is **not** driving decisions right now.

| Piece | State |
|---|---|
| Library (`src/solvebench/`, 16 methods) | done, 40 tests pass |
| Main sweep, 927 × 16 | done, 5.19 h |
| Spectral analysis | done, 0.95 h |
| Refinement study | done, 3.30 h |
| **Reordering study (the novel method)** | **not done** — see §4 |
| Figures (13) | done, audited, 8 real bugs found and fixed |
| `linear.tex` | 4 "Update" slides appended, proposal slides untouched |
| Report | not started |

Repository: <https://github.com/NOOBIJS/SolveBench> — public, commits authored as
`IJS Asif <2105038@cse.buet.ac.bd>`, no co-author trailers.

---

## 2. The novel method

**Convergence-oriented diagonal selection.** A stationary method divides by `a_ii`, so its
behaviour is decided by which entries sit on the diagonal — and that came from the order
rows happened to arrive in, not from any thought about convergence.

Proved and then confirmed numerically: **row permutation is the only lever**. Row scaling
gives `T_J' = I − (RD)⁻¹RA = T_J` exactly; column scaling is a similarity transform;
symmetric permutation leaves ρ(T_J) invariant and moves ρ(T_GS) only in the fifth decimal.

MC64 optimises the wrong objective for this purpose — the *product* of |diagonal| entries,
built for pivot stability in direct factorization. Convergence needs a small row ratio
`Σ_{j≠i}|a_ij| / |a_ii|`. Three objectives are implemented, plus a portfolio:

```
none        control
mc64        max Π|a_ii|          the established tool
minsum      min Σ log(1+ratio)
bottleneck  min max ratio        the idea
best        computes all four, ranks by an estimated rho(T_GS), keeps the winner
```

`best` cannot lose, because "no permutation" is one of its candidates. Evidence on the
probe matrices is in `RESULTS.md` §8; no single objective dominates, which is why the
portfolio exists.

---

## 3. Three scipy routines that could not be trusted

This cost more time than anything else in the project and is worth stating plainly.

| Routine | Failure |
|---|---|
| `min_weight_full_bipartite_matching` | hung on `nnc261` with raw ratios; hung on `west0067` (67×67, 294 nnz) under the log transform introduced to fix the first hang; 9.7 s on `oscil_dcop_23` (n=430) |
| `maximum_bipartite_matching` | 7.2 s for one call on `bcsstk19` (n=817, 3,764 edges), and the bottleneck search makes ten. Slowest exactly when a matching *does* exist |

All three now go through `scipy.optimize.linear_sum_assignment` on a dense array — MC64 and
min-sum as minimum-cost assignments, bottleneck feasibility as a 0/1 assignment where a
total of zero means every row found an allowed column.

```
bcsstk19      bottleneck   90 s+ hang  →  0.07 s
oscil_dcop_23 mc64              9.7 s  →  0.006 s      1,600x
```

Above `DENSE_ASSIGNMENT_CAP = 5000` the n×n array is too big: MC64 falls back to the sparse
routine, min-sum is not offered, and the portfolio picks among fewer candidates. 130 of 927
matrices are affected.

**A hang cannot be interrupted from Python.** The only defence is to try every matrix
locally first — `tools/preflight_reordering.py`.

---

## 4. What happened to the reordering run

* Pushed to Kaggle, ran **12.7 h**, hit the 12-hour limit, `CANCEL_ACKNOWLEDGED`.
* Kaggle still published the checkpoint CSV, which is how the cause was found.
* It had completed **30 matrices of 930**. The first 29 took 0.0 min in total; it then
  stalled inside a permutation on `west0067` and never moved again.
* A second, smaller "probe" kernel (200 stratified matrices, no exact spectra) was started
  in parallel as insurance. It is still running under the old account with the *unfixed*
  code, so it is probably stuck the same way. Cancelling it still yields its checkpoint.

Why the study was so much slower than the 16-method main sweep, which took 5.19 h: in that
sweep the stationary methods cost **15.5 min of 294**, because 1,473 of their 2,781 attempts
returned `not_applicable` instantly on a zero diagonal. Under `mc64` and `best` those same
491 systems have a working diagonal and actually run — and they are the hardest matrices in
the corpus, so most go the full 10,000 iterations. **The method working is what makes it
expensive.**

---

## 5. Decisions taken

**Kaggle account switched to `ijsasif`** (`E:\L-4-T-1 Study Materials\Kaggle Json\kaggle.json`)
so the user can read kernel logs directly. Both accounts are configured side by side:

```
KAGGLE_CONFIG_DIR=~/.kaggle-ijsasif    the new account, where kernels now go
KAGGLE_CONFIG_DIR=~/.kaggle-old        mdmarufhasanrubab, owns the corpus dataset
```

The corpus dataset was **made public** rather than re-uploaded — it is SuiteSparse data, and
a public dataset helps reproducibility. Its title was corrected from "830, 26 domains" to
"927 unique, 26 domains".

**Progress logging rewritten.** The old notebook printed one line per matrix only after all
three conditions finished, so a stall inside a permutation appeared as silence — twelve
hours of it, with no way to tell which matrix or step was responsible. Each matrix now
announces itself *before* the work, with elapsed time and ETA, and each condition reports as
it completes. The last line in the log always names whatever is stuck.

**Divergence guard raised 1e4 → 1e12.** ρ(T) is the asymptotic rate and says nothing about
the transient: `cdde6` has ρ(T_J) = 0.717 with ‖T_J‖₂ = 1.244, climbs to 44,764× its starting
residual by iteration 73, and converges at 178. The old threshold killed it at 61. Cost of
the old value: about 23 legitimate convergences lost, 17 of them SOR.

**`convergence_verdict` gained a fourth case.** NaN ρ means the iteration matrix does not
exist (zero diagonal), which is not the same as diverging. That conflation had put 491
systems into Jacobi's "diverges" column.

---

## 6. Open threads

1. **Preflight** over all 930 matrices is running locally. Worst call so far 2.41 s
   (`laser`, n=3002); no stalls since the fix. Must finish clean before anything is pushed.
2. **Re-run the reordering study** on `ijsasif` once preflight passes. Needs the user's
   confirmation before pushing, as always.
3. **99 missing matrices.** SuiteSparse has 930 usable matrices at n ≤ 10,000; the corpus
   has 831. Downloading the rest (73 MB, 2.9 M nnz) would make it a *complete census* rather
   than a sample — a much stronger claim — and adds 6 domains that were never downloaded at
   all. The user asked to do this **after** the reordering run finishes.
4. **Small domains.** Eight have fewer than 5 matrices. Verified against `ssstats.csv` that
   this is SuiteSparse's own scarcity, not under-sampling: only 3 more exist across all of
   them. `linear_programming` has 342 matrices but exactly **one** square — LP constraint
   matrices are rectangular by nature. Decision pending; recommendation is to keep them and
   always report n.
5. **Report** not started. Course requires a presentation and a report; contents were never
   specified by the teacher and the user should ask.

---

## 7. Things that must not be re-broken

* Success is decided by `metrics.score` from the residual, never by a solver's own flag.
* Applicability and conditional success are separate numbers and are **never multiplied**.
  Every figure divides by `APPLICABLE`, never by the corpus.
* Refinement is an orthogonal factor available to every method, not one method's feature.
* The notebook is generated from `src/solvebench/` by `tools/build_notebooks.py`. Never edit
  a notebook directly — that is how the library and the notebook drifted apart before.
* `tools/test_notebooks.py` executes every generated cell locally. Nothing goes to Kaggle
  without it passing; it has caught a missing import and two hangs.
* `kaggle.json` is git-ignored and must stay that way.
