# SolveBench — Run Summary

**Generated:** 2026-08-23 06:57:33
**Runtime:** 237.2 minutes
**Environment:** Kaggle · Python 3.12.13 · 4 CPU cores

## Scope

| | |
|---|---|
| Matrices benchmarked | 930 |
| Domains | 27 |
| Solvers | 10 (4 direct, 4 iterative) |
| Total measurements | 9,246 |
| Size range | n = 5 to 10,000 |
| Successful solves | 3,197 (34.6%) |

## Outcome breakdown

| Status | Count |
|---|---|
| ok | 3197 |
| not_applicable | 2856 |
| did_not_converge | 1821 |
| skipped_too_large | 1176 |
| structurally_singular | 190 |
| matrix_load_failed | 6 |


## Key findings

1. **Jacobi vs Gauss-Seidel on real data** — both converge on 75 matrices;
   Gauss-Seidel alone on 46; Jacobi alone on 0; neither on 803.
   The base paper's claim that Gauss-Seidel usually (but not always) wins is
   tested here at a scale the paper never reached.
2. **Step count** — direct methods always take exactly n or n-1 stages, fixed by
   size alone; iterative counts vary by orders of magnitude on the same-size matrix
   depending on conditioning.
3. **Residual is not error** — see `11_residual_vs_error.png`; on ill-conditioned
   systems a solver can satisfy the equation while still being far from the answer.

## Output layout

```
solvebench_output/
  figures/   13 PNG figures
  tables/    CSV result tables
  reports/   decision_guide.md, summary_report.md
  logs/      run configuration
```

## Figure index

| File | What it shows |
|---|---|
| `00_pipeline_diagram.png` | The benchmark pipeline, mirroring the project proposal. |
| `01_dataset_composition.png` | Composition and size/sparsity spread of the benchmark set. |
| `02_sparsity_patterns.png` | Nonzero structure of a representative matrix from each domain. |
| `03_outcome_matrix.png` | Outcome of every (matrix, method) pair. Rows sorted by matrix size. |
| `04_accuracy_heatmap.png` | Relative error per (matrix, method); darker means more accurate. |
| `05_runtime_vs_size.png` | Runtime against matrix size for both method families. |
| `06_iterations_direct_vs_iterative.png` | Direct methods sit exactly on steps=n; iterative counts scatter by matrix behaviour. |
| `07_conditioning_analysis.png` | Accuracy and convergence both degrade with condition number. |
| `08_jacobi_vs_gauss_seidel.png` | The base paper's Jacobi/Gauss-Seidel comparison, re-run on real data. |
| `09_solver_scoreboard.png` | Reliability, accuracy and speed for all ten solvers, including our APK. |
| `12_paired_accuracy.png` | APK vs each rival on matrices both solved. |
| `10_domain_method_matrix.png` | Per-domain success rate for each solver. |
| `11_residual_vs_error.png` | Residual and error diverge most on ill-conditioned systems. |
