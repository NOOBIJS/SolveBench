# Figures

- `01_method_scoreboard.png` -- Applicability (how much of the corpus a method is defined on) against conditional success (how much of that it solves). Multiplying the two into one rate is what understated several baselines.
- `02_outcomes.png` -- 'inaccurate' is the category that did not exist before: the solver returned without complaint and the residual says the answer is wrong.
- `03_domain_heatmap.png` -- Grey cells are domains where a method is defined on nothing. Rates are conditional; per-domain denominators are in per_domain.csv.
- `04_cost_profile.png` -- Counted work rather than wall-clock time. Direct methods sit at a constant: they do a fixed amount of work regardless of conditioning.
- `05_conditioning.png` -- Pooling the two strata is what let dataset ill-conditioning masquerade as solver inaccuracy in the earlier error medians.
- `06_dispatch_ablation.png` -- If the dispatched bar matches always-BiCGSTAB, the dispatch rule adds nothing. Earlier probes put both at 51/70.
- `11_jacobi_vs_gauss_seidel.png` -- The denominator is matrices where both methods are defined, not the whole corpus -- Jacobi and Gauss-Seidel are undefined wherever the diagonal carries a zero.
- `07_spectral_radius.png` -- The quantity the base paper exists to characterise, computed here for every matrix in the corpus.
- `08_hypothesis_coverage.png` -- Stein-Rosenberg (1948) forbids Jacobi-converges-while-Gauss-Seidel-does-not for M-matrices; Householder-John (1958) guarantees Gauss-Seidel for SPD. Coverage of these classes is what explains the benchmark's own headline.
- `09_prediction_vs_observation.png` -- 'too_slow' is the case the textbook criterion cannot express: rho < 1, so convergence is guaranteed, but not within any usable iteration budget.
- `10_refinement_effect.png` -- In the first sweep only one method received refinement, and that is where its accuracy advantage came from.
