# CSE 402 Project Idea Bank — RECENT PAPERS (2023–2026)

Companion to [papers.md](papers.md). Same constraints: 5-person team, **no ML/DL**, classical deterministic numerical computing, fast to implement, presentable, scalable to heavy visible work. This file specifically prioritizes **base papers published 2023 or later** (mostly 2024–2026) so the proposal cites current literature rather than decades-old classics.

All links below were fetched/verified live (or cross-confirmed via search-index metadata where a publisher paywall blocked direct fetch — noted explicitly where that happened).

---

## ⭐ Top Recommendation (recent-papers edition)

**Theme: Newton-Raphson / Root-Finding, Modernized — "Annealed" Newton-Raphson + Kepler's-Equation Initial Guesses**

Two 2024 papers modernize the exact root-finding methods your syllabus already covers (bisection, false position, Newton-Raphson — week 3-4), giving you a **Methodological Extension** and a **Cross-Domain Application** that share one codebase:

- **Methodological Extension:** implement the "annealed" Newton-Raphson variant (a temperature-like relaxation parameter added to the classic update rule) and benchmark its robustness/convergence-speed gains against plain Newton-Raphson, bisection, and false position.
- **Cross-Domain Application:** apply root-finding (classic + the annealed variant + Napier's improved initial guesses) to Kepler's equation for real asteroid/comet orbital elements from NASA JPL — a genuine astrodynamics problem, at real scale (hundreds to thousands of bodies).

**Why this is the easiest strong 2024-literature pick:**
- Both papers are single-author-scale ideas — a small, clean modification to code you're already writing for Lab Assignment 1.
- Fully deterministic: given (M, e), Kepler's equation has one exact root E; given a test function, Newton-Raphson has one converged root — no ambiguity, instantly checkable.
- Scales for free: run across hundreds of orbital bodies (NASA JPL) × dozens of synthetic test functions (for the annealed-NR benchmark) — easy 5-way split (one person per method/variant + one on the JPL data pipeline).

**Papers (both verified live, 2024):**
- J. Jo, A. Wagemakers, V. Periwal, *"Annealing approach to root-finding,"* arXiv:2404.15338, submitted Apr 10 2024 (v2 Aug 1 2024) — https://arxiv.org/abs/2404.15338 — also on PubMed: https://pubmed.ncbi.nlm.nih.gov/38903748/
- K.J. Napier, *"Improved Initial Guesses for Numerical Solutions of Kepler's Equation,"* submitted to the Open Journal of Astrophysics, arXiv:2411.15374, Nov 22 2024 — https://arxiv.org/abs/2411.15374

**Dataset:** NASA JPL Small-Body Database Query tool (verified live) — https://ssd.jpl.nasa.gov/tools/sbdb_query.html — orbital elements for 1.3M+ asteroids/comets, exportable as CSV.

---

## Category 1 — Cross-Domain Application (recent papers)

### 1A. Kepler's Equation with Modern Initial-Guess Methods ⭐ (see top recommendation)
Covered above.

### 1B. Stochastic Newton-Raphson / Gradient-Descent Parameter Estimation for ODE Systems
**Paper:** S. Syafiie, A. Subiantoro, V. Andasari, F. Tadeo, *"Systems of ODEs Parameters Estimation by Using Stochastic Newton-Raphson and Gradient Descent Methods,"* arXiv:2501.12856, submitted Jan 22 2025 — https://arxiv.org/abs/2501.12856 (verified: proposes NR and gradient-descent methods, including stochastic variants, for ODE-system parameter estimation; finds NR converges fast while GD is more robust on chaotic systems).
**Application:** fit this to a real domain ODE system — e.g., an epidemiological SIR/SEIR model on JHU COVID-19 data (https://github.com/CSSEGISandData/COVID-19), or a chemical-kinetics/pharmacokinetic ODE system with published rate-constant data.
**Methodology:** (1) discretize target ODE system; (2) implement plain Newton-Raphson and gradient-descent parameter fits (syllabus week 3-4 root-finding + week 9-10 optimization); (3) add the paper's stochastic variants (mini-batch-style updates over data subsets); (4) run across many countries/datasets; (5) compare convergence speed and robustness, especially on noisy/chaotic-like real data.
**Showcase results:** fitted-vs-actual curves across dozens of datasets, NR-vs-GD convergence-speed comparison table, robustness case study on noisy data.
**Why deterministic:** fixed data + fixed method → one converged parameter set; correctness = residual error, not a predictive accuracy score.

### 1C. Monte Carlo / Quasi-Monte Carlo Derivative Pricing on Real Market Data
**Paper:** G. Case, *"Comparative Study of Monte Carlo and Quasi-Monte Carlo Techniques for Enhanced Derivative Pricing,"* arXiv:2502.17731, submitted Feb 24 2025 — https://arxiv.org/abs/2502.17731 (verified: compares MC vs QMC — Sobol'/Faure sequences — for geometric basket and Asian call options under Black-Scholes, in 5-D).
**Dataset:** Kenneth R. French Data Library (verified live) — https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html — real historical portfolio returns to parametrize volatility/drift inputs.
**Methodology:** estimate σ, μ from real historical returns via least-squares on log-returns → implement naive Monte Carlo, Metropolis-Hastings, and Sobol/Faure quasi-Monte Carlo estimators → price basket/Asian options → compare against closed-form Black-Scholes benchmark and each other's convergence rate as sample size N grows.
**Showcase results:** error-vs-N convergence curves for MC vs QMC (O(N⁻¹ᐟ²) vs O(N^(1-ε))) across many portfolios/strikes, high-dimension breakdown demonstration (QMC degrades beyond ~70 dimensions — a nice "where each method wins" story).
**Why deterministic:** fixed seed + fixed market inputs → estimator converges provably to the known closed-form price; this is a convergence-rate study, not a prediction-accuracy study.

### 1D. Eigenvector Centrality for Power-Grid Security (Power Method Applied to Infrastructure)
**Paper:** *"Eigenvector centrality-enhanced graph network for attack detection in power distribution systems,"* Electric Power Systems Research / published via ScienceDirect, 2024 — https://www.sciencedirect.com/science/article/pii/S0378779624012252 (confirmed live via search index; direct fetch blocked by publisher paywall — recommend the team open/verify via institutional or library access before finalizing).
**Application:** implement the eigenvector-centrality (power method) core — syllabus week 5-6 — on real power-grid/transmission-network topology data, using it to rank critical nodes/buses by structural importance, independent of the paper's own graph-network-detection layer (which the team need not reproduce).
**Dataset:** IEEE test-case power system networks (standard, freely available, e.g. IEEE 14/30/57/118/300-bus test cases) via the Power Grid Lib: https://github.com/power-grid-lib/pglib-opf ; alternatively SNAP infrastructure networks: https://snap.stanford.edu/data/.
**Methodology:** build the grid admittance/adjacency graph → compute eigenvector centrality via power iteration → identify top-k critical buses → simulate node/line removal (single-point-of-failure analysis) → recompute centrality to show cascading importance shifts.
**Showcase results:** ranked criticality tables across all IEEE test-case sizes, before/after-failure centrality-shift visualizations, convergence plots.
**Why deterministic:** power iteration on a fixed graph converges to the same ranking every run — pure linear algebra.

---

## Category 2 — Methodological Extension (recent papers)

### 2A. Annealed Newton-Raphson ⭐ (see top recommendation)
Covered above — arXiv:2404.15338.

### 2B. Constrained-Optimization Reformulation of Bairstow's Method (very recent — 2026)
**Paper:** *"A Constrained Optimization Approach to Bairstow's Method,"* Algorithms (MDPI), Vol. 19, Issue 1, Article 50 — submitted Nov 2025, published Jan 2026. DOI: 10.3390/a19010050 — https://doi.org/10.3390/a19010050 → https://www.mdpi.com/1999-4893/19/1/50 (bibliographic details — title/journal/DOI/dates — cross-confirmed via search index; MDPI page itself returned 403 to automated fetch, so have the team open it directly in a browser to pull the full text).
**Extension idea:** this is the single most recent paper found in this entire search. It reformulates classical Bairstow's method (quadratic-factor extraction, syllabus week 3-4) as a constrained optimization problem for improved stability. A team can: (1) reimplement classical Bairstow; (2) reimplement the constrained-optimization reformulation; (3) extend further with adaptive initial-guess selection (e.g., Cauchy-bound-based starting points) and test different deflation orders; (4) validate on ill-conditioned polynomial benchmarks.
**Test data:** Wilkinson's polynomial (classic ill-conditioned benchmark with known integer roots), Chebyshev polynomials, randomly generated polynomials with clustered/multiple roots.
**Showcase results:** root-accuracy tables across polynomial degree/conditioning, deflation-order comparison, convergence-iteration heatmaps.
**Why deterministic:** exact known roots for validation (e.g. Wilkinson's polynomial); no randomness.

### 2C. Intrepid MCMC — Coordinate-Transform Exploration for Metropolis-Hastings
**Paper:** P. Chakroborty & M.D. Shields, *"Intrepid MCMC: Metropolis-Hastings with Exploration,"* arXiv:2411.17639, submitted Nov 26 2024 — https://arxiv.org/abs/2411.17639 (verified: proposes a coordinate-transformation scheme improving mode-finding on multimodal distributions, validated on a Bayesian 2-DOF oscillator inference problem).
**Extension idea:** implement standard random-walk Metropolis-Hastings (syllabus week 9-10) vs. the paper's coordinate-transform "Intrepid" variant on multimodal target distributions (mixture-of-Gaussians, Rosenbrock/banana density); measure mode-discovery rate, acceptance rate, and mixing time.
**Test problems:** standard synthetic MCMC benchmark targets (multimodal mixtures, banana-shaped density) — no external dataset needed; optionally validate on the paper's own 2-DOF oscillator Bayesian-inference example.
**Showcase results:** trace plots showing mode-jumping (or failure to jump) for plain MH vs. Intrepid MH, ESS-per-second bar charts, 2D contour overlays of sampled vs. true multimodal density.
**Why deterministic:** fixed seeds → exactly reproducible chains; the transform itself is a modest addition (~30-40 lines) to a standard MH sampler.

### 2D. Momentum-Accelerated Power Iteration as a Lanczos Preconditioner
**Paper:** A. Barletta, N. Marshall, S. Pollock, *"Momentum accelerated power iterations and the restarted Lanczos method,"* arXiv:2511.05364, submitted Nov 7 2025 (v2 Mar 1 2026) — https://arxiv.org/abs/2511.05364 (verified: analyzes convergence regimes of momentum-accelerated power iteration vs. restarted Lanczos, and proposes using momentum-accelerated power iteration as a preconditioning step for restarted Lanczos).
**Extension idea:** implement plain power method, momentum-accelerated power method, and restarted Lanczos (syllabus week 5-6 eigenvalues + QR method) → use the paper's preconditioning idea (seed Lanczos with a momentum-accelerated power-iteration estimate) → benchmark convergence speed across the regimes the paper identifies (dominant/subdominant eigenvalue ratio).
**Test data:** SuiteSparse Matrix Collection (verified live, 2,904 real matrices) — https://sparse.tamu.edu/ — pick matrices spanning a range of spectral gaps.
**Showcase results:** convergence-rate-vs-spectral-gap plots reproducing/extending the paper's theoretical regimes, iteration-count comparison table across dozens of SuiteSparse matrices, preconditioned-vs-unpreconditioned Lanczos speedup chart.
**Why deterministic:** pure linear algebra on fixed matrices; exactly reproducible; each variant is a short modification of the same iterative loop.

---

## Category 3 — Exploratory & Comparative Study (recent papers)

### 3A. Seven-Way Root-Finding Method Shootout (2025)
**Paper:** M.N.H. Romadani, *"From Theory to Code: Transforming Classical Root-Finding Methods into Efficient Python Implementations,"* ICCK Journal of Applied Mathematics, 2025 — https://www.icck.org/article/abs/jam.2025.840767 (verified: comparatively evaluates Bisection, Regula-Falsi, Fixed-Point Iteration, Newton-Raphson, Secant, Aitken's Δ², and Steffensen's method on convergence speed, accuracy, stability, and runtime, using Python/NumPy/SymPy/Pandas/Matplotlib).
**Extension for scale:** reproduce all 7 methods, then scale far beyond the paper's own test set — run on 50-100+ functions (polynomial, trigonometric, exponential, mixed; add pathological/multi-root cases), and add Bairstow's method for the polynomial subset (syllabus completeness).
**Test data:** synthetic test-function suite (no external dataset needed) — expandable with real engineering root-finding problems (e.g. Colebrook friction-factor equation, Van der Waals equation of state).
**Showcase results:** methods × functions convergence-iteration heatmap, accuracy/stability ranking table, computational-time-efficiency bar charts reproducing and extending the paper's own findings at much larger scale.
**Split:** 1-2 methods per person + 1 person on the extended test-function library and benchmarking harness.
**Why deterministic:** true roots known analytically; correctness is binary, instantly checkable; no GPU/training.

### 3B. 16-Solver Shootout on Ill-Conditioned Real Geophysical Matrices (2024)
**Paper:** M. Ferrari, *"A Comparison of Sparse Solvers for Severely Ill-Conditioned Linear Systems in Geophysical Marker-In-Cell Simulations,"* arXiv:2409.11515, submitted Sep 17 2024 (revised Sep 23 2024) — https://arxiv.org/html/2409.11515v1 (verified: benchmarks 16 solvers from 11 numerical libraries — including PARDISO, UMFPACK, MUMPS, and classical iterative methods — on real ill-conditioned matrices from geophysical marker-in-cell simulations; introduces a "Projected Adam" method to estimate condition numbers without full eigen/singular-value decomposition).
**Extension for a student team:** implement the syllabus's own solvers — Gauss elimination, Gauss-Jordan, LU decomposition (direct) and Jacobi, Gauss-Seidel, Conjugate Gradient (iterative) — and reproduce the paper's benchmarking methodology (condition-number sweep, residual/backward-error measurement) on both the paper's problem class and the broader SuiteSparse Matrix Collection (https://sparse.tamu.edu/, 2,904 real matrices) for much larger scale than the original paper.
**Showcase results:** runtime-vs-condition-number plots across dozens/hundreds of matrices, reliability heatmap (solver × matrix, success/failure/residual), reproduction of the paper's own headline finding ("which solvers survive severe ill-conditioning") at classroom scale.
**Split:** 1-2 methods per person + 1 person on the SuiteSparse/geophysical data pipeline and condition-number analysis.
**Why deterministic:** matrices ship with a known/reproducible solve; trivially scalable by looping over more matrices.

### 3C. Monte Carlo vs. Quasi-Monte Carlo Convergence Study (2025)
**Paper:** G. Case, *"Comparative Study of Monte Carlo and Quasi-Monte Carlo Techniques for Enhanced Derivative Pricing,"* arXiv:2502.17731, Feb 2025 — https://arxiv.org/abs/2502.17731 (see full verification in 1C above; used here as the comparative-methodology anchor rather than the cross-domain-application anchor).
**Methodology:** implement naive Monte Carlo, Metropolis-Hastings MCMC, and Sobol/Faure quasi-Monte Carlo estimators for (a) benchmark integrals with known closed-form answers and (b) the paper's own option-pricing test cases; sweep sample size and dimensionality (2-D up to 100-D) to empirically reproduce the paper's "QMC wins at low dimension, degrades above ~70-D" finding.
**Showcase results:** error-vs-N log-log convergence plots with theoretical rates overlaid, dimension-vs-accuracy breakdown chart, variance-reduction comparison table.
**Why deterministic:** ground truth is analytically known; the comparative claim itself ("QMC beats MC below dimension X") is directly, deterministically testable.

### 3D. Accurate Computation of the Newton Form of the Lagrange Interpolant (2024)
**Paper:** *"On the accurate computation of the Newton form of the Lagrange interpolant,"* Numerical Algorithms (Springer), 2024. DOI: 10.1007/s11075-024-01843-7 — https://link.springer.com/article/10.1007/s11075-024-01843-7 (bibliographic details — title, journal, 2024 DOI — confirmed via search index; Springer page requires institutional login for direct fetch, so have the team access via BUET library/Google Scholar to pull the full text).
**Comparative study idea:** implement Lagrange interpolation, Newton's divided-difference form, and the paper's numerically-stabilized Newton-form computation; benchmark numerical accuracy/conditioning across increasing polynomial degree on real time series (NOAA climate station data: https://www.ncei.noaa.gov/products/land-based-station/us-climate-normals ; JHU COVID-19 case curves: https://github.com/CSSEGISandData/COVID-19), directly quantifying Runge's phenomenon and showing where the stabilized Newton form avoids the numerical blow-up that naive implementations suffer.
**Showcase results:** RMSE-vs-degree curves for all three implementations across dozens of stations/countries, numerical-conditioning comparison (naive vs. stabilized Newton form), Runge's-phenomenon visual demonstration.
**Why deterministic:** held-out real data points are ground truth; the accuracy/stability comparison is a direct, reproducible numerical measurement.

### 3E. Classical Discrete-Event Queueing Benchmark, Inspired by QGym (2024)
**Paper:** H. Chen, A. Li, E. Che, T. Peng, J. Dong, H. Namkoong, *"QGym: Scalable Simulation and Benchmarking of Queuing Network Controllers,"* arXiv:2410.06170, submitted Oct 8 2024 — https://arxiv.org/abs/2410.06170 (verified: benchmarking framework across parallel-server, criss-cross, tandem, and re-entrant queueing networks; explicitly supports classical queueing policies alongside RL ones — **use only the classical-policy / simulation-environment side**, skip the RL-controller part entirely to stay non-ML).
**Methodology:** implement classical discrete-event simulation (syllabus: random number generation + discrete-event simulation, week 9-10) for the paper's own network topologies (parallel-server, tandem, criss-cross, re-entrant) using classical queueing policies (FIFO, shortest-queue, priority) instead of the paper's RL controllers; benchmark queue-length/wait-time statistics across policies and network structures via Monte Carlo replication.
**Showcase results:** wait-time/queue-length distributions across network types and policies (many replications for statistical confidence), policy-comparison dashboard, throughput-vs-load curves.
**Why deterministic:** for a fixed random seed and arrival/service-rate parameters, simulation statistics are exactly reproducible; the comparison is between well-defined classical policies, not a learned controller.

---

## Notes on Verification

- **Directly fetched and confirmed live:** arXiv:2404.15338, arXiv:2411.15374, arXiv:2501.12856, arXiv:2502.17731, arXiv:2411.17639, arXiv:2511.05364, arXiv:2409.11515, arXiv:2410.06170, icck.org/jam.2025.840767, ssd.jpl.nasa.gov, sparse.tamu.edu, mba.tuck.dartmouth.edu.
- **Confirmed via search-index metadata but blocked from direct fetch by publisher paywall/bot-protection** (verify yourselves before citing in the formal proposal): MDPI Algorithms 19(1):50 (Bairstow constrained-optimization paper, DOI resolves), ScienceDirect S0378779624012252 (power-grid eigenvector-centrality paper), Springer Numerical Algorithms 10.1007/s11075-024-01843-7 (Newton-form Lagrange interpolant paper).

## Suggested Decision (recent-papers edition)

1. **Root-finding modernization (2A/1A: Annealed Newton-Raphson + Kepler's equation)** — two clean, verified, genuinely 2024 papers; smallest implementation lift since it builds directly on syllabus Lab 1; strong visuals (orbit plots + convergence comparisons).
2. **Bairstow constrained-optimization reformulation (2B)** — the most recent paper found in this whole search (Jan 2026); directly extends a named syllabus method; novel enough to look impressive in a proposal.
3. **16-solver ill-conditioned benchmark (3B)** — biggest, most "heavy computational work" option; reproduces a real 2024 paper's benchmarking methodology at classroom scale using SuiteSparse's 2,904 matrices; natural 5-way split.

All three remain ML-free, deterministic, and reuse the exact algorithms already required by the syllabus.
