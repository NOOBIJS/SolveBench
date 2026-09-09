# CSE 402 Project Idea Bank — Numerical Analysis, Simulation & Modeling

**Team size:** 5 · **Constraint:** No ML/DL. Classical, deterministic numerical computing only.
**Goal:** Fast to implement, deterministic/reproducible results, scales to "heavy visible work" across a 5-person team, cleanly maps to syllabus methods (root-finding, Gauss/Gauss-Jordan/LU, power method/QR, Monte Carlo/Metropolis-Hastings, optimization, interpolation/regression, numerical integration, discrete-event simulation).

All paper and dataset links below were fetched/verified live during research (Aug 2026).

---

## ⭐ Top Recommendation (read this first)

**Theme: Power Method / Eigenvalue Ranking on Real Networks (SNAP datasets)**

This single theme legitimately satisfies **all three** spec categories at once, so whichever category you declare in the proposal, the same codebase/dataset carries the whole project:

- **Cross-Domain Application:** apply the power method (syllabus, week 5-6) to real web/social graphs to compute PageRank / eigenvector centrality — a numerical-linear-algebra method applied outside pure math, to network science.
- **Methodological Extension:** extend plain power iteration with **Hotelling deflation** (multiple eigenpairs) and **shifted inverse iteration** to handle clustered eigenvalues, benchmarked against the paper below.
- **Exploratory & Comparative Study:** benchmark power method vs. QR algorithm for dominant-eigenvector computation across dozens of real SNAP graphs of increasing size, measuring convergence iterations, runtime, and ranking agreement.

**Why it's the easiest high-scoring option:**
- Power iteration is ~15 lines of code; QR algorithm and deflation are modest additions.
- SNAP provides 50+ ready-to-download, pre-cleaned graphs — just loop over more graphs for "bonus-level" scale, no data cleaning pain.
- Fully deterministic (same graph + same seed vector → same ranked output every run).
- Extremely presentable: network visualizations colored by rank, top-10 tables, convergence plots, runtime-vs-graph-size charts.
- Splits perfectly across 5 people: (1) power method core, (2) deflation extension, (3) QR baseline, (4) shifted inverse iteration, (5) SNAP data pipeline + visualization dashboard.

**Core papers:**
- L. Page, S. Brin, R. Motwani, T. Winograd, *"The PageRank Citation Ranking: Bringing Order to the Web,"* Stanford InfoLab Tech Report, 1998/1999 — https://homepages.dcc.ufmg.br/~nivio/cursos/ri11/sources/pagerank.pdf (verified mirror; original http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf may be slow/legacy TLS)
- S. Brin & L. Page, *"The Anatomy of a Large-Scale Hypertextual Web Search Engine,"* Computer Networks 30, 1998, pp.107–117 — https://infolab.stanford.edu/pub/papers/google.pdf (verified)
- J.E. Gubernatis & T.E. Booth, *"Multiple Extremal Eigenpairs by the Power Method,"* arXiv:0807.1261 — https://arxiv.org/abs/0807.1261 (deflation/multiple-eigenpair extension source)

**Dataset:** SNAP — Stanford Large Network Dataset Collection (verified live) — https://snap.stanford.edu/data/
Recommended graphs: `web-Google` (875,713 nodes / 5.1M edges, verified), `web-Stanford` (281,903 nodes), `web-BerkStan`, `web-NotreDame`, `ego-Facebook`, citation and collaboration networks — all downloadable as plain edge-list `.txt.gz` files.

---

## Category 1 — Cross-Domain Application

*"Apply numerical methods, simulation models, or data fitting/optimization algorithms covered in the course to solve a problem in a different application field."*

### 1A. PageRank / Eigenvector Centrality via Power Method on Web Graphs ⭐ (see top recommendation)
Covered above.

### 1B. Orbital Mechanics — Solving Kepler's Equation for Real Solar-System Bodies
**Paper:** V. Raposo-Pulido & J. Peláez, *"An efficient code to solve the Kepler equation. Elliptic case,"* Monthly Notices of the Royal Astronomical Society 467(2), 2017, pp.1702–1713. DOI: 10.1093/mnras/stx138 — https://academic.oup.com/mnras/article/467/2/1702/2929272
**Dataset:** NASA JPL Small-Body Database Query tool (verified live) — https://ssd.jpl.nasa.gov/tools/sbdb_query.html — orbital elements (eccentricity, semi-major axis, mean anomaly) for 1.3M+ asteroids/comets, exportable as CSV; plus the 8 planets.
**Methodology:**
1. Pull orbital elements for ~200 bodies spanning eccentricity 0.01 (near-circular planets) to 0.95 (comets).
2. Solve Kepler's equation `M = E − e·sin(E)` for eccentric anomaly E via bisection, false position, and Newton-Raphson.
3. Compare iteration counts/failures, especially near high eccentricity where Newton-Raphson can diverge.
4. Reconstruct (x, y) orbital positions from E; use Lagrange/Newton divided-difference interpolation to animate motion between ephemeris timesteps.
**Showcase results:** animated orbit paths (planets vs. eccentric comets), iteration-count-vs-eccentricity plots for all 3 root-finders, divergence/failure case study.
**Why easy & deterministic:** each (M, e) pair has one exact root E, directly checkable by back-substitution — no ambiguity, no metric-chasing.

### 1C. Epidemic Curve Fitting — SIR Model Parameter Estimation (COVID-19)
**Paper:** D. Prodanov, *"Analytical parameter estimation of the SIR epidemic model. Applications to the COVID-19 pandemic,"* Entropy 23(1):59, 2021 — arXiv preprint verified: https://arxiv.org/abs/2010.07000
**Dataset:** JHU CSSE COVID-19 time series (verified, CC-BY-4.0, daily case/death counts by country) — https://github.com/CSSEGISandData/COVID-19 (file `time_series_covid19_confirmed_global.csv`); alt: https://ourworldindata.org/covid-deaths
**Methodology:**
1. Discretize the SIR ODE system into difference equations over daily case counts.
2. Fit transmission rate β and recovery rate γ via nonlinear least-squares regression.
3. Minimize the sum-of-squares cost with Newton-Raphson on the gradient/Hessian.
4. Use Simpson's/trapezoidal rule to integrate cumulative infections from fitted incidence.
5. Monte Carlo resampling of noisy case counts to build confidence bands on β, γ, R₀.
**Showcase results:** fitted-vs-actual epidemic curves for 100+ countries (20 per team member), R₀ comparison table, Newton-Raphson convergence plots, country × method error heatmap.
**Why easy & deterministic:** fixed data → one converged (β, γ) answer; pure numerical optimization, no train/test split.

### 1D. Quantitative Finance — Monte Carlo vs. Closed-Form Option Pricing
**Paper:** P.P. Boyle, *"Options: A Monte Carlo Approach,"* Journal of Financial Economics 4(3), 1977, pp.323–338. DOI: https://doi.org/10.1016/0304-405X(77)90005-8
**Dataset:** Kenneth R. French Data Library (Dartmouth Tuck, verified live, free CSV/TXT) — https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html — daily/monthly returns for 49 industry portfolios.
**Methodology:** estimate drift/volatility via least-squares on log-returns → simulate Geometric Brownian Motion via Monte Carlo → price options, compare to closed-form Black-Scholes → back out implied volatility via Newton-Raphson → optimize portfolio weights via golden-section search.
**Showcase results:** MC-convergence-vs-#simulations plots, MC-vs-Black-Scholes error tables across 45+ portfolios, implied-volatility surfaces.
**Why easy & deterministic:** fixed seed + fixed (μ, σ) → MC estimate provably converges to the known closed-form price.

---

## Category 2 — Methodological Extension

*"Take an existing paper in numerical analysis or simulation & modeling and extend its work (efficiency, relaxed assumptions, novel formulation, or enhanced stability)."*

### 2A. Deflation + Shifted Inverse Iteration for the Power Method ⭐ (pairs with top recommendation)
**Paper:** J.E. Gubernatis & T.E. Booth, *"Multiple Extremal Eigenpairs by the Power Method,"* arXiv:0807.1261 — https://arxiv.org/abs/0807.1261
**Extension:** implement power method + Hotelling deflation for top-k eigenpairs, then add Rayleigh-quotient shifted inverse iteration for faster convergence on clustered eigenvalues; benchmark against QR (as ground truth) sweeping condition number and eigenvalue-gap ratio.
**Test data:** SuiteSparse Matrix Collection (verified live, 2,904 real matrices, MATLAB/Matrix-Market/Rutherford-Boeing formats) — https://sparse.tamu.edu/ — plus synthetic gap-controlled matrices.
**Showcase results:** convergence-rate vs. eigenvalue-gap plots, condition-number sweep across dozens of matrices, accuracy-vs-QR error tables.
**Why easy & deterministic:** exact linear algebra, no randomness; power iteration core is ~10 lines.

### 2B. Robust Hybrid Bisection–Newton-Raphson Root Finder (Brent-style)
**Paper:** J. Kim, T. Noh, W. Oh, S. Park, N. Hahm, *"An improved hybrid algorithm to bisection method and Newton-Raphson method,"* Applied Mathematical Sciences 11(53-56), 2017, pp.2789–2797. DOI: 10.12988/ams.2017.710302 — https://www.m-hikari.com/ams/ams-2017/ams-53-56-2017/710302.html (verified: title/authors/journal/year confirmed)
**Extension:** the paper's hybrid switches bisection→Newton on a bracket but doesn't safeguard against flat-derivative/multiple-root cases. Build a Brent-style safeguarded hybrid (adds inverse-quadratic interpolation + Illinois/Anderson-Björck acceleration) and formally compare convergence order/iteration counts against bisection, false position, Newton, and the paper's original hybrid.
**Test data:** 30–50 synthetic test functions (transcendental, polynomial, pathological flat-derivative cases) — no external dataset needed.
**Showcase results:** methods × functions convergence-comparison matrix, iteration-count heatmaps, convergence-order log-log plots, failure-case study.
**Why easy & deterministic:** pure algebra, reproducible to machine precision, each method <50 lines; whole suite doable in 2–3 days.

### 2C. Adaptive-Proposal Metropolis-Hastings (Adaptive Metropolis)
**Paper:** H. Haario, E. Saksman, J. Tamminen, *"An adaptive Metropolis algorithm,"* Bernoulli 7(2), 2001, pp.223–242 — https://projecteuclid.org/journals/bernoulli/volume-7/issue-2/An-adaptive-Metropolis-algorithm/bj/1080222083.full (verified: title/authors/journal/year confirmed)
**Extension:** implement AM (proposal covariance adapted online from chain history) vs. fixed-covariance Metropolis-Hastings (syllabus week 9-10), then add acceptance-rate-coerced step-size adaptation (Robbins-Monro-style); measure autocorrelation time / effective sample size reduction.
**Test problems:** standard synthetic MCMC benchmark targets — ill-conditioned multivariate Gaussian, Rosenbrock ("banana") distribution, Gaussian mixture — all synthetic, no external data needed.
**Showcase results:** trace plots, autocorrelation function plots, ESS-per-second bar charts, 2D contour overlays of sampled vs. true density.
**Why easy & deterministic:** fixed seeds → exactly reproducible chains; AM update is ~20 lines on top of standard MH.

### 2D. Rook-Pivoting Gaussian Elimination for Ill-Conditioned Systems (backup option)
**Paper:** L.V. Foster, *"The growth factor and efficiency of Gaussian elimination with rook pivoting,"* Journal of Computational and Applied Mathematics 86, 1997, pp.177–194. DOI: 10.1016/S0377-0427(97)00154-4 — https://www.sciencedirect.com/science/article/pii/S0377042797001544
**Extension:** implement Gauss elimination/LU with partial, complete, and rook pivoting + iterative refinement; benchmark growth factor / residual / backward error vs. partial pivoting across a condition-number sweep on SuiteSparse (https://sparse.tamu.edu/) and synthetic ill-conditioned matrices (Hilbert, Vandermonde, Wilkinson).
**Why easy & deterministic:** exact linear algebra, LAPACK/NumPy gives instant ground truth for residual checks.

---

## Category 3 — Exploratory & Comparative Study

*"Conduct an in-depth exploratory project — benchmarking multiple numerical/simulation approaches against complex real-world datasets or theoretical modeling benchmarks."*

### 3A. Direct vs. Iterative Linear Solvers on Real Sparse Matrices ⭐ (largest, easiest scale-up)
**Paper:** P. Khrapov & N. Volkov, *"Comparative analysis of Jacobi and Gauss-Seidel iterative methods,"* arXiv:2307.09809, 2023 (also Int. J. Open Information Technologies 12(2), 2024) — https://arxiv.org/abs/2307.09809
**Dataset:** SuiteSparse Matrix Collection (verified live, 2,904 matrices from structural engineering, circuits, power grids) — https://sparse.tamu.edu/
**Methodology:** implement Gauss elimination, Gauss-Jordan, LU decomposition (direct) and Jacobi, Gauss-Seidel, Conjugate Gradient (iterative); pick ~30–50 matrices spanning dense/sparse and well-/ill-conditioned; solve Ax=b with known x; measure runtime, residual error, iteration count vs. condition number.
**Showcase results:** runtime-vs-matrix-size log-log plots, error heatmap (method × matrix), condition-number-vs-failure-rate chart, "which solver when" decision-tree summary.
**Split:** one method per person, or one matrix category (structural/circuit/power/graph) per person.
**Why easy & deterministic:** matrices ship with known solves; trivially scalable — just loop over more matrices from a set of 2,904.

### 3B. Root-Finding Method Shootout (Bisection / Regula Falsi / Newton-Raphson / Secant / Bairstow)
**Paper:** J.C. Ehiwario & S.O. Aghamie, *"Comparative Study of Bisection, Newton-Raphson and Secant Methods of Root-Finding Problems,"* IOSR Journal of Engineering 4(4), 2014, pp.01-07. DOI: 10.9790/3021-04410107 — https://www.iosrjen.org/Papers/vol4_issue4%20(part-1)/A04410107.pdf
**Test suite:** TEST_NONLIN — 23 classic nonlinear-equation test problems (John Burkardt, verified live) — https://people.math.sc.edu/Burkardt/f_src/test_nonlin/test_nonlin.html — expand with polynomial test cases for Bairstow's method.
**Methodology:** implement all 5 methods once; run on 50–100+ test functions; log iteration count, wall-clock time, final error, divergence cases.
**Showcase results:** methods × functions convergence heatmap, convergence-order theory-vs-practice plots.
**Split:** 1 method per person + 1 person on test-function curation/benchmark harness.
**Why easy & deterministic:** true roots known analytically; correctness is binary and instant to check.

### 3C. Interpolation Shootout on Real Time Series — Quantifying Runge's Phenomenon
**Paper:** M. Bellucci, L. Miralles, M.A. Qureshi, B. Mac Namee, *"ZeLiC and ZeChipC: Time Series Interpolation Methods for Lebesgue or Event-based Sampling,"* arXiv:1906.03110, 2019 — https://arxiv.org/abs/1906.03110
**Datasets:** NOAA U.S. Climate Normals (verified live, ~15,000 stations, bulk CSV) — https://www.ncei.noaa.gov/products/land-based-station/us-climate-normals ; JHU CSSE COVID-19 daily case series — https://github.com/CSSEGISandData/COVID-19
**Methodology:** subsample real series at increasing gaps, reconstruct missing points via Lagrange interpolation, Newton's divided-difference polynomial, and cubic spline; compare RMSE against real held-out points as polynomial degree/gap grows — directly exposes Runge's-phenomenon oscillation blow-up vs. spline stability.
**Showcase results:** interpolated-vs-actual overlay plots, RMSE-vs-degree curves showing the Runge blow-up, per-station/per-country error heatmaps.
**Split:** 2 people on NOAA stations, 2 on COVID country curves, 1 on the classic Runge test function + unifying dashboard.
**Why easy & deterministic:** held-out real points are ground truth; datasets are pre-cleaned CSVs, near-zero preprocessing.

### 3D. Monte Carlo Method Comparison for Integral / Option-Price Estimation
**Paper:** A.B. Owen & S.D. Tribble, *"A Quasi-Monte Carlo Metropolis Algorithm,"* PNAS 102(25), 2005, pp.8844–8849. DOI: 10.1073/pnas.0409596102 — https://www.pnas.org/doi/10.1073/pnas.0409596102 (open mirror: https://artowen.su.domains/reports/qmcmcmc.pdf)
**Dataset:** FRED (St. Louis Fed, verified live, 845,000+ free economic/financial series, public API) — https://fred.stlouisfed.org/
**Methodology:** implement naive Monte Carlo, Metropolis-Hastings MCMC, and quasi-Monte Carlo (Sobol) estimators for benchmark integrals with known closed forms and for Black-Scholes option prices using real volatility/rate inputs from FRED; sweep sample size N; plot error vs. N against theoretical O(N⁻¹ᐟ²) / O(N⁻¹) rates.
**Showcase results:** error-vs-N convergence curves (overlaid theoretical rates), variance-reduction bar charts, time-vs-accuracy tradeoff dashboard.
**Why easy & deterministic:** ground truth is analytically known; correctness check is trivial; scales to hundreds of repeated runs for statistical significance.

---

## Suggested Decision

Given the "fast, deterministic, heavily scalable, presentable" priority, rank the options as:

1. **Power Method / PageRank on SNAP graphs (1A + 2A + 3A-eigen variant)** — best link quality, best visuals, cleanest 5-way split, directly hits eigenvalue syllabus content (week 5–6), scales for free by adding more SNAP graphs.
2. **Direct vs. Iterative Linear Solvers on SuiteSparse (3A)** — largest available dataset (2,904 matrices), zero preprocessing, natural 5-way split by method.
3. **Root-Finding Shootout or Kepler's Equation (3B / 1B)** — simplest to implement (all synthetic or small CSV data), fastest to finish if time is very tight, still highly presentable (orbit animations, convergence heatmaps).

All three avoid ML/DL entirely, produce deterministic and instantly-verifiable numerical results, and can be inflated in scope (more graphs / more matrices / more test functions / more countries) to demonstrate heavy computational work for full marks + bonus.
