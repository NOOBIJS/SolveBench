# Deep Dive: Power-Grid Eigenvector Centrality vs. SuiteSparse Solver Shootout

Head-to-head comparison of two shortlisted candidates:

- **Project A** — "1D. Eigenvector Centrality for Power-Grid Security" ([papers2.md](papers2.md))
- **Project B** — "3A. Direct vs. Iterative Linear Solvers on SuiteSparse" ([papers.md](papers.md))

---

## Project A — Eigenvector Centrality for Power-Grid Security

### Methodology, step by step
1. **Source grid topology.** Pull IEEE standard test cases (14/30/57/118/300-bus) and larger synthetic cases from Power Grid Lib (`pglib-opf`, up to ~30,000 buses in the largest synthetic families).
2. **Parse into a graph.** These ship as MATPOWER `.m` case files (bus table + branch table with impedance data), not plain edge lists. You need either a hand-written parser or a helper library (`pandapower`, `pypower`, `matpowercaseframes`) to turn each case into an adjacency/admittance matrix — this is new territory, not something the syllabus labs prepare you for.
3. **Choose a graph model.** Decide whether centrality is computed on unweighted topology or admittance-weighted edges — a real modeling judgment call with no single correct answer, and reviewers can reasonably disagree with your choice.
4. **Power iteration (syllabus wk 5-6).** Compute the dominant eigenvector → eigenvector centrality, rank buses.
5. **QR cross-check.** Implement/apply the QR algorithm as a validation baseline.
6. **Contingency simulation.** Remove top-k "critical" buses/lines one at a time, rebuild the graph, recompute centrality — an N-1/N-k contingency-style analysis borrowed from power-systems engineering.
7. **Visualize.** Network diagrams colored by centrality, before/after failure.

### Dataset size & rigor
- IEEE standard cases: only 5 commonly used sizes (14, 30, 57, 118, 300 buses). `pglib-opf` adds maybe 20-30 more synthetic large-scale variants (`sad`, `api`, `scenario` families), topping out around 30,000 buses for the very largest.
- Realistic usable pool for a course project: **roughly 15-30 distinct network topologies** — a real ceiling on how far you can "loop over more datasets" for scale.
- There is no large, ready-made library of *real* power-grid topologies analogous to SNAP or SuiteSparse — genuine grid topology data is often withheld for security reasons in many countries, which is precisely why the field relies on a small set of standard synthetic test cases.
- Base paper is **paywalled** (ScienceDirect, could not be fetched directly during research; bibliographic details only confirmed via search-index metadata). The team would be extending a paper it may not be able to fully read.

### Coding footprint (rough estimate)
| Component | Est. lines | Notes |
|---|---|---|
| MATPOWER case parser / loader | 150–300 | New territory, fragile, extra dependency |
| Power iteration core | 20–30 | Syllabus-standard |
| QR algorithm | 80–150 | Full implementation for validation |
| Contingency simulation | 50–100 | Graph rebuild + rerun logic |
| Visualization (networkx/matplotlib) | 150–250 | Layouts get messy above ~500 nodes |
| Harness / driver / reporting | 150–200 | |
| **Total** | **~800–1,500 lines, ~8–12 files** | |

### Easiness
Moderate. The core numerical methods (power method, QR) are short and syllabus-standard, but the **data-ingestion layer is genuinely new work** outside anything the labs prepare the team for, and there's real domain-modeling ambiguity (weighted vs. unweighted graph, which contingency scenarios matter) that a grader could probe.

### Safety of the result / showcase-ability
Factually safe — IEEE/`pglib-opf` cases are standard published academic benchmarks created specifically for this kind of public analysis, not real sensitive grid data, so there's no actual disclosure risk. The risk is **framing, not safety**: presenting "these are the power grid's weak points" from a pure topology/eigenvector-centrality model (ignoring real AC power-flow physics, protection relays, redundancy) can look like overclaiming to an evaluator who knows power systems. This is manageable with careful caveats ("structural criticality under a simplified graph model," not "operational vulnerability"), but it's an extra thing to get right in the writeup and defend in the viva.

### Possibility of acceptance
Good — fits "Cross-Domain Application" cleanly and the paper is genuinely recent (2024). Two soft risks: (1) an unreadable paywalled base paper is awkward to defend if the evaluator asks for details of the original methodology, (2) the "attack/vulnerability" framing invites scrutiny from anyone with power-systems background about how much was actually modeled vs. asserted.

### Real-life importance
High "wow factor" — power-grid resilience against cascading failures/attacks is a widely-understood, high-stakes real-world concern, easy for any audience (including non-specialist evaluators) to immediately grasp why it matters.

---

## Project B — Direct vs. Iterative Linear Solvers on SuiteSparse

### Methodology, step by step
1. **Select matrices.** Pull 30-100+ matrices from the SuiteSparse Matrix Collection (2,904 available), deliberately spanning size, sparsity, symmetry, and condition number, across real application domains (structural mechanics, circuits, thermal, power networks, optimization, CFD).
2. **Load.** Matrix Market `.mtx` format is standard and directly readable with `scipy.io.mmread` — essentially zero custom parsing work.
3. **Implement solvers.** Direct: Gauss elimination, Gauss-Jordan, LU decomposition (**this is literally Lab Assignment 2** — largely reused, not new code). Iterative: Jacobi, Gauss-Seidel, Conjugate Gradient (all textbook-standard, extensively documented, low bug risk).
4. **Build ground truth.** For each matrix A, generate a known `x_true`, compute `b = A·x_true`, solve with every method.
5. **Measure.** Runtime, iteration count (iterative methods), relative residual `‖Ax−b‖/‖b‖`, error vs. `x_true`, and correlate against condition number/sparsity pattern (optionally estimate condition number via the power method — a nice second use of syllabus content).
6. **Sweep at scale.** Loop the whole pipeline across dozens–hundreds of matrices automatically — no per-matrix special-casing needed since the format and problem (`Ax=b`) are uniform.
7. **Report.** Heatmaps (method × matrix), runtime-vs-size plots, condition-number-vs-failure-rate charts, a "which solver when" decision summary.

### Dataset size & rigor
- **2,904 real matrices**, sourced from decades of real engineering/science problems, curated specifically as a numerical-linear-algebra benchmark and used in thousands of published papers since the 1990s — this is about as gold-standard/rigorous as a benchmark dataset gets in this field.
- Format is uniform and simple (Matrix Market), so scaling from 30 to 300 matrices is a change of one loop bound, not new engineering effort.
- Real, freely citable, recent (2023/2024) supporting comparative-methodology papers exist (Khrapov & Volkov 2023 on Jacobi/Gauss-Seidel; Ferrari 2024 on 16-solver benchmarking of ill-conditioned real matrices in `papers2.md` §3B) — both were directly fetched and verified, no paywall issue.

### Coding footprint (rough estimate)
| Component | Est. lines | Notes |
|---|---|---|
| Matrix loader | ~20 | `scipy.io.mmread` wrapper |
| Direct solvers (Gauss, Gauss-Jordan, LU) | ~150 | Mostly reused from Lab Assignment 2 |
| Iterative solvers (Jacobi, Gauss-Seidel, CG) | ~110 | Standard textbook pseudocode |
| Condition-number estimation (power method) | ~30 | Optional, reuses another syllabus method |
| Benchmark harness (loop, timing, error capture) | ~150–250 | |
| Visualization/reporting | ~150–250 | |
| **Total** | **~700–1,100 lines, ~8–10 files** | |

### Easiness
Highest of the two. No exotic file formats, no domain-modeling ambiguity (the problem `Ax=b` is fully specified by the matrix itself — no judgment calls to defend), and a large share of the "hard" algorithm work is **already required coursework** (Lab Assignment 2), so the project mostly adds a benchmarking harness and visualization layer on top of code the team writes anyway.

### Safety of the result / showcase-ability
Essentially zero risk. A method converging is a good result; a method diverging or losing accuracy on an ill-conditioned matrix is *also* a good, expected, celebrated result — that's precisely the point of a comparative numerical-methods study, not a failure to explain away. There is no framing risk analogous to Project A's "vulnerability" language — you're reporting numerical facts about solver behavior, full stop.

### Possibility of acceptance
Very high. It matches the spec's literal wording for "Exploratory & Comparative Study" almost verbatim ("benchmarking multiple numerical/simulation approaches against complex real-world datasets"), draws on syllabus content across two full weeks (Lab 2: Gauss/Gauss-Jordan/LU, plus iterative methods as a natural extension), and rests on an uncontroversial, universally-recognized benchmark. Nothing about the setup invites the kind of "did you actually model this correctly" pushback Project A risks.

### Real-life importance
Less immediately dramatic to a lay audience than "power grid security," but arguably *more* fundamentally important: direct and iterative linear solvers are the invisible backbone of essentially all engineering simulation — structural analysis, circuit design, computational fluid dynamics, chemical process modeling — anything that reduces to `Ax=b`. For a CSE faculty audience specifically, this importance is well understood and easy to sell without oversimplifying.

---

## Side-by-Side Summary

| Dimension | A — Power-Grid Eigenvector Centrality | B — SuiteSparse Solver Shootout |
|---|---|---|
| Methodology complexity | Moderate–high (custom parsing, modeling choices) | Low (uniform format, no modeling ambiguity) |
| Dataset scale | ~15–30 usable networks | 2,904 matrices, trivially scalable |
| Dataset rigor | Standard but niche (power-systems community) | Gold-standard, 30-year benchmark in numerical analysis |
| Base paper access | Paywalled, unverified full text | Both supporting papers fully verified, open |
| Code footprint | ~800–1,500 lines, more novel code | ~700–1,100 lines, majority reuses Lab 2 |
| Overlap with existing labs | Partial (power method only) | Strong (Gauss/Gauss-Jordan/LU directly reused) |
| Risk of "ugly"/hard-to-present result | Low-moderate (framing/overclaim risk) | Essentially none (all outcomes are valid findings) |
| Acceptance risk | Good, with two soft risk points | Very low risk, matches spec almost verbatim |
| Real-life "wow factor" | High (relatable, dramatic) | High but more technical/behind-the-scenes |

---

## Recommendation

**Choose Project B — Direct vs. Iterative Linear Solvers on SuiteSparse.**

Given what you've said matters most this round — finish fast, stay deterministic, minimize struggle, maximize visible scale for full marks + bonus — B wins on every axis that carries real project risk: it reuses Lab Assignment 2 code directly instead of requiring a new data-parsing subsystem, its dataset is larger, more rigorous, and effortless to scale (2,904 matrices vs. a ceiling of ~20-30 grid topologies), both its supporting papers are fully verified and open (no paywall gap in your knowledge to defend), and it carries no framing risk — every possible outcome (convergence or graceful failure) is a valid, presentable finding. It also maps almost word-for-word onto the spec's own description of the "Exploratory & Comparative Study" category, which lowers acceptance risk the most.

Project A is a genuinely good idea and more dramatic to pitch verbally ("we ranked the power grid's weak points"), but it costs you a new parsing subsystem, a smaller dataset ceiling, an unreadable base paper, and a framing tightrope to walk — extra risk for a payoff that's more about narrative appeal than about the actual grading criteria (methodology rigor, scale, correctness).

If you want a middle path: build Project B as the main deliverable, and if time remains, fold in a *small* Project-A-style add-on — run your already-implemented power method on one or two IEEE test-case graphs (14-bus, 118-bus) as a "bonus real-world application" section. That gets you the dramatic power-grid visual without taking on its data-pipeline risk as the backbone of the whole project.
