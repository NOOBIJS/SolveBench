# SolveBench — Everything You Need to Understand This Project

A plain-language walkthrough of the base paper, the methods, and what your group is actually building. Read top to bottom once; after that, use it as a reference.

---

## 1. The one-sentence version

Every time an engineer simulates a power grid or checks whether a bridge design is safe, the computer has to solve a giant system of linear equations, **Ax = b**. There are several standard ways to solve it — some *exact* (direct methods), some *approximate-but-iterative* (iterative methods). A 2023 paper worked out, in painstaking mathematical detail, exactly when two of the iterative methods (Jacobi and Gauss-Seidel) are guaranteed to work — but only tested it on tiny, fake (randomly generated) systems of 2–5 unknowns. **Your group is taking that same comparison and running it on 50 real engineering matrices (real power grids, real building/bridge structures) with up to ~5,000 unknowns**, adding in the direct methods too, to see which solver actually wins in practice.

That's it. Everything else below is filling in detail.

---

## 2. Background: what problem are we even solving?

A system of linear algebraic equations (**SLAE**) looks like this:

```
10x + 1y + 1z = 12
1x + 10y + 1z = 12
1x + 1y + 10z = 12
```

In matrix form, that's **Ax = b**, where A is the matrix of coefficients, x is the vector of unknowns, and b is the right-hand side. Solving it means finding the x that makes the equations true. Here the answer is x = y = z = 1 (check: 10+1+1=12 ✓).

This isn't a toy problem. Structural engineering, circuit design, power-grid load-flow, fluid simulation — nearly all of them reduce, at some inner step, to solving Ax = b, often with **thousands to millions of unknowns**. So the question "what's the *best* way to solve Ax = b, and when?" is a real, high-stakes engineering question, not just a math exercise.

There are two families of solution methods:

- **Direct methods** — do a fixed, predictable sequence of arithmetic steps and land on the *exact* answer (up to rounding error). Examples: Gauss elimination, Gauss-Jordan, LU decomposition.
- **Iterative methods** — start from a guess and repeatedly refine it, getting closer and closer to the answer. They never give an exact answer, only "close enough," but each step is cheap, which matters a lot when the matrix is huge. Examples: Jacobi, Gauss-Seidel, Conjugate Gradient.

**The catch with iterative methods: they don't always work.** Sometimes the sequence of guesses gets closer to the true answer (this is called *converging*); sometimes it drifts further away forever (*diverging*). Whether a given method converges depends on the structure of matrix A. Figuring out exactly *when* a method converges is a serious mathematical question — and that's what the base paper is about.

---

## 3. The base paper, explained without the heavy math

**Khrapov, P. & Volkov, N. (2023). "Comparative Analysis of Jacobi and Gauss-Seidel Iterative Methods." arXiv:2307.09809**

### 3.1 What Jacobi and Gauss-Seidel actually do

Both methods solve Ax = b by iteration. At each step, you use equation *i* to solve for unknown *i* in terms of all the *other* unknowns' current values, then update.

**Jacobi method** — when updating x1, x2, x3 for the *next* round, use only the values from the *previous* round (update everything "at once," like a snapshot).

**Gauss-Seidel method** — same idea, but as soon as you compute a new value (say, the new x1), *immediately* use it when computing x2 in the same round, instead of waiting for the next round. It's the same method except it always uses the freshest information available.

### 3.2 A worked example (do this by hand once — it makes everything click)

Take the system from Section 2:
```
10x + y + z = 12
x + 10y + z = 12
x + y + 10z = 12
```
True answer: x = y = z = 1. Start the guess at (0, 0, 0).

**Update formula for each variable** (solve row *i* for unknown *i*):

```
Row 1:  10x + y + z = 12   ->  x = (12 - y - z) / 10
Row 2:  x + 10y + z = 12   ->  y = (12 - x - z) / 10
Row 3:  x + y + 10z = 12   ->  z = (12 - x - y) / 10
```

Now apply them:

**Jacobi, iteration 1** (all three use the *old* guess (0,0,0)):
```
x = (12 - 0 - 0) / 10 = 1.2
y = (12 - 0 - 0) / 10 = 1.2
z = (12 - 0 - 0) / 10 = 1.2
```

**Gauss-Seidel, iteration 1** (each uses the *newest* values so far):
```
x = (12 - 0   - 0)   / 10 = 1.2     <- same as Jacobi, nothing to reuse yet
y = (12 - 1.2 - 0)   / 10 = 1.08    <- reused the fresh x = 1.2
z = (12 - 1.2 - 1.08)/ 10 = 0.972   <- reused the fresh x and y
```

Compare how close each got to the true answer (1, 1, 1) after just one round:

| Method | x | y | z | Worst error |
|---|---|---|---|---|
| Jacobi | 1.2 | 1.2 | 1.2 | 0.2 |
| Gauss-Seidel | 1.2 | 1.08 | 0.972 | 0.2 (but 2 of 3 already much closer) |

Gauss-Seidel is "cheating" by reusing information the instant it's available, so it typically needs fewer rounds to get close. That head start is the entire intuition behind the paper's central finding.

### 3.3 What "convergence" means here, geometrically

Each method converges if and only if a certain polynomial built from the matrix's entries has all its **roots inside the unit circle** (i.e., every root's magnitude is < 1) — this is standard for iterative methods and comes from classical linear algebra (not something the paper invented). The paper's real contribution is figuring out **exactly which matrices** satisfy this, by mapping out the boundary region in a 2D plane (they call the coordinates *p* and *q*, or *d*, *a*, *b*) where convergence holds. You do **not** need to reproduce this derivation — it's advanced complex-analysis machinery (a variant of the "Hurwitz stability criterion" used in control theory). Just know that it exists and that it's how they got their answer rigorously instead of just guessing from examples.

### 3.4 What they actually found

1. **For systems with exactly 2 unknowns:** Jacobi and Gauss-Seidel have the *exact same* convergence range. If one works, so does the other, always.
2. **For 3+ unknowns:** the two methods' convergence ranges are different shapes. Gauss-Seidel's "safe zone" is usually bigger — it converges more often. But it's not a strict superset in every case: they found actual example matrices where **Jacobi converges but Gauss-Seidel does not**. So "Gauss-Seidel is just better" is not 100% true — it's *usually* true.
3. **Statistical confirmation:** they generated 100,000 random matrices (2 to 5 unknowns) and checked convergence numerically. Results (their Table 1):

| Unknowns | Both converge | Only Gauss-Seidel converges | Only Jacobi converges |
|---|---|---|---|
| 2 | 49,916 | 0 | 0 |
| 3 | 11,818 | 7,521 | 1,095 |
| 4 | 1,436 | 3,411 | 528 |
| 5 | 111 | 726 | 76 |

Read this as: at 2 unknowns, they're perfectly correlated (matches point 1 above). From 3 unknowns onward, Gauss-Seidel wins far more often than it loses (e.g., at n=3: 7,521 wins vs. 1,095 losses for Gauss-Seidel), and **both methods converge less and less often as the system gets bigger** — convergence gets harder at scale.

### 3.5 The gap — this is the whole reason your project exists

The paper stops at **5 unknowns**, and every matrix is **randomly generated** — not a real engineering system. Real power grids and real structures have **hundreds to thousands** of unknowns, and their matrices have specific structure (sparse, often symmetric, diagonally-dominant-ish) that random matrices don't necessarily have. Nobody in the paper checks whether the "Gauss-Seidel usually wins" conclusion still holds at real-world scale, on real-world matrices. That's the open question your group is answering.

---

## 4. Your project: SolveBench

### 4.1 The pitch

> Take every solver method taught in this course (not just Jacobi/Gauss-Seidel — also the direct methods and Conjugate Gradient), and benchmark them all against 50 real matrices from actual power-grid and structural-engineering problems. Find out, empirically, which solver you should actually trust for which kind of real system.

This is filed under the spec's **"Exploratory & Comparative Study"** category — you're not proving a new theorem, you're running a rigorous, large-scale experiment and reporting what you find.

### 4.2 The methods you're implementing

**Direct methods** (give an exact answer via a fixed number of steps):

| Method | Idea | Note |
|---|---|---|
| **Gauss elimination** | Systematically zero out everything below the diagonal (using row operations), then solve from the bottom up ("back-substitution"). | The classic method, this is Lab Assignment 2 territory. |
| **Gauss-Jordan** | Like Gauss elimination, but keep going until the matrix is fully diagonal (identity) — so the answer just falls out, no back-substitution needed. | Slightly more arithmetic than Gauss elimination, same idea. |
| **LU decomposition** | Factor A = L·U (L = lower-triangular, U = upper-triangular) once, then solving Ax=b becomes two easy triangular solves. | Efficient when you need to solve with the *same* A but *different* b's repeatedly. |

**Iterative methods** (approximate, refine step by step, cheap per step):

| Method | Idea | Note |
|---|---|---|
| **Jacobi** | Section 3.1 above. | From the base paper. |
| **Gauss-Seidel** | Section 3.1 above. | From the base paper. |
| **Conjugate Gradient (CG)** | A smarter iterative method for symmetric, positive-definite matrices. Instead of blindly refining coordinate-by-coordinate like Jacobi/Gauss-Seidel, it picks a sequence of mathematically "independent" (conjugate) directions to move in, so it doesn't waste effort re-correcting the same error twice. In theory it reaches the exact answer in at most n steps. | Not in the base paper — this is the "extra" method you're adding. Works especially well on structural stiffness matrices, which are typically symmetric positive-definite. |

### 4.3 Optional solvers worth adding

The 6 methods above aren't a fixed requirement — nothing in the spec or the base paper caps the count. A couple of natural, low-effort additions if your group wants a bit more depth:

| Method | Type | Why it fits |
|---|---|---|
| **SOR (Successive Over-Relaxation)** | Iterative | Literally Gauss-Seidel with one extra tuning knob (a relaxation factor ω that over/under-corrects each step to speed up convergence) — maybe 5 lines on top of code you're already writing. It's also directly relevant to the base paper: one of its own references (Sun, 2005) is specifically a convergence study of SOR, so adding it extends the base paper's own lineage instead of going off-topic. |
| **Cholesky decomposition** | Direct | The direct-method counterpart to CG. Most of your structural stiffness matrices are symmetric positive-definite — exactly the case CG is suited for — and Cholesky is the direct method built specifically for that case (~2x faster than LU, no pivoting needed for stability). Adding it lets you show a clean "for SPD systems, the SPD-specialized methods (Cholesky, CG) beat the general-purpose ones (LU, Jacobi)" result. |

Two other candidates exist but are lower priority: **Preconditioned CG** (small extension of your existing CG lane, shows how much a simple preconditioner speeds convergence) and **GMRES** (handles non-symmetric matrices — only useful if you decide to keep the 6 non-power-grid/non-structural matrices discussed earlier instead of excluding them). Adding SOR and Cholesky is enough to meaningfully deepen the comparison without stretching a 5-minute pitch or a 5-person split too thin — going much past 8 solvers starts working against you.

### 4.4 The data

**50 real matrices**, already downloaded into `matrices/matrices/` (Matrix Market `.mtx` format, from the SuiteSparse Matrix Collection — a standard, widely-cited academic benchmark library, not something you built yourselves):

- **Power-grid admittance matrices** (11 systems, 118–1,723 buses) — filenames starting `bcspwr` (e.g. `bcspwr03`, `bcspwr09`). These represent real electrical grid topologies; solving Ax=b on them is what a power-flow simulation does under the hood.
- **Structural stiffness matrices** (24 systems, 112–4,884 degrees of freedom) — filenames starting `bcsstk`/`bcsstm` (e.g. `bcsstk08`, `bcsstm11`). These represent real building/bridge/frame structural models; solving Ax=b on them is what a structural safety check does under the hood.

Each `.mtx` file is directly readable in Python with `scipy.io.mmread(...)` — no custom parsing needed.

### 4.5 How each experiment works (the ground-truth trick)

For each of the 50 matrices A:
1. Make up a **known answer**, x_true (e.g., a vector of all 1's, or random values).
2. Compute **b = A · x_true**. Now you know the exact right-hand side.
3. Solve **Ax = b** using each of the 6 methods above, *pretending you don't know x_true*.
4. Compare what each method *found* against x_true. Measure:
   - **Runtime** (wall-clock time)
   - **Iteration count** (for iterative methods only)
   - **Residual**: ‖A·x_found − b‖ (how far off the equation is)
   - **Error**: ‖x_found − x_true‖ (how far off the answer is)
   - **Condition number** of A (roughly: how sensitive/unstable the system is — higher condition number = harder problem, more error amplification)

This is fully deterministic and unambiguous — there's no fuzzy "did it work" question, just numbers you compute and compare.

### 4.6 The pipeline (matches the diagram in linear.tex)

```
Real sparse matrix (.mtx)
        │
        ├──> Direct solvers (Gauss Elim. / Gauss-Jordan / LU)
        │
        └──> Iterative solvers (Jacobi / Gauss-Seidel / CG)
                        │
                        v
         Benchmark harness (runtime, iterations, residual, condition no.)
                        │
                        v
         Results dashboard (heatmaps, plots, solver decision guide)
```

### 4.7 What the final output should look like

- A **heatmap**: method × matrix, colored by error or by success/failure — instantly shows which method handles which kind of matrix.
- **Runtime-vs-matrix-size** plots (log-log), for direct vs. iterative.
- **Condition-number-vs-convergence/failure** plots — does Gauss-Seidel's "usually wins" conclusion from the base paper hold at scale? Does it correlate with condition number the same way?
- A plain-English **"which solver, for which system" decision guide** — the actual payoff of the whole project, and a natural closing slide/section.

### 4.8 Suggested 5-way split (one person per lane)

1. Direct solvers (Gauss elimination + Gauss-Jordan + LU)
2. Iterative solvers — Jacobi + Gauss-Seidel (ties directly back to the base paper)
3. Iterative solvers — Conjugate Gradient + condition-number estimation
4. Data pipeline (loading all 50 `.mtx` matrices, generating ground truth, running the sweep)
5. Benchmark harness + visualization/dashboard + report writing

---

## 5. Going deeper: how the paper actually proves its claims, and how this project is framed

Section 3 already covered *what* the paper found. This section goes one level deeper — *how* it proves that, precisely what its limitations are, and how your project's framing (the category, the real-world stakes, the two datasets) holds together as a coherent argument rather than a list of separate facts.

### 5.1 How the paper actually proves convergence: the Hurwitz trick

A method converges exactly when a certain polynomial, built from the matrix's entries, has every root sitting inside the unit circle (magnitude less than 1). The brute-force way to check this would be to solve for every root of that polynomial and measure each one directly — but solving a general cubic or higher-degree polynomial symbolically, for arbitrary coefficients, gets ugly fast, and you'd have to redo the whole derivation for every new matrix.

So the paper borrows a much older trick from control engineering: the **Hurwitz stability criterion**. It was originally built to answer a related but different question — "will this control system settle down, or oscillate out of control?" — without ever solving for the system's characteristic roots. The method: arrange the polynomial's coefficients into a specific matrix (the Hurwitz matrix), then check the sign of a handful of that matrix's determinants. If they're all positive, every root is guaranteed to sit in the "safe" region — no root-finding required. The paper adapts this (their "complex analog") from checking "is the root in the left half of the complex plane" (the original control-theory version) to checking "is the root inside the unit circle" (what convergence actually needs), via a coordinate transformation.

The payoff of doing it this way: instead of testing convergence one matrix at a time, they get a **closed-form geometric boundary** — an actual curve in a 2D parameter plane separating "always converges" from "never converges." That's strictly stronger than a rule of thumb like "the matrix is diagonally dominant" (which only tells you convergence is *guaranteed in some cases*, never gives you the *exact* boundary).

### 5.2 Why the derivation stops at 3 unknowns, and what the gap means

For a 2-unknown system, the polynomial from Section 5.1 is only degree 2 — a quadratic, solvable by hand with the quadratic formula, giving one clean exact condition. For 3 unknowns it's a cubic — still solvable by hand using the same Hurwitz machinery, but the algebra sprawls across several pages of the paper (visible if you skim Sections 3–4 of `Paper.pdf` — lots of complex-plane geometry, φ angles, r radii). Past 3 unknowns, hand-deriving a new formula for every case stops being realistic; the polynomials get too large and the casework explodes.

Instead of a fourth or fifth closed-form formula, the paper gives a general *algorithm* — build the Hurwitz matrix, check the determinant signs; this is literally the Python function shown on page 18 of the paper — that a computer can run on any matrix of any size. To confirm this general check tells the same story as the hand-derived results, the authors ran it on **100,000 randomly generated matrices**, sizes 2 through 5, and tallied how often each method converged. That's exactly where Table 1 (Section 3.4 above) comes from.

One more nuance worth knowing: for exactly 2 unknowns, Jacobi's and Gauss-Seidel's convergence conditions reduce to the *literal same inequality* — the two methods are functionally identical at that size. For 3 unknowns, plotted on the same plane, the regions are genuinely different shapes; in the paper's main example Gauss-Seidel's region fully contains Jacobi's (wins everywhere Jacobi wins, plus more). But the paper also gives a specific counter-example 3×3 matrix where it flips — **Jacobi converges, Gauss-Seidel does not.** So "Gauss-Seidel is strictly better" is a trend, not a law; which method actually wins depends on the specific matrix, not just its size.

Put together, everything the paper proves is either exact-but-tiny (hand-derived, capped at n=3) or general-but-synthetic (the algorithm works at any size, but was only ever run on random matrices, and only up to n=5 — this is stated explicitly in Section 6 of the paper, verified directly against the text, not paraphrased secondhand). Nowhere does the paper touch a matrix from a real power grid or a real building. That absence is precisely what your project has to contribute: checking whether "Gauss-Seidel usually wins" still holds once the matrix stops being a toy.

### 5.3 Why the real-world stakes are more than a hook

Power-flow simulation and structural safety analysis both reduce, at their core, to solving Ax=b, repeatedly, as one step inside a larger simulation loop. If the solver picked for that inner step doesn't converge for a *particular* real grid or structure, the consequence isn't abstract: the outer simulation either runs indefinitely without producing an answer, or the iteration blows up numerically and the tool reports failure outright. In a real engineering pipeline, that means a blackout-risk study or a structural safety check simply never finishes. That's the concrete version of "pick the wrong solver and you're in real trouble."

### 5.4 Why this counts as "Exploratory & Comparative Study," specifically

`Spec.txt` offers three project lanes, and it's worth being able to say precisely why this project is the third one and not the other two. A *Cross-Domain Application* would mean taking a numerical method from the course and applying it somewhere unrelated to its usual home — e.g. using root-finding on orbital mechanics, or Monte Carlo on epidemiology. That's not this project, since it stays inside the exact domain (linear-solver convergence) the base paper already occupies. A *Methodological Extension* would mean modifying the algorithm itself — inventing a new variant, relaxing an assumption, improving the underlying math. Also not this project: Jacobi, Gauss-Seidel, and the other solvers are implemented in their standard, unmodified textbook form. What *is* happening: take existing, untouched methods, and rigorously benchmark them against a large set of real datasets the original paper never touched, at a scale the paper couldn't reach by hand. That is close to word-for-word the spec's own description of "Exploratory & Comparative Study" — "benchmarking multiple numerical approaches against complex real-world datasets."

### 5.5 What the two real datasets physically represent

A power-grid **admittance matrix** encodes, for every pair of buses (substations/nodes) in an electrical network, how current relates to voltage across the line connecting them — the entries come from the physical conductance/susceptance of each transmission line. Solving Ax=b against it is literally what a power-flow study computes to find voltages across the whole grid.

A structural **stiffness matrix** encodes, for every degree of freedom in a finite-element model of a building or bridge (each node's possible directions of movement), how much force is needed to produce a given displacement — the entries come from material stiffness and structural geometry. Solving Ax=b against it is what a structural analysis does to find how the structure deforms under load.

The size ranges — 118–1,723 buses for the power-grid matrices, 112–4,884 degrees of freedom for the structural ones — matter because they sit two to three orders of magnitude past the paper's own 5-unknown ceiling. That jump in scale is the entire empirical contribution of the project, in numeric form.

### 5.6 Why 50 matrices, framed as a sample and not a ceiling

50-of-2,904 is meant to read as a deliberately chosen sample from a large, standard, citable benchmark library — not something the group had to build or clean themselves, and not a hard limit. There's obvious room to add more matrices later if a stronger result or more evaluation time is wanted.

---

## 6. Glossary (quick lookup)

| Term | Meaning |
|---|---|
| **SLAE** | System of Linear Algebraic Equations — the Ax = b problem. |
| **Direct method** | Solves exactly in a fixed number of steps (e.g. Gauss elimination). |
| **Iterative method** | Refines a guess repeatedly; may or may not converge (e.g. Jacobi). |
| **Convergence** | The iterative guesses get closer and closer to the true answer as you keep going. |
| **Divergence** | The opposite — guesses get worse/blow up. |
| **Diagonally dominant matrix** | A matrix where, in each row, the diagonal entry's magnitude is bigger than the sum of the other entries' magnitudes. This is a common *sufficient* (not necessary) condition that guarantees both Jacobi and Gauss-Seidel converge. |
| **Residual** | ‖Ax − b‖ for your computed x — how far off the equation is; the standard way to check a solver's answer without knowing the true x. |
| **Condition number** | A number describing how sensitive Ax=b's solution is to small changes/errors in A or b. High condition number = "ill-conditioned" = numerically difficult. |
| **Sparse matrix** | A matrix that's mostly zeros — real engineering matrices almost always are, which is why special sparse-aware storage/algorithms matter at scale. |
| **Hurwitz stability criterion** | A classical tool (from control theory) for checking whether all roots of a polynomial lie in a certain safe region, without solving for the roots directly. The base paper's advanced-math engine — you don't need to reproduce it. |
| **Positive-definite matrix** | A symmetric matrix where all eigenvalues are positive. Conjugate Gradient works best on these; structural stiffness matrices are usually this type. |
| **SuiteSparse Matrix Collection** | A large, standard, freely available library of real-world sparse matrices used across numerical-analysis research — the source of your 50 matrices. |
| **Matrix Market (.mtx)** | A simple standard text file format for storing matrices, readable directly by `scipy.io.mmread`. |

---

## 7. Where things stand right now

- ✅ Base paper picked and read (this file covers it)
- ✅ 50 real matrices downloaded (`matrices/matrices/`)
- ✅ Presentation deck for the proposal pitch written (`linear.tex` → `linear.pdf`)
- ⬜ Actual solver implementations (Gauss elim, Gauss-Jordan, LU, Jacobi, Gauss-Seidel, CG)
- ⬜ Benchmark harness that loops over all 50 matrices × 6 methods
- ⬜ Visualization/dashboard
- ⬜ Final report

You're currently at the proposal stage — the deck already reflects everything in Section 4 correctly. The next real milestone, once the proposal is approved, is starting the implementation.
