# Convergence Theory Deep Dive: Root-Finding vs. the Hurwitz Shortcut

A worked-example companion to `SolveBench_Guide.md`. This file walks the two ways to check whether Jacobi or Gauss-Seidel converges on a matrix (solving for roots directly, vs. the paper's Hurwitz-criterion shortcut) through three real matrices, in full arithmetic detail, then lays out exactly what the base paper did versus what SolveBench adds.

---

## 1. The rule everything here is built on

Both Jacobi and Gauss-Seidel are **stationary iterative methods**: each update has the form

$$\vec{x}^{(k+1)} = M\vec{x}^{(k)} + \vec{c}$$

for some fixed matrix $M$ (the *iteration matrix*), rebuilt once from $A$ and never changed between steps. The classical theorem (not this paper's own result — cited from prior work) says:

$$\vec{x}^{(k)} \to \vec{x}_{\text{true}} \text{ for every starting guess} \iff \rho(M) < 1$$

where $\rho(M)$ is the *spectral radius* of $M$: the largest magnitude among all of $M$'s eigenvalues $\lambda_i$. So convergence is entirely a question about the roots of $M$'s characteristic polynomial $\det(\lambda I - M) = 0$.

**Jacobi's version of this polynomial** (equation 2.2 in the base paper) is built directly from $A$ without ever explicitly forming $M$: scale every diagonal entry of $A$ by $\lambda$, leave every off-diagonal entry untouched, and take the determinant:

$$\begin{vmatrix}
\lambda a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & \lambda a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{n1} & a_{n2} & \cdots & \lambda a_{nn}
\end{vmatrix} = 0$$

**Gauss-Seidel's version** (equation 2.3) uses a slightly different template: $\lambda$ scales every entry **on or below** the diagonal, leaving entries **above** the diagonal untouched.

$$\begin{vmatrix}
\lambda a_{11} & a_{12} & \cdots & a_{1n} \\
\lambda a_{21} & \lambda a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
\lambda a_{n1} & \lambda a_{n2} & \cdots & \lambda a_{nn}
\end{vmatrix} = 0$$

Two ways to use either polynomial to answer "does it converge?":

- **Path 1 — solve directly.** Find every root $\lambda_i$ explicitly, then check $|\lambda_i| < 1$ for each one.
- **Path 2 — Hurwitz shortcut.** Never find the roots. Build a second matrix (the Hurwitz matrix) from the polynomial's coefficients, and check the sign of a few of its determinants.

Both paths must always agree, since they're answering the exact same question two different ways. That agreement is what we're going to verify by hand, three times.

### What the Hurwitz matrix actually looks like

For the $n=3$ (three-unknown) case with real coefficients, the paper reduces the whole check to three numbers, $\alpha, \beta, \gamma$, computed from the polynomial's $p,q$ coefficients. Those three numbers get slotted into one fixed $6\times6$ template — always the same layout, only the numbers change:

$$H = \begin{pmatrix}
1 & 0 & -\beta & 0 & 0 & 0 \\
0 & \alpha & 0 & -\gamma & 0 & 0 \\
0 & 1 & 0 & -\beta & 0 & 0 \\
0 & 0 & \alpha & 0 & -\gamma & 0 \\
0 & 0 & 1 & 0 & -\beta & 0 \\
0 & 0 & 0 & \alpha & 0 & -\gamma
\end{pmatrix}$$

A **leading principal minor of order $k$** just means: cut out the top-left $k\times k$ block of $H$ and take its determinant. The check only needs the *even*-order ones:

- $\Delta_1$ = determinant of the top-left $2\times2$ block
- $\Delta_2$ = determinant of the top-left $4\times4$ block
- $\Delta_3$ = determinant of the *entire* $6\times6$ matrix $H$

If $\Delta_1, \Delta_2, \Delta_3$ are all positive, the method converges. That's the whole test — three sub-determinants of one fixed matrix shape. Each worked example below shows $H$ with its own $\alpha,\beta,\gamma$ plugged in, then reports these three numbers.

---

## 2. Example 1 — a well-behaved matrix (Path 1 is easy here)

$$A = \begin{pmatrix} 10 & 1 & 1 \\ 1 & 10 & 1 \\ 1 & 1 & 10 \end{pmatrix}$$

### Building the polynomial (Jacobi)

$$\begin{vmatrix} 10\lambda & 1 & 1 \\ 1 & 10\lambda & 1 \\ 1 & 1 & 10\lambda \end{vmatrix} = 1000\lambda^3 - 30\lambda + 2 = 0$$

Normalized to $\lambda^3 + p\lambda + q = 0$ form: divide by 1000, giving $p = -0.03$, $q = 0.002$.

### Path 1 — solve directly

Test $\lambda = -0.2$: $(-0.2)^3 - 0.03(-0.2) + 0.002 = -0.008 + 0.006 + 0.002 = 0$ ✓ — an exact root.

Factor it out: $\lambda^3 - 0.03\lambda + 0.002 = (\lambda + 0.2)(\lambda^2 - 0.2\lambda + 0.01)$.

Solve the remaining quadratic: discriminant $= 0.2^2 - 4(0.01) = 0$, giving a repeated root $\lambda = 0.1$.

**Roots: $-0.2,\ 0.1,\ 0.1$.** All magnitudes under 1 → **Jacobi converges.**

### Path 2 — Hurwitz shortcut

From $p = -0.03$, $q = 0.002$, compute three intermediate values (the paper's own formulas, real-coefficient case):

$$\alpha = \frac{3-p-3q}{1+p+q} = \frac{3.024}{0.972} \approx 3.111, \quad
\beta = \frac{3-p+3q}{1+p+q} = \frac{3.036}{0.972} \approx 3.123, \quad
\gamma = \frac{1+p-q}{1+p+q} = \frac{0.968}{0.972} \approx 0.996$$

Plug these three numbers into the $6\times6$ template from Section 1:

$$H = \begin{pmatrix}
1 & 0 & -3.123 & 0 & 0 & 0 \\
0 & 3.111 & 0 & -0.996 & 0 & 0 \\
0 & 1 & 0 & -3.123 & 0 & 0 \\
0 & 0 & 3.111 & 0 & -0.996 & 0 \\
0 & 0 & 1 & 0 & -3.123 & 0 \\
0 & 0 & 0 & 3.111 & 0 & -0.996
\end{pmatrix}$$

Now take the three leading principal minors:

- $\Delta_1$ = determinant of the top-left $2\times2$ block $\begin{pmatrix}1 & 0\\0 & 3.111\end{pmatrix}$ = $1\times3.111 - 0\times0 = \mathbf{3.111}$
- $\Delta_2$ = determinant of the top-left $4\times4$ block. Working it out (by cofactor expansion, or via the closed form $\Delta_2=\alpha(\alpha\beta-\gamma)$ that falls out of expanding it) gives $\approx \mathbf{27.14}$
- $\Delta_3$ = determinant of the *whole* $6\times6$ matrix $H$ shown above. Same process, one size bigger, gives $\approx \mathbf{75.76}$ (double-checked here with an independent Gaussian-elimination pass, since a $6\times6$ determinant by cofactor expansion alone is easy to slip up on)

All three positive → **Jacobi converges.** Same answer as Path 1 — but the numbers $-0.2$, $0.1$, $0.1$ never appeared anywhere in this path. All we ever touched were $\alpha, \beta, \gamma$ and three determinant signs.

### Checking Gauss-Seidel too

Jacobi and Gauss-Seidel have different iteration matrices, so in principle they need separate checks. For real $3\times3$ systems, the paper gives a shortcut for Gauss-Seidel specifically (equation 4.35) that skips the Hurwitz matrix entirely — three numbers pulled directly from $A$'s entries:

$$a = a_{11}a_{22}a_{33}, \qquad b = a_{12}a_{23}a_{31}, \qquad d = a_{21}a_{13}a_{32} - a_{13}a_{22}a_{31} - a_{32}a_{11}a_{23} - a_{21}a_{33}a_{12}$$

$$\text{converges} \iff |d| < |a+b| \ \text{ and } \ \left|\frac{b}{a}\right| < 1$$

For $A$: $a = 10\times10\times10 = 1000$, $b = 1\times1\times1 = 1$, and

$$d = (1)(1)(1) - (1)(10)(1) - (1)(10)(1) - (1)(10)(1) = 1 - 10 - 10 - 10 = -29$$

Check both conditions: $\left|\frac{b}{a}\right| = \frac{1}{1000} = 0.001 < 1$ ✓, and $|d| = 29 < |a+b| = 1001$ ✓.

Both hold → **Gauss-Seidel also converges** on $A$. No surprise: this matrix is strongly diagonally dominant, which is a well-known sufficient condition for *both* methods at once — matching Section 3.4's point that for well-behaved matrices, the two methods usually agree.

---

## 3. Example 2 — a matrix where Path 1 stops being practical

$$B = \begin{pmatrix} 4 & 2 & 1 \\ 1 & 5 & 2 \\ 2 & 1 & 6 \end{pmatrix}$$

### Building the polynomial (Jacobi)

$$\begin{vmatrix} 4\lambda & 2 & 1 \\ 1 & 5\lambda & 2 \\ 2 & 1 & 6\lambda \end{vmatrix} = 120\lambda^3 - 30\lambda + 9 = 0 \;\Longrightarrow\; \lambda^3 - 0.25\lambda + 0.075 = 0 \quad (p=-0.25,\ q=0.075)$$

### Path 1 — solve directly (where it breaks down)

Try the same trick as Example 1: test small rational candidates ($\pm1, \pm\tfrac12, \pm\tfrac34, \pm\tfrac14,\dots$) — none land on zero. Numerically bisecting instead:

$$f(-0.6) = 0.36, \qquad f(-0.65) = -1.485, \qquad f(-0.61) \approx 0.021$$

The real root is only found approximately, $\lambda \approx -0.611$ — an irrational number with no clean closed form reachable by factoring. The other two roots are a complex-conjugate pair, extractable only through the full cubic formula (which, for this coefficient pattern, forces intermediate complex numbers even though the final roots may be real — the "casus irreducibilis" problem mentioned in the guide). **Path 1 is technically still possible here, but no longer practical by hand.**

### Path 2 — same four mechanical steps, no slowdown

Same recipe as Example 1: compute $\alpha,\beta,\gamma$ from $p=-0.25,\,q=0.075$, plug them into the same $6\times6$ template $H$ from Section 1, then read off the three leading principal minors.

$$\alpha = \frac{3.025}{0.825}\approx3.667, \quad \beta=\frac{3.475}{0.825}\approx4.212, \quad \gamma=\frac{0.675}{0.825}\approx0.818$$

$$H = \begin{pmatrix}
1 & 0 & -4.212 & 0 & 0 & 0 \\
0 & 3.667 & 0 & -0.818 & 0 & 0 \\
0 & 1 & 0 & -4.212 & 0 & 0 \\
0 & 0 & 3.667 & 0 & -0.818 & 0 \\
0 & 0 & 1 & 0 & -4.212 & 0 \\
0 & 0 & 0 & 3.667 & 0 & -0.818
\end{pmatrix}$$

- $\Delta_1$ = top-left $2\times2$ determinant = $\mathbf{3.667}$
- $\Delta_2$ = top-left $4\times4$ determinant $\approx \mathbf{53.63}$
- $\Delta_3$ = full $6\times6$ determinant $\approx \mathbf{175.0}$

All positive → **Jacobi converges** (consistent with $B$ being diagonally dominant, a known sufficient condition). Path 2 finished in exactly the same shape of steps as Example 1, on the same fixed $6\times6$ template — it does not care whether the underlying roots are clean or ugly. **This is the entire point of the shortcut.**

### Checking Gauss-Seidel too

Same closed-form test as Example 1 ($a=a_{11}a_{22}a_{33}$, $b=a_{12}a_{23}a_{31}$, $d$ as before). For $B$: $a = 4\times5\times6 = 120$, $b = 2\times2\times2 = 8$, and

$$d = (1)(1)(1) - (1)(5)(2) - (1)(4)(2) - (1)(6)(2) = 1 - 10 - 8 - 12 = -29$$

$$\left|\frac{b}{a}\right| = \frac{8}{120} \approx 0.067 < 1 \ \checkmark \qquad\qquad |d| = 29 < |a+b| = 128 \ \checkmark$$

Both hold → **Gauss-Seidel also converges** on $B$, same as Jacobi. Again unsurprising: $B$ is diagonally dominant too, so both methods being safe here is expected. Example 3 is where that expectation finally breaks.

---

## 4. Example 3 — the paper's own matrix, where the two methods disagree

This is not a constructed example — it's lifted directly from the base paper (Section 4.2, "Example 2"), used there to prove that Gauss-Seidel is not always better than Jacobi.

$$C = \begin{pmatrix} -8 & 6 & -4 \\ -9 & 8 & 6 \\ 4 & -5 & 3 \end{pmatrix}$$

### Checking Jacobi

Building the Jacobi polynomial the same way as before:

$$\begin{vmatrix} -8\lambda & 6 & -4 \\ -9 & 8\lambda & 6 \\ 4 & -5 & 3\lambda \end{vmatrix} = -192\lambda^3 + 50\lambda - 36 = 0 \;\Longrightarrow\; \lambda^3 - \tfrac{25}{96}\lambda + \tfrac{3}{16} = 0 \quad (p\approx-0.2604,\ q=0.1875)$$

**Path 2 (Hurwitz):** same $6\times6$ template again, this matrix's own $\alpha,\beta,\gamma$ plugged in.

$$\alpha \approx 2.910, \qquad \beta \approx 4.124, \qquad \gamma \approx 0.595$$

$$H = \begin{pmatrix}
1 & 0 & -4.124 & 0 & 0 & 0 \\
0 & 2.910 & 0 & -0.595 & 0 & 0 \\
0 & 1 & 0 & -4.124 & 0 & 0 \\
0 & 0 & 2.910 & 0 & -0.595 & 0 \\
0 & 0 & 1 & 0 & -4.124 & 0 \\
0 & 0 & 0 & 2.910 & 0 & -0.595
\end{pmatrix}$$

- $\Delta_1$ = top-left $2\times2$ determinant = $\mathbf{2.910}$
- $\Delta_2$ = top-left $4\times4$ determinant $\approx \mathbf{33.19}$
- $\Delta_3$ = full $6\times6$ determinant $\approx \mathbf{77.47}$

All positive → **Jacobi converges.**

### Checking Gauss-Seidel

Same closed-form test used in Examples 1 and 2 (the paper's equation 4.35 for real $3\times3$ systems) — as a reminder:

$$a = a_{11}a_{22}a_{33}, \qquad b = a_{12}a_{23}a_{31}, \qquad d = a_{21}a_{13}a_{32} - a_{13}a_{22}a_{31} - a_{32}a_{11}a_{23} - a_{21}a_{33}a_{12}$$

$$\text{converges} \iff |d| < |a+b| \ \text{ and } \ \left|\frac{b}{a}\right| < 1$$

For $C$: $a = (-8)(8)(3) = -192$, $b = (6)(6)(4) = 144$, and expanding $d$ term by term:

$$d = (-9)(-4)(-5) - (-4)(8)(4) - (-5)(-8)(6) - (-9)(3)(6) = -180+128-240+162 = -130$$

Check the two conditions:

$$\left|\frac{b}{a}\right| = \left|\frac{144}{-192}\right| = 0.75 < 1 \ \checkmark \qquad\qquad |d| < |a+b| \;\Longrightarrow\; 130 < |{-48}| = 48 \ \times$$

The second condition fails. **Gauss-Seidel does not converge.**

### The result

On this exact matrix, **Jacobi converges and Gauss-Seidel does not** — the opposite of the "usual" trend. This is the concrete case behind the guide's Section 3.4, point 2 ("the two methods don't always agree") — not a hypothetical, an actual matrix with actual numbers where it happens.

---

## 5. What the base paper did, vs. what SolveBench does

| | **Base paper** (Khrapov & Volkov) | **SolveBench** (this project) |
|---|---|---|
| **Methods studied** | Jacobi and Gauss-Seidel only | Jacobi, Gauss-Seidel, Conjugate Gradient, Gauss elimination, Gauss-Jordan, LU (+ optionally SOR, Cholesky) |
| **Question asked** | *Will it converge, yes or no?* | *Does it converge, how many iterations, how long does it take, how accurate is the answer?* |
| **How convergence is checked** | Theoretical criterion only (root location / Hurwitz determinant signs) — never actually runs the iteration | Actually runs each method on each matrix and observes the real behavior |
| **Matrix source** | Hand-picked toy examples, plus 100,000 randomly generated matrices | 48 real matrices from the SuiteSparse Matrix Collection (real power grids, real structural models, real circuits) |
| **Matrix size** | 2 to 5 unknowns | 112 to 5,860 unknowns |
| **Runtime / iteration count measured?** | No | Yes — this is one of the core outputs |
| **Real-world data used?** | No, never | Yes, exclusively |
| **Direct methods considered?** | No | Yes (Gauss elimination, Gauss-Jordan, LU) |
| **Deliverable** | A theoretical convergence-region comparison | An empirical "which solver, for which real system" decision guide |

**One-sentence summary of the paper:** a pure convergence-theory comparison of two 19th-century iterative methods, on tiny synthetic systems — nothing about speed, nothing about real data, nothing about any other solver.

**One-sentence summary of SolveBench:** takes that same two-method comparison and everything it left untested — speed, scale, real data, and four more solver types — and actually measures it.

---

## 6. The three real domains, explained row by row

The "Problem Statement & Scope" table (linear.tex) has one row per domain. Here's what each column of each row actually means, and where the raw values come from.

### Row 1 — Power-grid admittance | 11 | 118 to 1,723 buses

**Domain.** An admittance matrix (also called a "Y-bus" matrix in power systems) describes how electrical current relates to voltage at every connection point in a grid. Each entry comes from the physical conductance of the transmission line joining two points. Solving $Ax=b$ against it is exactly what a power-flow study computes to find the voltage at every point in a real grid.

**Systems: 11.** `bcspwr03` through `bcspwr09` (7 files) plus `494_bus`, `662_bus`, `685_bus`, `1138_bus` (4 files).

**Size range: 118 to 1,723 buses.** A "bus" is the power-systems term for a node/connection point in the grid — the same role $n$ plays everywhere else in this document. 118 (`bcspwr03`) is the smallest; 1,723 (`bcspwr09`) is the largest.

**Caveat worth knowing (not on the slide):** only the 4 `*_bus` files are declared `real` type in their Matrix Market header. All 7 `bcspwr*` files are `pattern` type — they record only *which* connections exist, not the actual conductance values. Those 7 need synthetic values assigned to their nonzero positions before they're usable in the benchmark.

### Row 2 — Structural (stiffness & mass) | 33 | 112 to 4,884 DOF

**Domain.** Bundles two related matrix types. A **stiffness matrix** encodes how much force is needed to produce a given displacement at each point of a structure, built from material properties and geometry — used alone for static safety checks (does this beam bend too much under load?). A **mass matrix** encodes how mass/inertia is distributed across those same points — used together with a stiffness matrix for dynamic analysis (how does this bridge vibrate?).

**Systems: 33.** 23 `bcsstk*` files (stiffness) + 10 `bcsstm*` files (mass), both drawn from real finite-element models of actual structures.

**Size range: 112 to 4,884 DOF.** "DOF" (degrees of freedom) is the structural-engineering term for $n$ — each node in the finite-element mesh can move in a few possible directions, and each direction is one unknown, one equation. 112 is the smallest system; 4,884 (`bcsstk16`) is the largest in the entire 48-matrix set.

**Good news, unlike Row 1:** none of these 33 files are `pattern` type — all are already populated with real numeric values, no synthetic-value prep needed.

### Row 3 — Circuit simulation | 4 | 2,624 to 5,860 nodes

**Domain.** Matrices arising from simulating an electronic circuit's behavior — how current and voltage propagate through resistors, capacitors, and other components as the circuit operates. This is the kind of matrix a tool like SPICE solves internally.

**Systems: 4.** `circuit_1`, `circuit_2` (Bomhof group) and `meg1`, `meg4` (Grund group) — all genuinely real circuit-simulation matrices, downloaded and verified directly against their SuiteSparse source pages.

**Size range: 2,624 to 5,860 nodes.** "Node" here means a circuit node, the junction point where components connect — the same role as "bus" for power grids, "DOF" for structures. 2,624 (`circuit_1`) is the smallest; 5,860 (`meg4`) is the largest in the whole dataset, exceeding even the largest structural matrix.

**Also good news:** all 4 are `real` type, fully populated, confirmed at download time — no extra prep needed here either.

### Why this matters for implementation, not just the pitch

Two of the three domains (structural, circuit) are ready to use as-is. The power-grid domain needs a small preprocessing step first: assigning synthetic conductance values to the 7 `pattern`-type `bcspwr*` matrices before they can be run through any solver. Budget that as a small, one-time task for whoever owns the data pipeline (Section 4.8 of the guide) — not a blocker, just a step that shouldn't be discovered mid-benchmark.

---

## 7. Simulating the benchmark harness: residual, error, and condition number

This is what the "Benchmark harness" box in `linear.tex`'s methodology diagram is actually doing, worked by hand on two matrices you already know from Sections 2–3, across one direct method (Gauss elimination) and one iterative method (Jacobi). Ground truth throughout: $\vec{x}_{\text{true}} = (1,1,1)$.

### Setting up ground truth

$$\vec{b}_A = A\vec{x}_{\text{true}} = \begin{pmatrix}10+1+1\\1+10+1\\1+1+10\end{pmatrix} = \begin{pmatrix}12\\12\\12\end{pmatrix}, \qquad
\vec{b}_B = B\vec{x}_{\text{true}} = \begin{pmatrix}4+2+1\\1+5+2\\2+1+6\end{pmatrix} = \begin{pmatrix}7\\8\\9\end{pmatrix}$$

($\vec{b}_A$ is exactly the same right-hand side already used in Section 2's worked example — no coincidence, same matrix, same true answer.)

### Matrix A, Gauss elimination (direct)

Eliminate column 1 using row 1 ($R_2 \to R_2 - 0.1R_1$, $R_3 \to R_3 - 0.1R_1$):

$$\begin{pmatrix}10 & 1 & 1 & | & 12\\0 & 9.9 & 0.9 & | & 10.8\\0 & 0.9 & 9.9 & | & 10.8\end{pmatrix}$$

Eliminate column 2 using row 2 (factor $= 0.9/9.9 = 1/11$): row 3 becomes $\frac{108}{11}x_3 = \frac{108}{11}$, so $x_3 = 1$ **exactly** (both sides are the identical fraction $108/11$, not a rounded coincidence). Back-substituting: $9.9x_2 + 0.9(1) = 10.8 \Rightarrow x_2 = 1$; then $10x_1 + 1 + 1 = 12 \Rightarrow x_1 = 1$.

$$\vec{x}_{\text{found}} = (1,1,1) = \vec{x}_{\text{true}} \quad\Rightarrow\quad \textbf{residual} = 0, \quad \textbf{error} = 0$$

This is the direct-method signature: one pass, exact answer (in real floating-point code this would be residual/error on the order of $10^{-15}$, not literally 0 — but by hand, with exact fractions, it's genuinely zero).

### Matrix A, Jacobi (iterative), first 3 steps

Reusing the update formulas and $b=(12,12,12)$ from Section 2, starting at $(0,0,0)$:

| Iteration | $\vec{x}$ | Error $\|\vec{x}-\vec{x}_{\text{true}}\|$ | Residual $\|A\vec{x}-\vec{b}\|$ |
|---|---|---|---|
| 1 | $(1.2, 1.2, 1.2)$ | $0.346$ | $4.157$ |
| 2 | $(0.96, 0.96, 0.96)$ | $0.069$ | $0.831$ |
| 3 | $(1.008, 1.008, 1.008)$ | $0.014$ | $0.166$ |

Take the ratio between consecutive errors: $0.069/0.346 = 0.2$, and $0.014/0.069 = 0.2$ again. **That 0.2 is not a coincidence** — it's exactly $|-0.2|$, the magnitude of the dominant root we found for matrix A's Jacobi polynomial back in Section 2 ($-0.2, 0.1, 0.1$). The error shrinks geometrically at precisely the rate the theory predicted, iteration after iteration. Residual shrinks at the same 0.2 rate here too, because $(1,1,1)$ happens to be an eigenvector of $A$ (its row sums are all 12), which makes residual and error exactly proportional for this particular matrix.

### Matrix B, Gauss elimination (direct)

$\det(B) = 4(30-2) - 2(6-4) + 1(1-10) = 112 - 4 - 9 = 99 \neq 0$, so a unique exact solution exists. Elimination (same mechanics as above, omitted for space) recovers $\vec{x}_{\text{found}} = (1,1,1)$ exactly, since $B\vec{x}_{\text{true}}$ was constructed to equal $\vec{b}_B$ exactly.

$$\textbf{residual} = 0, \quad \textbf{error} = 0$$

### Matrix B, Jacobi (iterative), first 3 steps

Using the update formulas from Section 3 and $b=(7,8,9)$, starting at $(0,0,0)$:

| Iteration | $\vec{x}$ | Error | Residual |
|---|---|---|---|
| 1 | $(1.75, 1.6, 1.5)$ | $1.083$ | $8.406$ |
| 2 | $(0.575, 0.65, 0.65)$ | $0.652$ | $5.169$ |
| 3 | $(1.2625, 1.225, 1.2)$ | $0.399$ | $3.145$ |

Error ratios here: $0.652/1.083 \approx 0.60$, $0.399/0.652 \approx 0.61$ — converging toward $0.611$, the magnitude of $B$'s dominant root from Section 3 ($\lambda \approx -0.611$). Unlike matrix A, this ratio isn't exactly constant from iteration 1 (B lacks A's special symmetric structure), but it's visibly homing in on the same theoretical number as the iterations continue.

### Condition numbers

**Matrix A** is symmetric, so its condition number is just the ratio of its largest to smallest eigenvalue magnitude. Notice $A = 9I + J$ where $J$ is the all-ones matrix (every entry 1): $J$ has eigenvalues $3$ (once) and $0$ (twice), so $A$'s eigenvalues are $9+3=12$ and $9+0=9$ (twice).

$$\kappa(A) = \frac{12}{9} \approx \mathbf{1.33}$$

Very well-conditioned — consistent with $A$'s strong diagonal dominance.

**Matrix B** isn't symmetric, so we use $\kappa_\infty(B) = \|B\|_\infty \cdot \|B^{-1}\|_\infty$. $\|B\|_\infty$ = largest row-sum of absolute values = $\max(7,8,9) = 9$. Computing $B^{-1}$ via the cofactor/adjugate method gives $B^{-1} = \frac{1}{99}\begin{pmatrix}28&-11&-1\\-2&22&-7\\-9&0&18\end{pmatrix}$, whose largest row-sum is $40/99$.

$$\kappa_\infty(B) = 9 \times \frac{40}{99} = \frac{360}{99} \approx \mathbf{3.64}$$

Still well-conditioned, but noticeably more sensitive than $A$ — consistent with $B$ being the matrix whose polynomial roots came out "uglier" back in Section 3.

### Everything side by side

| Matrix | Method | Iterations | Residual | Error | Condition number |
|---|---|---|---|---|---|
| A | Gauss elimination | — | 0 | 0 | 1.33 |
| A | Jacobi (step 3) | 3 | 0.166 | 0.014 | 1.33 |
| B | Gauss elimination | — | 0 | 0 | 3.64 |
| B | Jacobi (step 3) | 3 | 3.145 | 0.399 | 3.64 |

**What this table demonstrates, in one glance:** direct methods have no "iterations" column at all (dash, not zero — the concept doesn't apply) and land on essentially exact answers in one pass. Iterative methods have a genuine, shrinking-but-nonzero residual and error that depend on how many steps you let them run — and critically, *how fast* they shrink is not arbitrary, it's set exactly by the matrix's dominant root magnitude from Sections 2–3, the same number theory already predicted before a single iteration was run. Condition number is the one column that doesn't change between the two methods on the same matrix — it's a property of $A$ alone, exactly as established earlier in this conversation.

---

## 8. Presentation script (Bangla), all 4 slides, under 5 minutes

Matches the current final state of `linear.tex` exactly (8 solvers, 48 matrices, 3 domains). Rough timing noted per slide — total lands around 4 to 4.5 minutes, leaving buffer for pacing and a natural pause between slides.

### Slide 1 — Title (~20–25 sec)

আসসালামু আলাইকুম। আমাদের প্রজেক্টের নাম **SolveBench**। প্রতিটা blackout-prevention স্টাডি, প্রতিটা bridge safety check, শেষ পর্যন্ত এই একটা সমীকরণে গিয়ে দাঁড়ায় — **A x = b**। আমরা খুঁজে বের করছি, কোন numerical method এই সমীকরণ সমাধানের জন্য আসলে বিশ্বাসযোগ্য, আর কখন।

### Slide 2 — Base Paper (~70–80 sec)

আমাদের base paper হলো Khrapov আর Volkov-এর ২০২৪ সালের একটা paper, International Journal of Open Information Technologies-এ প্রকাশিত — *Comparative Analysis of Jacobi and Gauss-Seidel Iterative Methods*। এই paper Jacobi আর Gauss-Seidel, দুইটা classical iterative method, এদের convergence নিয়ে কাজ করে।

এরা দেখিয়েছে, কোন matrix-এর জন্য এই দুইটা method আসলে কাজ করবে, সেটা একদম exact ভাবে বের করা সম্ভব, root-location reasoning অথবা Hurwitz-style stability test ব্যবহার করে। ২ আর ৩ unknown-এর জন্য এটা হাতে করে প্রমাণ করেছে, আর তারপর এক লক্ষ random matrix দিয়ে ৫ unknown পর্যন্ত পরীক্ষা করে দেখিয়েছে একই pattern ধরে থাকে কিনা।

কিন্তু বড় সমস্যা হলো, এই পুরো analysis মাত্র পাঁচটা unknown পর্যন্ত, আর সবগুলো matrix সম্পূর্ণ random, বাস্তব কোনো data না। বাস্তব engineering system-এ হাজার হাজার unknown থাকে। এই paper কখনো real data-তে test করেনি।

### Slide 3 — Problem Statement & Scope (~70–80 sec)

আমাদের project-এর মূল প্রশ্ন হলো, প্রতিটা real engineering simulation, সেটা power grid model করা হোক, structure-এর safety check হোক, বা circuit simulate করা হোক, সবগুলোই শেষমেশ একই core step-এ গিয়ে দাঁড়ায়: A x = b সমাধান করা। ভুল solver বেছে নিলে, যেটা milliseconds-এ শেষ হওয়ার কথা, সেটা হয় অনেক ধীর হয়ে যায়, নয়তো কখনোই converge করে না।

এই project spec-এর Exploratory and Comparative Study category-তে পড়ে — আমরা আটটা classical direct আর iterative solver বেছে নিয়ে বাস্তব, synthetic না, engineering data-র উপর benchmark করছি।

আমাদের কাছে আছে ৪৮টা real matrix, তিনটা ভিন্ন domain থেকে — power-grid admittance matrix, structural stiffness আর mass matrix, আর circuit simulation matrix। এদের size ১১২ থেকে প্রায় ৫৮৬০ unknown পর্যন্ত, যেটা base paper-এর পাঁচ unknown limit-এর চেয়ে বহুগুণ বড়।

### Slide 4 — Expected Methodology (~70–80 sec)

Methodology-টা সহজ। প্রতিটা real sparse matrix নিয়ে, আমরা সেটা আটটা solver দিয়েই সমাধান করব — direct method যেমন Gauss elimination, Gauss-Jordan, LU, Cholesky, আর iterative method যেমন Jacobi, Gauss-Seidel, SOR, আর Conjugate Gradient। তারপর একটা benchmark harness runtime, iteration count, residual, error, আর condition number মাপবে।

Ground truth-এর trick হলো, আমরা আগে থেকেই একটা known x বানিয়ে নিব, b = A x হিসাব করে ফেলব, তারপর সেই b দিয়ে প্রতিটা method solve করাবো, যেন আমরা জানি আসল উত্তর কী। এতে পুরো experiment fully deterministic, "correct" মানে কী তা নিয়ে কোনো দ্বিধা থাকে না।

শেষমেশ আমাদের payoff হলো একটা concrete, evidence-based উত্তর: কোন solver, কোন বাস্তব system-এর জন্য সবচেয়ে ভালো, এই প্রশ্নটার উত্তর, যেটা প্রতিটা real engineering simulation শুরু করার আগেই জানা দরকার।

### Delivery notes

- Practice out loud once with a timer before the actual pitch — reading speed varies a lot person to person, and this script is written to *land* around 4 to 4.5 minutes, not guaranteed to.
- Slide 2's gap box ("কখনো real data-তে test করেনি") and Slide 3's payoff line are the two moments worth a short pause before speaking — they're the sentences doing the most persuasive work.
- If a question interrupts mid-pitch (common in a 5-minute live setting), Slide 3's problem paragraph and Slide 4's ground-truth paragraph are the safest to trim first if you're running long, since Slides 1 and 2 set up context the rest depends on.
