# SolveBench — The Complete Story, Explained From Scratch

**Purpose of this document.** You are about to present this project to teammates who may
not have opened the repository before. This file is written so that a person with zero
prior context can read it top to bottom and come out understanding: what problem we are
solving, what the original research paper did, what we built on top of it, every
decision we made and why, what every piece of code does, what every result means, and
how to explain any chart to someone else.

Read it in order. Each section assumes only the sections before it.

CSE 402 (Numerical Analysis Lab), BUET, Section A2.

---

# Table of Contents

**Part 1 — The Idea**
1. [What problem are we solving?](#1-what-problem-are-we-solving)
2. [The research paper we are extending](#2-the-research-paper-we-are-extending)
3. [What we implemented from the paper](#3-what-we-implemented-from-the-paper)
4. [How we scaled the paper's experiment up](#4-how-we-scaled-the-papers-experiment-up)
5. [The headline results, in one place](#5-the-headline-results-in-one-place)

**Part 2 — The Story of How We Got Here**
6. [Milestone 0 — the starting point: APK](#6-milestone-0--the-starting-point-apk)
7. [Milestone 1 — building an honest measuring instrument](#7-milestone-1--building-an-honest-measuring-instrument)
8. [Milestone 2 — the discovery that changed everything: the 491](#8-milestone-2--the-discovery-that-changed-everything-the-491)
9. [Milestone 3 — the idea: choose the diagonal](#9-milestone-3--the-idea-choose-the-diagonal)
10. [Milestone 4 — proving what can and cannot work](#10-milestone-4--proving-what-can-and-cannot-work)
11. [Milestone 5 — building the objective, and a false start](#11-milestone-5--building-the-objective-and-a-false-start)
12. [Milestone 6 — four scipy routines that could not be trusted](#12-milestone-6--four-scipy-routines-that-could-not-be-trusted)
13. [Milestone 7 — the reordering study result](#13-milestone-7--the-reordering-study-result)
14. [Milestone 8 — the ILU hole](#14-milestone-8--the-ilu-hole)
15. [Milestone 9 — the relaxation factor](#15-milestone-9--the-relaxation-factor)
16. [Milestone 10 — assembling the pipeline](#16-milestone-10--assembling-the-pipeline)

**Part 3 — The Technical Reference**
17. [Pseudocode: the paper's methods](#17-pseudocode-the-papers-methods)
18. [Pseudocode: our novel contributions](#18-pseudocode-our-novel-contributions)
19. [The dataset, explained with real examples](#19-the-dataset-explained-with-real-examples)
20. [Every code file, and what it produces](#20-every-code-file-and-what-it-produces)
21. [Every output table and figure](#21-every-output-table-and-figure)
22. [Mistakes we made, and what changed when we fixed them](#22-mistakes-we-made-and-what-changed-when-we-fixed-them)
23. [Where things stand right now](#23-where-things-stand-right-now)

---
---

# Part 1 — The Idea

---

## 1. What problem are we solving?

Almost every engineering simulation — a circuit, a bridge, an airflow model, a power
grid — eventually turns into one piece of algebra:

```
A x = b
```

`A` is a big matrix of numbers describing how the parts of the system relate to each
other. `b` is what you are pushing into the system (a voltage, a force, a heat source).
`x` is the answer you want (the currents, the displacements, the temperatures).

For a real engineering problem, `A` can have **thousands or millions of rows and
columns**, but it is almost always **sparse** — the huge majority of its entries are
exactly zero, because most parts of a physical system only interact with their close
neighbours. A circuit component only connects to the wires touching it, not to every
other component in the whole circuit.

There are two completely different families of technique for solving `Ax = b`:

### Direct methods

Factor the matrix, the same way you learned to do Gaussian elimination by hand in
school: turn `A` into a product of two triangular matrices, `A = LU`, and then solve two
easy triangular systems. This gives you the **exact** answer (up to floating-point
rounding) in a fixed, predictable number of steps.

The problem: factoring can create a lot of new nonzero entries where `A` had zeros — this
is called **fill-in** — and for a big enough matrix that can need more memory than
exists on the computer.

### Iterative methods

Start from a guess (usually `x = 0`) and repeatedly improve it:

```
x₀  →  x₁  →  x₂  →  x₃  →  ...
```

At every step you check how far off you are — the **residual**, `‖A x − b‖` — and you
stop once it is small enough. Each step is cheap (basically one multiplication of the
matrix by a vector), so iterative methods can handle far bigger matrices than direct
methods can. But there is a catch: **the steps might never get you closer to the
answer.** Whether they do depends entirely on the structure of `A`. This dependency is
the entire subject of this project.

---

## 2. The research paper we are extending

Our starting point is a real, published paper:

> P. Khrapov and N. Volkov, *Comparative Analysis of Jacobi and Gauss-Seidel Iterative
> Methods*, International Journal of Open Information Technologies, vol. 12, no. 2, pp.
> 23–34, 2024.

**What the paper does.** It studies two of the oldest and simplest iterative methods,
Jacobi and Gauss-Seidel (both explained in full in Section 17 with pseudocode). For
systems with only **2 or 3 unknowns**, the authors work out, by hand, exactly which
matrices make each method converge and which make it diverge — a precise mathematical
region. They then generate a large number of **random** matrices with up to 5 unknowns
and check, statistically, how often their theoretical predictions hold up.

**What the paper does not do.** It never once touches a matrix that came from a real
engineering problem. Every matrix in the paper is either tiny (2×2 or 3×3, worked out by
hand) or randomly generated by a computer. Random numbers do not look like real
engineering data — a real circuit matrix, for example, has a very particular pattern of
zeros and nonzeros that comes from which components are physically wired to which other
components. A random matrix has no such pattern at all.

**The gap, and our question.** Nobody has checked whether the paper's conclusions
survive contact with matrices that actual engineers actually have to solve, at sizes
where the methods are actually used. That is precisely the gap this project fills.

---

## 3. What we implemented from the paper

Everything the paper studies, we implemented ourselves, from first principles, in
`src/solvebench/iterative_solvers.py`:

* **Jacobi's method** — the simplest stationary iterative method (full explanation and
  pseudocode in Section 17).
* **Gauss-Seidel** — a small change to Jacobi that usually converges faster.
* **The convergence theory behind both** — the *spectral radius* of something called the
  "iteration matrix," which is the single number that tells you, mathematically, whether
  an iterative method will converge, and roughly how fast. This lives in
  `src/solvebench/spectral.py`.
* **The paper's own experimental method** — generating random matrices and checking
  convergence statistically. We reproduced this too, as our own `random_matrices` domain
  (100 matrices, sizes 5 to 300), specifically so we could compare our results directly
  against the paper's own validation approach.

We did not just implement the two methods named in the paper — we also implemented a
third stationary method, **SOR** (Successive Over-Relaxation), which is Gauss-Seidel with
one extra tuning knob. SOR turns out to matter a great deal to our own contribution,
described starting in Section 9.

---

## 4. How we scaled the paper's experiment up

| | The paper | This project |
|---|---|---|
| Matrix source | hand-derived 2×2/3×3, then random | **real matrices from real engineering problems** |
| Matrix count | a handful by hand, then a batch of random ones | **927 unique matrices** |
| Matrix size | up to 5 unknowns | **up to 10,000 unknowns** |
| Domains | none (random numbers have no domain) | **26 real engineering domains** (circuits, structures, fluid dynamics, power grids, chemistry, economics, and more) |
| Methods compared | Jacobi and Gauss-Seidel only | **16 methods**, across 5 families (see Section 17) |
| Success criterion | did the theory's predicted region match reality | **the residual, measured independently for every method, every time** |

Where do the 927 matrices come from? The **SuiteSparse Matrix Collection**, the standard
public library of real sparse matrices, each contributed by engineers or scientists from
an actual project. Every matrix in it is tagged with the domain it came from. This is
what lets us ask a question the paper's random matrices simply cannot answer: *"does
Gauss-Seidel actually work on circuit-simulation matrices?"* rather than only *"does it
work on numbers picked at random?"*

Section 19 walks through this dataset in full, with real matrix names and numbers.

---

## 5. The headline results, in one place

These are the numbers you should lead with if you only have two minutes.

### The single most important fact about the corpus

```
Jacobi, Gauss-Seidel and SOR are ALL undefined on 491 of 927 real matrices (53%)
```

This is not the methods being slow, or inaccurate. They are **mathematically undefined**
on more than half the corpus, because of a zero on the diagonal (explained fully in
Section 8). This single fact is the reason the rest of the project exists.

### On the paper's own test bed, real numbers make the theory look worse

```
100 random matrices (exactly the kind the base paper studies), sizes 5 to 300:

  Jacobi          0 solved / 100
  Gauss-Seidel    0 solved / 100
  a direct solver 100 solved / 100
```

And the spectral radius that governs convergence grows roughly in proportion to the
matrix size on random matrices — so the paper's own validation method stops working
almost as soon as you make the matrices bigger than the paper tested.

### A sparse direct solver dominates on this corpus

```
best direct solver (spsolve)     874 / 927 solved
best iterative configuration     617 / 927 solved
```

### Our own contribution: a preprocessing pipeline

```
                    as given   with our pipeline
Jacobi                  76          124        (+63%)
Gauss-Seidel           122          208        (+70%)
SOR                    119          184        (+55%)

199 systems recovered. Zero lost.
```

The solvers themselves are never touched. We only change **which row of the matrix ends
up on the diagonal**, and, for SOR, **one tuning number**. That is the entire idea, and
Sections 9 through 16 tell the story of how we found it, step by step, including the
false starts and the mistakes.

---
---

# Part 2 — The Story of How We Got Here

This part is written as a timeline. Nothing here is invented after the fact — this is
the actual order in which things were tried, discovered, and fixed. It exists so that
when you present this project, you can explain *why* each decision was made, not just
what the final code does.

---

## 6. Milestone 0 — the starting point: APK

Before any of the analysis in this document existed, the group's original course
proposal already contained an idea for a "novel" contribution. It was a rule for
picking which solver to run, based on a simple test of the matrix:

```
if A is symmetric:
    use Preconditioned Conjugate Gradient (PCG)
else:
    use Preconditioned BiCGSTAB
```

Both branches use the same preconditioner: **ILU** (Incomplete LU — a cheap,
approximate factorization used to speed up an iterative method; explained fully in
Section 17). The group called this rule **APK**, and it was written up as the project's
proposed novelty.

**What we found when we measured it honestly.** This exact decision rule — "symmetric
means use CG, otherwise use BiCGSTAB" — already has a name in the numerical-methods
literature. It is the default dispatch rule described in Barrett et al.'s *Templates for
the Solution of Linear Systems* (SIAM, 1994), and it is the default behaviour built into
both PETSc (a widely-used scientific computing library) and MATLAB's backslash operator.
It was not something we invented; it was something everybody already does.

Worse, when we actually measured what this rule contributes on top of *always* using
BiCGSTAB (never bothering to check symmetry at all), the answer was almost nothing:

```
ILU-Krylov (dispatched)     617 of 736 solved
always ILU-BiCGSTAB         616 of 736 solved
                             ------------------
                             +1 matrix
```

**One matrix, out of 736.** That number is why the project needed a genuine new
contribution — APK is kept in the final benchmark today, but only as an honestly
labelled *baseline*, not as our novelty. You will find it in the code as
`ilu_krylov_dispatched` in `src/solvebench/reference_solvers.py`, with a docstring that
says exactly what is written above: this is a known, standard technique, and it is worth
one matrix in 736. The figure `17_dispatch_ablation.png` shows this comparison visually.

This is the first lesson of the whole project: **measure everything honestly, including
your own idea, before you claim it is new.**

---

## 7. Milestone 1 — building an honest measuring instrument

Before any new idea could be tested fairly, the benchmark itself had to be trustworthy.
The very first version of this project's code had several serious problems that were
only found by carefully auditing the results:

* **A solver's own claim of success was trusted.** If a direct-solver routine ran
  without throwing an error, it was marked "successful" — even though some of those
  "successes" had a relative residual of **9.71 × 10¹²** (an answer wrong by twelve
  orders of magnitude).
* **Success rates were divided by the whole corpus**, even for methods that are only
  mathematically defined on a small slice of it. This punished specialised methods
  unfairly.
* **Refinement (a technique that polishes an answer with one extra cheap step) was only
  given to one method**, which is exactly why that method looked most accurate — the
  advantage was the refinement, not the method.
* **Gauss-Seidel and SOR were re-factoring their triangular system on every single
  iteration**, inflating their measured running time by a factor of 11 to 14.

Fixing these was not optional — it was the precondition for everything else in this
project to mean anything. The fixes are explained in full, one by one, in Section 22.
The result of this work is `src/solvebench/metrics.py`, the single place in the codebase
that decides whether a solve succeeded (Section 8 explains its rules in detail), and a
completely rebuilt `src/solvebench/iterative_solvers.py`.

---

## 8. Milestone 2 — the discovery that changed everything: the 491

Once the harness could be trusted, the first full sweep over all 927 real matrices with
all 16 methods produced one number that reshaped the entire rest of the project.

Jacobi, Gauss-Seidel, and SOR all work by splitting the matrix `A` into three pieces:

```
A = D + L + U
```

`D` is the diagonal (just the entries `a_11, a_22, a_33, ...`), `L` is everything below
the diagonal, `U` is everything above it. Every one of these three methods needs to
**divide by the diagonal entries**. If even one diagonal entry `a_ii` is exactly zero,
that division is impossible, and the method is not merely slow or wrong — it is
**undefined**, in the same way that `1/0` is undefined.

On our 927-matrix corpus:

```
491 of 927 matrices (53%) have at least one zero on the diagonal
```

This is the single most consequential fact discovered in this project. Half the corpus
was **off-limits** to Jacobi, Gauss-Seidel, and SOR before a single iteration was ever
attempted. Any comparison that does not separate "the method could not even try" from
"the method tried and failed" is comparing the wrong things. This is why the codebase has
eight distinct outcome categories rather than a simple pass/fail (fully listed in
Section 8's counterpart, and detailed with all eight in Section 22).

This discovery is what turned the project's central question from *"is Jacobi slow?"*
into *"can we make Jacobi applicable in the first place?"* — a much bigger and more
answerable question.

---

## 9. Milestone 3 — the idea: choose the diagonal

Once the 491 was understood, an important realisation followed. The rows of a matrix are
just a *list* — row 1, row 2, row 3, and so on — and that order was decided by whoever
built the matrix (the meshing software, the circuit simulator, the numbering convention
of the engineer). **The order is arbitrary. It has nothing to do with which solver you
will later use on the matrix.**

Crucially, you can **reorder the rows without changing the answer at all**:

```
A' = P A          b' = P b            (P is a permutation — just a row shuffle)

If  A x = b   then   A' x = P A x = P b = b'
```

The unknown `x` you get out is identical either way. Permuting rows is completely free
in terms of correctness — it only changes which entries of `A` end up sitting on the
diagonal.

**A tiny worked example** makes this concrete:

```
A = [ 0   5 ]        a_11 = 0.  D does not exist.  Jacobi/Gauss-Seidel/SOR CANNOT run.
    [ 3   1 ]

Swap the two rows:

A'= [ 3   1 ]        a_11 = 3, a_22 = 1.  Now every method can run.
    [ 0   5 ]
```

Same equations, same answer, completely different eligibility for the stationary
methods. This single idea — **choose which row goes where, before you hand the matrix to
the solver** — is the seed of the entire contribution described in the rest of Part 2.

---

## 10. Milestone 4 — proving what can and cannot work

Before searching for a good row order, we needed to know what kinds of changes to the
matrix *could possibly* help, so the search would not waste time on something
mathematically pointless. Three candidate ideas were checked, and two were ruled out by
direct proof, not by trial and error:

* **Scaling every row by some number.** It turns out this leaves the convergence
  behaviour of Jacobi **exactly, algebraically unchanged** — you can write out the
  formula and see the scaling factor cancels perfectly.
* **Scaling every column by some number.** This is what mathematicians call a
  *similarity transform*, and a similarity transform never changes a matrix's spectral
  radius (the one number that governs convergence).
* **Permuting rows *and* columns together, symmetrically** (a common trick used to
  reduce fill-in for direct solvers, called reordering). When measured on real matrices,
  this barely moved the spectral radius at all — the fifth decimal place, from `0.99012`
  to `0.99011`.

**Row permutation on its own** is the only one of these operations that actually changes
which numbers sit on the diagonal, and therefore the only one that can change whether
Jacobi, Gauss-Seidel, or SOR are even *applicable*, let alone how fast they converge.
This proof is what turned "let's try shuffling rows" from a guess into a principled,
justified plan — we searched over row permutations specifically because we could show
nothing else was worth searching over.

---

## 11. Milestone 5 — building the objective, and a false start

Given that row permutation is the only lever, the next question is: **which** row
permutation? There are `n!` possible orderings for an `n`-row matrix — far too many to
try one by one. This has to be posed as an *optimisation problem*: pick one entry per
row (and one per column, since it must be a permutation) to sit on the diagonal, so as to
maximise or minimise some score. In the literature, this is called an **assignment
problem**, and it can be solved *exactly* rather than approximately, using well-known
algorithms.

**The existing, industry-standard tool for this is called MC64** (after Duff and
Koster's original algorithm). MC64 picks the diagonal that makes the **product of the
absolute diagonal values as large as possible**:

```
MC64's goal:   maximise    |a_11| × |a_22| × ... × |a_nn|
```

This is an excellent objective **for direct solvers** — a big number on the diagonal
means the elimination process is numerically stable. But MC64 was never designed with
*iterative* convergence in mind, and it shows: a diagonal entry of size 1000 looks
wonderful in isolation, but if the rest of that row sums to 50,000, the row is still
terrible for Jacobi or Gauss-Seidel. MC64 cannot see that, because it only ever looks at
the single diagonal number, never the rest of the row.

**Our proposed objective** looks at the whole row instead. Define, for row `i` with
diagonal candidate `a_ii`:

```
ratio_i  =  ( sum of |a_ij| for every other column j )  /  |a_ii|
```

This `ratio_i` is not an arbitrary score. **A `ratio_i` below 1 for every row is exactly
the classical mathematical condition called "strict diagonal dominance," which is a
sufficient condition to *guarantee* both Jacobi and Gauss-Seidel converge.** So the ratio
is the precise quantity the convergence theorem is stated in terms of — not something we
invented, but something we chose to *optimise directly*, which nobody had previously
done for this purpose.

We call minimising the **worst** (largest) such ratio across all rows the **bottleneck
objective**, because it is a "bottleneck assignment problem" in the same family as MC64's
"maximum weight assignment problem," just with a different scoring rule. The exact
algorithm (binary search plus bipartite matching) is given as pseudocode in Section 18.

**The false start.** A second candidate objective, called **min-sum**, was also
implemented — the idea being to minimise the *total* of all the ratios instead of just
the worst one, as a natural point of comparison. When we actually worked out the algebra
by hand, though, we discovered something surprising:

```
1 + ratio_i  =  (row i's total)  /  |a_ii|

sum over all rows of  log(1 + ratio_i)
   =  sum of log(row totals)   −   sum of log(|a_ii|)
      \_______ does not depend on the permutation at all _______/

So minimising the left-hand side is EXACTLY THE SAME as maximising sum of log(|a_ii|),
which is exactly MC64's own objective.
```

We verified this on 193 randomly generated matrices: min-sum and MC64 produced an
identical objective value on every single one. **Min-sum is MC64 wearing a different
name.** It was not a real second option; it was the same option in disguise. This was an
important, humbling finding — it is written into the `minsum_permutation` function's
docstring in `src/solvebench/reordering.py` today so nobody re-discovers it the hard way,
and there is a permanent automated test (`test_minsum_is_mc64_under_another_name`) that
would fail immediately if this stopped being true.

So the real competition, in the end, is between **two** genuinely different objectives:
MC64 (the established one) and bottleneck (ours).

---

## 12. Milestone 6 — four scipy routines that could not be trusted

This milestone cost more real time than anything else in the project, and is worth
telling in full because of what it teaches about testing code before trusting it.

The first implementation of the bottleneck search used two ready-made functions from the
`scipy` scientific computing library that are specifically built for this kind of
matching problem: `min_weight_full_bipartite_matching` and `maximum_bipartite_matching`.
They are the "obvious" tools for the job. Four separate times, in four separate ways,
they turned out to **hang forever** on real matrices from our corpus:

1. On the matrix `nnc261`, using raw ratio values as weights, the matching routine had
   still not returned after 240 seconds.
2. After switching to a mathematically safer version of the weights (to fix problem #1),
   the *same* routine hung on a completely different, much smaller matrix, `west0067`
   (only 67 rows, 294 nonzero entries).
3. The other routine, `maximum_bipartite_matching`, took 7.2 seconds for a *single* call
   on `bcsstk19` (817 rows), and the bottleneck search needs about ten such calls per
   matrix — and it was *slowest exactly when a matching did exist*, which is the common
   case.
4. After replacing both scipy routines with a different technique (explained below), a
   memory-saving size cap meant the largest matrices were still quietly being routed
   through the same broken code, and the fourth hang appeared on `rw5151` (5,151 rows) —
   caused by the very safety limit meant to prevent problems.

**The fix, each time, was the same idea**: convert the matching problem into a **dense
assignment problem** and hand it to `scipy.optimize.linear_sum_assignment`, a completely
different (and, for our matrix sizes, extremely fast and reliable) algorithm. Where the
two original routines took over 90 seconds or simply never returned, the dense
replacement took a fraction of a second — in one measured case, **more than 1,600 times
faster**.

**A hang cannot be interrupted from inside a running program.** The only real defence is
to test every single matrix in the corpus *locally*, before ever spending Kaggle time on
it. This is exactly what `tools/preflight_reordering.py` does: it runs every objective
over every matrix in the corpus on a normal computer first, and refuses to let anything
proceed to the cloud until all 930 matrices pass without a single hang. The eventual
full preflight run completed all 930 matrices in 66.5 minutes with zero problems — proof
that the fix actually worked, rather than an assumption that it did.

---

## 13. Milestone 7 — the reordering study result

With the search made reliable, the actual experiment could finally be run: for every
matrix in the corpus, for every one of the three stationary methods (Jacobi,
Gauss-Seidel, SOR), try solving it under three conditions — the row order *as given*, the
row order chosen by *MC64*, and the row order chosen by *our full selector* (which
computes all four candidate diagonals and keeps whichever one produces the smallest
estimated spectral radius — see the pseudocode in Section 18 for exactly how).

The result was two things at once — a very large positive number, and a much smaller
number attached to our specific new idea:

```
systems solved:
  as given (row order untouched)      317
  MC64                                 514
  our full selector                    516

199 systems rescued in total.  Zero lost — the selector can never do
worse than leaving the matrix alone, because "change nothing" is always
one of its own candidates.
```

**But almost all of that gain belongs to MC64, not to our bottleneck idea.** Breaking
the 199 rescued systems down by which objective actually gets the credit:

```
rescued by BOTH MC64 and bottleneck     195
rescued by bottleneck ONLY                4
rescued by MC64 ONLY                      2
                                       -------
net gain of our objective over MC64:     +2
```

This mirrors the APK situation from Milestone 0 almost exactly: the *idea* of reordering
is genuinely powerful, but the specific *choice* of objective barely changes the outcome.
**However**, when the two objectives are compared on the thing bottleneck was actually
designed to minimise — the worst row ratio itself, not the count of systems solved —
bottleneck wins outright:

```
On 873 matrices where both objectives produce a finite ratio:
  bottleneck strictly better than MC64:   367
  MC64 strictly better than bottleneck:     0     <- never, not even once
  tied:                                    506
```

And the theoretical guarantee we built the objective around — driving the ratio below 1,
which forces convergence — turns out to be **unreachable on this corpus**: 43 matrices
were already below the threshold before any reordering, and *not one further matrix* can
be pushed under it by any row permutation. The theorem is true, and, on real data, empty.
Both of these facts — the win on the target metric, and the limit of what it buys — are
reported honestly together, and are the subject of `results/figures/03_dominance.png`.

---

## 14. Milestone 8 — the ILU hole

While writing up the reordering results, a related question came up: **does the same
"zero-on-the-diagonal" problem also break the preconditioned methods** (ILU-BiCGSTAB,
ILU-GMRES, and so on — explained in Section 17), not only the three stationary ones?

The answer was yes, and the mechanism was interesting enough to investigate directly. The
routine that builds an ILU preconditioner, `build_ilu`, tries a proper incomplete
factorization at three different settings; if all three fail, it falls back to a much
cruder approximation — simple diagonal scaling. But diagonal scaling **also** needs a
nonzero diagonal. So a zero on the diagonal closes every single door, including the
emergency one, and the entire preconditioned family becomes unusable at once on that
matrix.

We measured this directly. On the 166 matrices where no ILU preconditioner could be
built at all:

```
Gauss-Seidel                0 / 166   solved
GMRES (no preconditioner)  11 / 166   solved
BiCGSTAB (no preconditioner) 12 / 166  solved
a sparse direct solver     148 / 166   solved
```

That last number — 148 out of 166 — is the important one. It proves these are **not**
intrinsically hard systems to solve; a direct solver handles nearly all of them without
difficulty. They are systems the *iterative toolkit specifically* cannot reach, because
of one narrow, fixable cause.

Applying the same diagonal-selection idea from Milestone 9 in front of ILU:

```
ILU builds successfully, as given:         0 / 166
ILU builds successfully, after reordering: 150 / 166
and of those, actually SOLVES:              17 / 166
```

The gap between 150 (buildable) and 17 (actually solved) is reported deliberately, not
hidden: being able to *construct* a preconditioner is not the same as the system then
*converging*. This is `results/figures/02_ilu_hole.png`.

---

## 15. Milestone 9 — the relaxation factor

SOR (Successive Over-Relaxation) has one extra tuning parameter that Jacobi and
Gauss-Seidel do not: a number called **omega (ω)**, which controls how far to
"overshoot" each update. Throughout the main sweep, ω was fixed at **1.25** for every
single matrix in the corpus — a single number applied uniformly, regardless of what the
matrix actually looked like.

We measured directly whether that single fixed choice was actually a good idea, and
found it was the wrong call **in both directions at once**:

```
plain Gauss-Seidel (equivalent to omega = 1) solves 11 systems that
  SOR at the fixed omega = 1.25 does NOT solve

SOR at the fixed omega = 1.25 solves 9 systems that
  plain Gauss-Seidel does NOT solve
```

So **20 systems in the corpus have their outcome decided entirely by the choice of
omega** — that number is the absolute ceiling of what any smarter choice of omega could
possibly recover. There is a classical formula, due to Young (1950), for computing the
mathematically optimal omega from the spectral radius of the Jacobi iteration matrix:

```
omega*  =  2  /  ( 1 + sqrt( 1 − rho(T_Jacobi)² ) )
```

We estimate `rho(T_Jacobi)` cheaply, using a technique called *power iteration* (roughly
60 extra matrix-vector multiplications — a tiny cost compared to the thousands of
iterations a full solve can take), and plug it into Young's formula for every matrix
individually, instead of using one fixed number for all of them. (Young's formula is
proven exact only for a special class of matrices; on this corpus, most matrices do not
belong to that class, so applying it anyway is a *measured* choice, not an assumed one.)

```
Gauss-Seidel (omega = 1)              122 solved
SOR, fixed omega = 1.25               119 solved
SOR, omega chosen per matrix          127 solved
```

That is **8 net systems recovered, out of the 20-system ceiling identified above** — 40%
of everything that was even possible to gain this way. The full pseudocode for this
technique is in Section 18.

---

## 16. Milestone 10 — assembling the pipeline

Putting the pieces from Milestones 9, 14, and 15 together gives the final, complete
contribution of this project: a two-step **preprocessing pipeline** that runs in front of
an otherwise completely unmodified solver.

```
    A, b
      |
      v
  STEP 1 — choose which row sits on which diagonal position
           (an assignment problem; four competing objectives; keep the best)
      |
      v
  STEP 2 — for SOR specifically, choose omega from an estimate of rho(T_Jacobi)
           instead of using a fixed 1.25 for every matrix
      |
      v
  STEP 3 — hand the (possibly reordered) matrix to the ORIGINAL, unmodified
           solver: Jacobi, Gauss-Seidel, SOR, or an ILU-preconditioned Krylov method
```

Every arm of this pipeline is measured **separately** as well as **together** — because
"the pipeline helps" is not, by itself, a real result. *Which specific step* helps, and
by how much, is the real result, and it is what lets us report Section 5's headline
numbers honestly, with every claim backed by its own measurement rather than assumed by
association. The mechanism for wiring an arbitrary solver into the pipeline is the
`make_solver` function in `src/solvebench/reordering.py`, and the seven measured
combinations ("arms") live in `PIPELINE_METHODS` inside `src/solvebench/benchmark.py`.

At the time of writing, the full-corpus run measuring every pipeline arm together
(including the two-step combination on SOR, and the reordering-before-ILU combination
across the *entire* 927-matrix corpus rather than only the 166-matrix ILU hole) is
**running on Kaggle**. The reordering-only result (Milestone 13) and the ILU-hole probe
result (Milestone 14, measured on the 166 affected matrices) are both already final.
Section 23 explains exactly what is settled and what is still in progress.

---
---

# Part 3 — The Technical Reference

---

## 17. Pseudocode: the paper's methods

This section gives runnable-style pseudocode for every method the base paper studies,
plus the other twelve baseline methods we added so the comparison would have a floor and
a ceiling. All of it is implemented for real in `src/solvebench/*.py`; these are
simplified versions for explaining the idea.

### Jacobi's method

Split `A = D + L + U` (diagonal, strictly-below, strictly-above). The update uses only
values from the *previous* full iteration — nothing computed so far in the current pass
is used.

```
function JACOBI(A, b, tolerance, max_iterations):
    x <- zeros(n)
    for k in 1 .. max_iterations:
        r <- b - A @ x                  # the residual
        if norm(r) / norm(b) <= tolerance:
            return x, converged = true
        x <- x + r / diagonal(A)        # elementwise division by a_ii
    return x, converged = false
```

### Gauss-Seidel

The same idea, but each row's update can immediately use the *already-updated* values
from earlier rows in the same pass — usually converges noticeably faster than Jacobi for
the same matrix.

```
function GAUSS_SEIDEL(A, b, tolerance, max_iterations):
    x <- zeros(n)
    for k in 1 .. max_iterations:
        r <- b - A @ x
        if norm(r) / norm(b) <= tolerance:
            return x, converged = true
        solve (D + L) @ delta = r   for delta      # one triangular solve
        x <- x + delta
    return x, converged = false
```

### SOR (Successive Over-Relaxation)

Gauss-Seidel, but the update is scaled by a factor `omega` before being applied —
"overshooting" the correction on purpose, which can converge faster if `omega` is chosen
well (omega = 1 reduces exactly to plain Gauss-Seidel).

```
function SOR(A, b, omega, tolerance, max_iterations):
    x <- zeros(n)
    for k in 1 .. max_iterations:
        r <- b - A @ x
        if norm(r) / norm(b) <= tolerance:
            return x, converged = true
        solve (D + omega * L) @ delta = omega * r   for delta
        x <- x + delta
    return x, converged = false
```

### Whether any of the above will actually converge

Convergence is governed by a single number, the **spectral radius** `rho(T)` of that
method's "iteration matrix" `T` (a different `T` for Jacobi than for Gauss-Seidel):

```
if rho(T) < 1:   the error shrinks a little more every step, eventually to zero
if rho(T) >= 1:  the error does not shrink -- the method will not converge
```

`rho(T)` also controls *speed*: reaching a tolerance of `epsilon` takes roughly
`log(epsilon) / log(rho(T))` steps. A `rho` of 0.5 needs about 27 steps to reach
`1e-8`; a `rho` of 0.999 needs about 18,000.

### The three families we added so the comparison has context

**Direct methods** (Gauss elimination, Gauss-Jordan, LU, Cholesky, and the industrial
strength `spsolve`/`splu` from the SuperLU library): factor the matrix once, then solve
two easy triangular systems. Gives the exact answer in a fixed number of steps, at the
cost of possibly needing a lot of memory for a big sparse matrix.

**Krylov methods** (Conjugate Gradient, BiCGSTAB, GMRES): instead of a fixed formula,
build up a small search space from repeatedly multiplying by `A` (`b, Ab, A²b, ...`) and
pick the mathematically best possible answer inside that space at every step. Much more
powerful than the stationary methods above, at a higher cost per step.

```
function CONJUGATE_GRADIENT(A, b, tolerance, max_iterations):   # A must be
    x <- zeros(n)                                               # symmetric
    r <- b - A @ x                                               # positive
    p <- r                                                       # definite
    for k in 1 .. max_iterations:
        if norm(r) / norm(b) <= tolerance:
            return x, converged = true
        alpha <- (r . r) / (p . (A @ p))
        x <- x + alpha * p
        r_new <- r - alpha * (A @ p)
        beta <- (r_new . r_new) / (r . r)
        p <- r_new + beta * p
        r <- r_new
    return x, converged = false
```

**Preconditioned methods** (ILU only, ILU-BiCGSTAB, ILU-GMRES(30), and the
symmetry-dispatched "APK" rule from Milestone 0): build a cheap approximate factorization
`M` of `A` (an **Incomplete LU**, which deliberately throws away the small fill-in
entries a full factorization would keep, so `M` stays sparse), and solve `M⁻¹A x = M⁻¹b`
instead of `A x = b` directly with a Krylov method. Because `M⁻¹A` behaves much more like
the identity matrix than `A` does on its own, convergence is usually far faster.

```
function ILU_PRECONDITIONED_BICGSTAB(A, b, tolerance, max_iterations):
    M <- incomplete_LU_factorization(A)      # cheap, approximate, sparse
    apply_M_inverse(v) <- solve M @ w = v for w
    return BICGSTAB(A, b, tolerance, max_iterations,
                     preconditioner = apply_M_inverse)
```

---

## 18. Pseudocode: our novel contributions

### Step 1 — choosing the diagonal (the core idea)

First, the quantity every objective is judged by: for a candidate diagonal choice, the
**row ratio** of row `i` measures how "diagonally dominant" that row is —

```
function ROW_RATIO(A, i, chosen_diagonal_column_for_row_i):
    off_diagonal_sum <- sum of |A[i, j]| for every column j != chosen_diagonal_column
    diagonal_value    <- |A[i, chosen_diagonal_column]|
    return off_diagonal_sum / diagonal_value
    # a ratio below 1 for every row means the matrix is STRICTLY DIAGONALLY DOMINANT,
    # which is a classical sufficient condition guaranteeing Jacobi and Gauss-Seidel
    # both converge
```

**MC64** (the established baseline) picks the assignment of columns to rows that
maximises the product of the chosen diagonal magnitudes — equivalently, minimises the sum
of `-log|a_ii|`. This is a standard *minimum-cost bipartite assignment* problem, solvable
exactly:

```
function MC64_PERMUTATION(A):
    cost[i][j] <- -log(|A[i,j]|)   for every nonzero entry, else infinity
    assignment <- SOLVE_MIN_COST_ASSIGNMENT(cost)   # one column per row, exactly
    return assignment   # the permutation that puts A[i, assignment[i]] on the diagonal
```

**Bottleneck** (our objective) instead picks the assignment that minimises the **worst**
row ratio across the whole matrix — a *bottleneck assignment problem*, solved by binary
search over the possible threshold values, asking at each step "does a perfect matching
still exist using only entries at or below this threshold?":

```
function BOTTLENECK_PERMUTATION(A):
    candidate_thresholds <- all distinct values of ROW_RATIO across A, sorted

    function FEASIBLE(threshold):
        allowed[i][j] <- true if A[i,j] is nonzero AND its row-ratio <= threshold
        matching <- SOLVE_ASSIGNMENT(cost = 0 where allowed else 1)
        return (total cost of the matching == 0)     # every row got an allowed column

    lo, hi <- 0, length(candidate_thresholds) - 1
    best_permutation <- none
    while lo <= hi:                                   # standard binary search
        mid <- (lo + hi) / 2
        if FEASIBLE(candidate_thresholds[mid]):
            best_permutation <- the matching just found
            hi <- mid - 1          # try to do even better
        else:
            lo <- mid + 1          # this threshold is too strict, relax it
    return best_permutation
```

*(Real-world detail actually used in the code: computing MC64's own permutation first
gives a cheap upper bound on the best possible bottleneck value, which lets most of the
threshold candidates be discarded before the binary search even starts — this alone cut
a 103-second search down to a fraction of a second on the largest matrices.)*

**The selector** runs every objective and keeps whichever one is actually estimated to
work best on this specific matrix:

```
function SELECT_BEST_DIAGONAL(A):
    candidates <- { "none": leave A unchanged,
                     "mc64": MC64_PERMUTATION(A),
                     "bottleneck": BOTTLENECK_PERMUTATION(A) }
                     # ("minsum" is also computed, but Milestone 11 proved
                     #  it is mathematically identical to "mc64")

    best_choice, best_estimated_rho <- "none", infinity
    for each candidate in candidates:
        permuted_A <- apply the candidate's permutation to A
        if permuted_A still has a zero on the diagonal:
            skip this candidate -- still unusable
        estimated_rho <- cheap power-iteration estimate of rho(T_Gauss-Seidel)
        if estimated_rho < best_estimated_rho:
            best_choice, best_estimated_rho <- candidate, estimated_rho

    return the permutation belonging to best_choice
    # NOTE: "none" is always one of the candidates, so this selector can
    # never produce a worse result than doing nothing at all
```

### Step 2 — choosing omega adaptively (for SOR only)

```
function ESTIMATE_RHO_JACOBI(A, power_iterations = 60):
    v <- a random unit vector
    for step in 1 .. power_iterations:
        v <- v - (A @ v) / diagonal(A)      # one application of the Jacobi
        v <- v / norm(v)                    # iteration matrix, then re-normalise
    return norm(v)          # converges to rho(T_Jacobi) as power_iterations grows

function OPTIMAL_OMEGA(rho):
    if rho is not usable, or rho >= 1:
        return 1.0                          # falls back to plain Gauss-Seidel;
                                             # never over-relax an iteration that
                                             # is not even converging in the first place
    return 2 / (1 + sqrt(1 - rho * rho))    # Young's 1950 formula

function SOR_ADAPTIVE(A, b, tolerance, max_iterations):
    rho    <- ESTIMATE_RHO_JACOBI(A)
    omega  <- OPTIMAL_OMEGA(rho)
    return SOR(A, b, omega, tolerance, max_iterations)
```

### The full pipeline, end to end

```
function SOLVE_WITH_PIPELINE(A, b, base_solver):
    permutation <- SELECT_BEST_DIAGONAL(A)
    A_new       <- apply permutation to the rows of A
    b_new       <- apply the same permutation to b

    if base_solver is SOR:
        return SOR_ADAPTIVE(A_new, b_new, tolerance, max_iterations)
    else:
        return base_solver(A_new, b_new, tolerance, max_iterations)

    # x comes back solving A_new x = b_new, which is exactly A x = b
    # under a relabelling of the rows -- no "undoing" of the permutation
    # is ever needed
```

---

## 19. The dataset, explained with real examples

### What SuiteSparse is

The **SuiteSparse Matrix Collection** is a large, free, public library of sparse
matrices, each one contributed by a real engineer or scientist from an actual project —
not generated randomly. Every matrix is tagged with metadata: how many rows and columns
it has, how many nonzero entries, and which real-world **domain** it comes from
(structural engineering, circuit design, and so on).

### Our selection rule

We took **every square, real-valued matrix in the collection with at most 10,000
rows/columns**. The size cap of 10,000 is a memory limit: one of our analyses (computing
the *exact* spectral radius) needs to build a full, dense `n × n` array, and at
`n = 10,000` that array alone is already 800 megabytes.

### Corpus bookkeeping — the numbers you will see everywhere

```
930 files were originally downloaded
  - 3 of them are EXACT duplicates of other files under a different name
    (bcsstk07 == bcsstk06, bcsstk12 == bcsstk11, t2dal_a == t2dal)
    -> 927 unique matrices
  - 6 files turned out to be TRUNCATED downloads: the file's own header
    declared far more nonzero entries than the file actually contained,
    so they silently failed to load at all
    (cavity20, nemeth17, nemeth22, nemeth23, nemeth25, psmigr_2)
    -> the actually usable corpus is 924 matrices
  - 1 domain, "computational fluid dynamics", was spelled two slightly
    different ways in the source data, splitting it into two domains by
    accident; we merge these back into one
```

100 of the 930 files are **not from SuiteSparse at all** — they are random matrices we
generated ourselves (sizes 5 to 300), specifically so we could reproduce the base paper's
own validation method and compare our results to it directly (see Milestone 5's headline
result for what came out of that comparison).

### The 26 domains, with how many matrices each contributes

```
  200  circuit_simulation_problem
  121  computational_fluid_dynamics_problem
  100  random_matrices                        (our own, see above)
   87  structural_problem
   78  optimal_control_problem
   49  2d_3d_problem
   44  chemical_process_simulation_problem
   39  optimization_problem
   37  undirected_weighted_graph
   27  theoretical_quantum_chemistry_problem
   25  electromagnetics_problem
   19  eigenvalue_model_reduction_problem
   17  directed_weighted_graph
   17  power_network_problem
   16  economic_problem
   13  model_reduction_problem
    9  materials_problem
    6  statistical_mathematical_problem
    4  counter_example_problem
    4  semiconductor_device_problem
    2  acoustics_problem
    2  thermal_problem
    1  computational_chemistry_problem
    1  computer_graphics_vision_problem
    1  linear_programming_problem
    1  robotics_problem
```

Notice how uneven this is — one domain has 200 matrices, eight domains have fewer than
5. Any claim about "how well a method does in domain X" needs to say how many matrices
that domain actually has, or the claim is not trustworthy. This is exactly why every
per-domain table and figure in this project always prints its own denominator alongside
the rate.

### Real matrices, by name, so you can point at specific examples

| domain | example matrix | size (n) | nonzero entries |
|---|---|---|---|
| circuit simulation | `meg1` | 2,904 | 58,142 |
| circuit simulation | `circuit_1` | 2,624 | 35,823 |
| structural engineering | `Kuu` | 7,102 | 173,651 |
| structural engineering | `raefsky5` | 6,316 | 168,658 |
| fluid dynamics | `ex40` | 7,740 | 458,012 |
| fluid dynamics | `GT01R` | 7,980 | 430,909 |
| power networks | `TSC_OPF_1047` | 8,140 | 1,012,521 |
| power networks | `TSC_OPF_300` | 9,774 | 415,289 |
| quantum chemistry | `nemeth26` | 9,506 | 760,633 |
| economic modelling | `psmigr_3` | 3,140 | 543,162 |
| electromagnetics | `fp` | 7,548 | 848,553 |
| our own random matrices | `rand_100_n300` | 300 | 90,000 |

### Specific matrices that drove specific discoveries in this project

These are worth knowing by name, because each one is a concrete "receipt" for a claim
made earlier in this document:

* **`cdde6`** (fluid dynamics domain, n = 961) — its spectral radius for Jacobi is
  `0.717`, comfortably under 1, meaning convergence is mathematically guaranteed. But its
  residual actually *grows* to **44,764 times** its starting value before finally
  shrinking and converging at iteration 178. This is why the code's "has this diverged
  yet?" safety check has to be lenient — an over-eager check would kill a run that was
  always going to succeed. (See Milestone 7 in Section 22.)
* **`mcca`, `odepa400`** (2D/3D domain) — two of the only four matrices in the entire
  corpus where the bottleneck objective genuinely outperforms MC64 on whether the system
  gets solved at all (Milestone 13).
* **`nnc261`, `west0067`, `bcsstk19`, `rw5151`** — the four matrices, from four different
  domains, that each separately exposed a hang in an off-the-shelf scipy routine
  (Milestone 12). Every one of these is now a permanent, fast test case.
* **`d_ss`** (chemical process domain, n = 53) — a rare and important counter-example:
  applying the bottleneck objective here actually makes the spectral radius *worse*
  (from 1.884 to 29.605), because protecting the single worst row can sometimes come at
  the expense of every other row. This is reported honestly as a known limitation, not
  hidden.
* **`iprob`** (linear programming domain, n = 3,001) — one of the four matrices rescued
  specifically by the bottleneck objective and not by MC64.

---

## 20. Every code file, and what it produces

This lists only the current, working code — anything superseded or unused has already
been removed from the repository (see Milestone 1 and Section 22 for what was cleaned up
and why).

### `src/solvebench/` — the library (the single source of truth)

| file | job | key things it defines |
|---|---|---|
| `config.py` | every tolerance, cap, and constant, in one place | `MAX_ITERATIONS = 10,000`, `TOLERANCE = 1e-8`, `SOR_OMEGA = 1.25`, `POWER_ITERS_OMEGA = 60` |
| `metrics.py` | the single place that decides whether a solve counted as a success | the 8 status categories, `score()` |
| `io_utils.py` | loading a `.mtx` matrix file and building a known-answer `(x_true, b)` pair | `load_matrix`, `make_ground_truth`, `structural_singularity` |
| `direct_solvers.py` | the four hand-written direct methods | `gauss_elimination`, `gauss_jordan`, `lu_solve`, `cholesky_solve` |
| `iterative_solvers.py` | Jacobi, Gauss-Seidel, SOR, CG, BiCGSTAB, GMRES, ILU construction, and the adaptive-omega pieces | `jacobi`, `gauss_seidel`, `sor`, `sor_adaptive`, `estimate_rho_jacobi`, `optimal_omega`, `build_ilu` |
| `reference_solvers.py` | the library-grade baselines an engineer would actually reach for | `sparse_spsolve`, `sparse_lu`, `ilu_bicgstab`, `ilu_gmres`, `ilu_krylov_dispatched` (this is "APK", Milestone 0) |
| `refinement.py` | iterative refinement as a factor available to *every* solver, not one method's private trick | `refine` |
| `reordering.py` | **our contribution** — the diagonal-selection objectives, the selector, the pipeline wrapper | `row_ratios`, `worst_row_ratio`, `mc64_permutation`, `minsum_permutation`, `bottleneck_permutation`, `select_diagonal`, `make_solver` |
| `spectral.py` | the spectral radius and the classical convergence theory | `spectral_radius`, `convergence_verdict`, `classify` |
| `benchmark.py` | the actual harness that runs every method on every matrix | `METHODS` (the 16 baselines), `PIPELINE_METHODS` (the 7 pipeline arms), `run_one_matrix` |
| `corpus.py` | corrections applied when reading results, not when generating them | merges the split domain, drops the 3 exact duplicates |

### `tools/` — scripts you actually run from the command line

| file | job | what it produces |
|---|---|---|
| `build_notebooks.py` | takes the library above and bakes it into ready-to-run Kaggle notebooks | the `.ipynb` files under `notebooks/` |
| `test_notebooks.py` | runs every generated notebook locally, on a handful of matrices, before anything is trusted on Kaggle | pass/fail — nothing is ever pushed to Kaggle without this passing |
| `make_figures.py` | draws every chart from the result tables | every `.png` under `results/figures/` |
| `download_corpus.py` | fetches matrices from SuiteSparse, verifying each one against its own declared size before keeping it | populates a local matrix folder |
| `preflight_reordering.py` | runs every diagonal-selection objective over the *entire* corpus locally, specifically to catch a hang before it costs cloud time (Milestone 12) | a pass/fail report, no hangs allowed |
| `probe_ilu_reorder.py` | the small, fast experiment that discovered the ILU-hole result (Milestone 14) | `results/tables/ilu_reorder_probe.csv` |
| `probe_adaptive_sor.py` | the small, fast experiment that discovered the adaptive-omega result (Milestone 15) | `results/tables/adaptive_sor_probe.csv` |
| `rhs_cancellation.py` | a diagnostic tool, unrelated to the main results, measuring floating-point cancellation | diagnostic output only |

### `notebooks/` — what actually runs on Kaggle

Each of these is **generated automatically** from the library above by
`build_notebooks.py`. Nobody should ever hand-edit a notebook file directly — that is
literally how the library and the notebook drifted apart from each other once already,
early in the project, and it is why they are generated rather than written by hand today.

| notebook | what it runs |
|---|---|
| `solvebench-main-sweep` | all 927 matrices through all 16 baseline methods |
| `solvebench-spectral` | the spectral radius and matrix-class labels for every matrix (no solving) |
| `solvebench-refinement-study` | every method, given 0, 1, or 2 rounds of refinement |
| `solvebench-reordering-study` | the 3-condition, 3-method reordering experiment (Milestone 13) |
| `solvebench-pipeline` | the 7 pipeline arms, all together (Milestone 16) |

### Documents

| file | what it is for |
|---|---|
| `README.md` | the project's front door: the headline result, then how to run everything |
| `SCOPE.md` | precisely what this project claims, and what it deliberately does not claim |
| `RESULTS.md` | every measured number, each one traceable to the exact table it came from |
| `docs/GUIDE.md` | a shorter, more code-and-figure-focused companion to this document |
| `docs/PROJECT_STATE.md` | a running log of where the project stands and what is still open |
| **this file** | the full narrative, for teaching a new teammate from zero |

---

## 21. Every output table and figure

### Tables (all under `results/tables/`)

| table | rows | what is in it |
|---|---|---|
| `benchmark_results.csv` | 14,880 | one row per (matrix, method, refinement level) — the core result set |
| `method_summary.csv` | 16 | applicability and conditional success, per method |
| `per_domain.csv` | — | the same, broken down by domain, with the denominator always included |
| `spectral.csv` | 930 | the spectral radius and matrix-class flags for every matrix file |
| `reordering_study.csv` | 8,170 | the 3-condition reordering experiment, plus each objective's own ratio |
| `ilu_reorder_probe.csv` | 166 | the ILU-hole experiment |
| `adaptive_sor_probe.csv` | 400 | the adaptive-omega experiment |

### Figures (all under `results/figures/`, numbered in the order this document tells the story)

**Our contribution:**

1. **`01_pipeline_scoreboard.png`** — the headline chart. Two bars per stationary method
   — as given, and with the pipeline — with the percentage gain printed above each pair.
2. **`02_ilu_hole.png`** — two panels: which methods can solve the 166 ILU-hole matrices
   today (left), and what reordering recovers (right).
3. **`03_dominance.png`** — a scatter plot, one dot per matrix, comparing MC64's row
   ratio against bottleneck's. Every dot sits on or below the diagonal line — bottleneck
   never loses on the quantity it was built to optimise.

**The overall benchmark:**

4. `04_method_scoreboard.png` — applicability against conditional success, all 16 methods.
5. `05_performance_profile.png` — the standard solver-comparison chart (a "performance
   profile"): how often each method is the outright fastest, and how much of the corpus
   it can solve at all.
6. `06_outcomes.png` — the full breakdown across all 8 outcome categories.
7. `07_domain_heatmap.png` — method against domain.
8. `08_cost_profile.png` — cost measured in matrix-vector multiplications, not seconds.
9. `09_runtime_scaling.png` — how running time grows with matrix size.
10. `10_iteration_counts.png` — how many iterations each method needs, when it succeeds.

**The underlying theory:**

11. `11_spectral_radius.png` — the distribution of the spectral radius across the corpus.
12. `12_hypothesis_coverage.png` — how many matrices fall into each classical
    convergence-guarantee category.
13. `13_prediction_vs_observation.png` — does the textbook theory predict the outcome?
14. `14_jacobi_vs_gauss_seidel.png` — the base paper's own question, answered on real data.

**Supporting studies:**

15. `15_refinement_effect.png` — the effect of giving every method the same extra
    refinement step.
16. `16_conditioning.png` — accuracy, split by whether the matrix is numerically
    ill-conditioned.
17. `17_dispatch_ablation.png` — the APK dispatch rule from Milestone 0, measured
    honestly against always using BiCGSTAB.

---

## 22. Mistakes we made, and what changed when we fixed them

Every one of these was present in an earlier version of the project and changed a
headline number once it was fixed. They are listed here in full so nobody re-introduces
any of them.

**1. Trusting a solver's own claim of success.** The very first version of the harness
marked a solve "successful" whenever the underlying function call did not raise an
error. This let through 57 "successful" solves whose actual relative residual reached
**9.71 × 10¹²** — wrong by twelve orders of magnitude. **Fix:** success is now always
computed independently, from the measured residual, in one single place
(`metrics.score`), for every method identically.

**2. Dividing success rates by the whole corpus.** Conjugate Gradient is mathematically
only valid on symmetric positive-definite matrices — about 11.7% of this corpus. Dividing
its successes by the *whole* corpus reported it succeeding only 10.4% of the time, making
it look nearly useless. Its actual success rate, on the matrices it is even allowed to
run on, is 84.3%. **Fix:** applicability and conditional success are now always kept as
two completely separate numbers, and are never multiplied together.

**3. Giving refinement to only one method.** Iterative refinement is a generic technique
that polishes *any* solver's answer with one extra cheap step. The first version of the
project applied it to exactly one method, which is precisely why that method appeared to
be the most accurate — the advantage being measured was the refinement step, not
anything intrinsic to the method. **Fix:** refinement is now a separate, orthogonal
experiment, applied identically to every method, in its own dedicated study.

**4. Re-factoring inside the iteration loop.** The original Gauss-Seidel and SOR
implementations were re-analysing their triangular system on *every single iteration* —
up to 10,000 times per matrix — instead of doing it once up front. This inflated their
measured running time by a factor of 11 to 14. **Fix:** the triangular factor is now
built exactly once per solve ("delta form"), which is also what the pseudocode in
Section 17 shows.

**5. A divergence check that was too aggressive.** The old safety check aborted a run as
soon as its residual grew past 10,000 times its best value so far — intended to save time
on genuinely divergent runs. But the spectral radius only describes the *long-run*
average behaviour of an iteration; a matrix can have a perfectly convergent spectral
radius and still see its residual balloon enormously in the short term before it starts
shrinking. The matrix `cdde6` (Section 19) is the concrete proof: it grows to 44,764
times its starting residual and *still* goes on to converge. The old, tighter check would
have killed this run on the way up, wrongly recording a genuine convergence as a failure.
**Fix:** the growth threshold was raised to 1,000,000,000,000× (1e12), which the evidence
shows is high enough to never kill a real convergence while still catching genuinely
divergent runs quickly.

**6. Two copies of the same library, quietly drifting apart.** Every generated Kaggle
notebook embeds its own copy of the entire library so it can run standalone in the cloud.
When that embedded copy was also accidentally being tracked in the git repository, and a
plain `python` command from the project's root folder would pick up *that* copy instead
of the real one (because the current folder is searched before anything else) — the two
copies quietly disagreed with each other about a size limit while producing supposedly
comparable results. **Fix:** notebooks are strictly generated, never hand-edited, and the
accidental copy is now excluded from the repository entirely.

**7. Downloaded files that were never verified.** Six matrix files, downloaded once and
used for months, turned out to be truncated: each file's own header declared far more
nonzero entries than the file physically contained, and this was not discovered until it
silently broke a benchmark run much later. **Fix:** every newly downloaded matrix is now
checked against its own declared size the moment it arrives, before it is trusted.

**8. Trusting a library routine without measuring it first.** As told in full in
Milestone 12, two ready-made `scipy` matching routines hung — sometimes for tens of
seconds, once for over 240 seconds, and twice indefinitely — on real matrices from this
corpus, costing an entire 12-hour cloud session before the cause was tracked down. **Fix:**
every new routine that touches the whole corpus is now tested locally, on *every* matrix,
before it is trusted with cloud time — this is exactly what `preflight_reordering.py`
exists to do.

---

## 23. Where things stand right now

**Fully measured and final:**
* The 16-method main sweep, across all 927 matrices.
* The spectral analysis (theory vs. reality).
* The refinement study.
* The reordering study — reordering alone, on the three stationary methods
  (Milestone 13).
* The ILU-hole diagnosis, on the 166 matrices where it applies directly (Milestone 14).
* The adaptive-omega probe, on a 400-matrix sample (Milestone 15).

**Currently running:** the full pipeline sweep — all seven pipeline arms (Milestone 16),
across the entire corpus, in one run. This will confirm the reordering numbers on a fresh
run, and for the first time measure, across the *whole* corpus rather than only a
sub-sample:
* what the two-step pipeline (reordering **and** adaptive omega together) achieves on
  SOR, which is currently unknown, and
* what reordering-before-ILU achieves across the entire corpus, rather than only the 166
  matrices where ILU was already known to be completely broken.

**Deliberately not done, and why:** downloading the rest of SuiteSparse (a further 353
qualifying matrices) has been carried out into a *separate* folder that is intentionally
kept apart from the 927-matrix corpus every result in this project was measured on —
mixing them in would silently invalidate every number above at once. Whether to re-run
everything on the larger, more complete corpus is a genuine future decision, not an
oversight.

**What is still needed:** the written report and the live presentation — this document,
together with `README.md`, `SCOPE.md`, and `RESULTS.md`, is the raw material for both.
