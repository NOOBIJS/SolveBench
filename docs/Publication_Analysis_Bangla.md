# SolveBench: Publication Strategy — সম্পূর্ণ বিশ্লেষণ

> এই ডকুমেন্টটি ২৭টি expert agent-এর গবেষণা, ৪টি adversarial peer reviewer-এর আক্রমণ, এবং
> venue deadline-গুলোর web-verification থেকে তৈরি। সবচেয়ে গুরুতর দাবিগুলো আমি নিজে
> **আপনাদের নিজের data-র উপর যাচাই করেছি** — নিচে প্রতিটার verification আলাদা করে দেখানো আছে।
>
> তারিখ: ৯ সেপ্টেম্বর ২০২৬

---

## ১. মূল কথা (Bottom Line)

**Method paper মারা গেছে। Empirical paper বেঁচে আছে — এবং তার deadline মাত্র ১১ দিন দূরে।**

তিনজন স্বাধীন adversarial reviewer APK-কে refute করেছে, এবং সেটা কথার কথা না — **আপনাদের নিজের
data-র measurement দিয়ে**। কিন্তু এর নিচে একটা সত্যিকারের publishable contribution টিকে আছে।

দুইটা জিনিস আলাদা করে বোঝা জরুরি:

| | অবস্থা |
|---|---|
| APK একটা নতুন method — এই দাবি | ❌ **সম্পূর্ণ বাতিল** |
| "924 matrices-এ zero only-Jacobi" — এই headline | ❌ **টেকে না** (নিচে proof) |
| Hypothesis-coverage census + domain viability map | ✅ **প্রকাশযোগ্য, কেউ আগে করেনি** |

**Realistic tier:** ICCIT 2026 (IEEE-indexed, ১১ দিন বাকি) + arXiv এখনই; তারপর ৩-৪ মাসে
mid-tier journal (Numerical Algorithms বা JCAM)। **SISC না, NLAA না, IPDPS না** — এই data দিয়ে কখনোই না।

---

## ২. যে সমস্যাগুলো আমি নিজে verify করেছি

এগুলো reviewer-এর দাবি হিসেবে না — আমি `results_final/.../03_benchmark_results.csv`,
`dataset_large/`-এর ৯২৪টা matrix, আর notebook generator-এর source code নিজে চালিয়ে
বের করেছি। প্রতিটার actual output নিচে দেওয়া।

> **কীভাবে পড়বেন:** এখানকার প্রতিটা সমস্যার **সমাধান § ৭-এ আছে**, একই ক্রমে।
> § ২ = "কী ভাঙা", § ৭ = "কীভাবে ঠিক করবেন"।

### ২.১ আপনাদের denominator ভুল — এবং অল্প ভুল না  
*→ সমাধান: § ৭, Gap ২*

```
Jacobi / Gauss-Seidel: 924-এর মধ্যে 491টা matrix = not_applicable (diagonal-এ zero আছে)
দুইটা method-ই applicable এমন matrix: 433
```

Headline-এ যে **"924 matrices"** বলা হয়েছে সেটা **২.১ গুণ overstatement**। আসল comparison
দাঁড়িয়ে আছে ৪৩৩টা system-এর উপর। Zero diagonal থাকলে Jacobi/Gauss-Seidel-এর update rule-ই
undefined — ওগুলো কখনো চালানোই হয়নি, কিন্তু denominator-এ গুনে ফেলা হয়েছে।

### ২.২ Flagship "ZERO only-Jacobi" সঠিক success test-এ টেকে না ⚠️  
*→ সমাধান: § ৭, Gap ১*

এটাই সবচেয়ে গুরুতর finding। আমি দুইভাবে হিসাব করেছি:

```
শুধু residual test (rel_residual ≤ 1e-8):        75 both | 46 only-GS | 0 only-Jacobi
+ forward-error gate (rel_error ≤ 1e-6):         57 both | 17 only-GS | 1 only-Jacobi  ← fs_680_1
```

মানে: উত্তরটা **আসলেই সঠিক** কিনা সেটাও যদি চেক করেন (শুধু residual ছোট কিনা না), তাহলে
zero টা **one** হয়ে যায়। আপনাদের পুরো paper-এর flagship claim **threshold-sensitive** —
নিজের data-তেই টেকে না।

### ২.৩ Success criterion কখনো চেক করেনি উত্তর ঠিক কিনা  
*→ সমাধান: § ৭, Gap ১*

```
Gauss-Jordan: 596টা "ok" row-এর মধ্যে 57টার relative residual > 1e-8 — সর্বোচ্চ 9.71e+12
```

Direct method-গুলোকে `ok` মার্ক করা হয়েছে **শুধু exception raise না করার কারণে**, accuracy
যাচাই ছাড়াই। residual 10^12 মানে উত্তরটা সম্পূর্ণ আবর্জনা, অথচ "সফল" হিসেবে গণনা হয়েছে।
তাই **97.9% direct-method success rate টা defensible না**।

### ২.৪ Success rate ভুল denominator দিয়ে ভাগ করা  
*→ সমাধান: § ৭, Gap ২*

| Method | আপনারা যা report করেছেন | আসলে (applicable-এর মধ্যে) |
|---|---|---|
| Conjugate Gradient | 10.4% | **83.9%** (94/112) |
| Cholesky | 10.1% | **100%** (62/62) |
| Gauss-Seidel | 13.4% | **29.2%** (121/414) |

লক্ষ্য করুন — এটা আসলে **আপনাদের নিজের baseline-গুলোকে ছোট করে দেখাচ্ছে**, যা কৃত্রিমভাবে
APK-এর সুবিধা বাড়িয়ে দেখায়। CG আসলে যেসব matrix-এ প্রযোজ্য সেখানে 83.9% সফল, 10.4% না।

### ২.৫ Refinement শুধু APK পেয়েছে, বাকি ৯টা method পায়নি  
*→ সমাধান: § ৭, Gap ৬*

Notebook generator-এ `refinement_passes` config-টা **একমাত্র APK-র function-এ** ব্যবহৃত
হয়েছে (line 640-650)। Jacobi, Gauss-Seidel, SOR, CG, BiCGSTAB — কেউ একটাও refinement pass
পায়নি, direct method-গুলোও না।

এটা confound, কারণ আগেই মাপা হয়েছে: **একই refinement wrapper Gauss-Seidel-এ লাগালে
GS পায় 5.44e-16, আর refined APK পায় 1.45e-14** — অর্থাৎ GS প্রায় ২৭ গুণ বেশি accurate।
তাই "APK সবচেয়ে accurate" — এই দাবিটা আসলে refinement-এর দাবি, method-এর না।

দ্বিতীয় অসমতা: **iteration-এর একক এক না**। APK-র এক iteration = ১টা matvec + ১টা ILU
triangular solve; Jacobi-র এক iteration = ১টা matvec। iteration count দিয়ে তুলনা করা মানে
APK-কে প্রতি iteration-এ একটা ফ্রি preconditioner apply উপহার দেওয়া।

### ২.৬ ১৬০টা system কার্যত singular — error statistics দূষিত  
*→ সমাধান: § ৭, Gap ৭*

`03_benchmark_results.csv`-এর condition number column থেকে ৯৩০টা matrix-এর হিসাব:

```
cond > 1e10  : 405 matrices
cond > 1e12  : 290
cond > 1e15  : 160      ← double precision-এ কার্যত singular
condition = inf : 22
```

double precision-এ কাজের precision ~1e16। `cond > 1e15` মানে forward error-এর কোনো
অর্থপূর্ণ bound-ই নেই। এই ১৬০টাকে বাকিদের সাথে মিশিয়ে median error বের করলে সংখ্যাটা
method-এর গুণ না, dataset-এর ill-conditioning মাপে।

আরো খারাপ — আমি প্রতিটা matrix-এ `‖A·1‖ / ‖|A|·1‖` (cancellation ratio) মেপেছি।
৬টা matrix-এ RHS প্রায় সম্পূর্ণ cancellation থেকে তৈরি:

| Matrix | ratio |
|---|---|
| t2dal_bci | **3.52e-17** |
| problem1 | 5.83e-17 |
| crystk01 | 1.75e-15 |
| t2dal | 6.36e-15 |
| t2dal_a | 6.36e-15 |
| nos7 | 4.31e-09 |

এগুলোতে `b = A·1` মূলত **rounding noise** — b-র প্রতিটা অঙ্ক cancellation-এ হারিয়ে গেছে।
এখানে কোনো solver-এর accuracy মাপা অর্থহীন।

### ২.৭ প্রতিটা runtime number অব্যবহারযোগ্য  
*→ সমাধান: § ৭, Gap ৮*

তিনটা আলাদা কারণ, প্রতিটাই code-এ যাচাই করা:

1. **GS/SOR প্রতি iteration-এ L আবার factor করে।** Code-এ
   `spla.spsolve_triangular(L, b - U @ x, lower=True)` loop-এর **ভেতরে**। L একবারও
   pre-factor করা হয়নি। ১০,০০০ iteration মানে ১০,০০০ বার triangular setup।
2. **Stopping test-এর খরচ অসমান।** GS/SOR প্রতি iteration-এ `_rel_residual(A, x_new, b)`
   ডাকে — একটা বাড়তি matvec। scipy-র `cg`/`bicgstab` recursive residual ব্যবহার করে,
   এই খরচটা দেয় না।
3. **প্রতিটা measurement একবার চালানো** — repetition নেই, warm-up discard নেই,
   `OMP_NUM_THREADS` pin করা নেই, তাই Kaggle-এর thread contention সরাসরি number-এ ঢুকেছে।

মানে Jacobi/GS/SOR "ধীর" দেখাচ্ছে, কিন্তু সেটা **algorithm-এর কারণে না, implementation
আর measurement protocol-এর কারণে**। এই ভিত্তিতে speed ranking কোনো referee মানবে না।

### ২.৮ Release করা source code paper-এর headline method-ই ধারণ করে না  
*→ সমাধান: § ৭, Gap ১০*

`src/solvebench/iterative_solvers.py`-এ মাত্র ৪টা solver আছে:
`jacobi`, `gauss_seidel`, `sor`, `conjugate_gradient`।

```
src/solvebench/ ফোল্ডারে grep: bicgstab | apk | spilu | refine  →  0 match
src/solvebench/benchmark.py:38  DIRECT_METHOD_SIZE_CAP = 3000
notebook-এ                       direct_size_cap        = 2000
```

অর্থাৎ যে repository আপনারা paper-এর সাথে release করবেন, সেখানে **BiCGSTAB নেই, APK নেই,
ILU নেই, refinement নেই**, আর direct method-এর cap-ও আলাদা। কেউ এই code চালিয়ে paper-এর
একটা সংখ্যাও reproduce করতে পারবে না। আজকাল প্রায় প্রতিটা numerical venue artifact
check করে — এটা ধরা পড়বেই।

### ২.৯ একটা reviewer claim যা আমি যাচাই করে ভুল পেয়েছি

একজন reviewer বলেছিল "APK, Jacobi-র চেয়ে median-এ 2.65x ধীর"। আমার যাচাই:

```
common matrices: 75
median: APK 0.0038s vs Jacobi 0.0086s  →  APK আসলে দ্রুততর
কিন্তু: 75টার মধ্যে 43টাতে (57%) APK ধীর
```

তাই speed-এর গল্পটা **মিশ্র, সরাসরি ভুল না**। আমি এটা আলাদা করে বলছি কারণ যে সংখ্যা আমি
নিজে reproduce করতে পারিনি, সেটা আপনাদের কাছে চালিয়ে দেব না।

---

## ৩. কেন APK-কে novel দাবি করা যাবে না

এটা মেনে নেওয়া কঠিন, কিন্তু প্রমাণগুলো খুব শক্ত:

1. **আপনাদের dispatch rule একটা ১৯৯৪ সালের SIAM বই।**
   Barrett et al., *Templates for the Solution of Linear Systems* (SIAM, 1994) — এই বইয়ে
   একটা decision flowchart আছে যার branch হলো "A কি symmetric? positive definite?" →
   CG / BiCGSTAB / GMRES। **ওটাই APK**, ৩০ বছর আগে প্রকাশিত।

2. **এটা প্রতিটা major library-র default behaviour।**
   PETSc-র KSP, Trilinos Stratimikos, MATLAB-এর `backslash` dispatch tree — সবাই runtime-এ
   structure দেখে solver বাছাই করে।

3. **১২ লাইনের scipy script আপনাদের headline number reproduce করে।**
   আর `scipy.sparse.linalg.spsolve` APK-কে হারায় — 64/70 vs 48/70, এবং ৪-৬ গুণ দ্রুত।
   এটা **আপনাদেরই data-র উপর মাপা**।

4. **Dispatch rule আক্ষরিক অর্থে কিছুই যোগ করে না।**
   symmetry-dispatch 51/70 vs always-BiCGSTAB 51/70 vs GMRES(30) 52/70। অর্থাৎ যে জিনিসটাকে
   আপনাদের একমাত্র novelty বলা হচ্ছে, সেটার contribution **শূন্য**।

5. **Auto-selection একটা ২০ বছরের subfield, যা আপনারা cite করেননি।**
   SALSA (Bhowmick et al., 2006), Lighthouse, SolverSet (2026 — ১২,৬৫১ matrix)।
   APK একটামাত্র boolean (symmetric কিনা) দেখে বাছাই করে; এই field-এর floor-এর নিচে।

6. **একটা technical error যা referee সাথে সাথে ধরবে।**
   SuperLU-র `spilu` column permutation ও partial pivoting ব্যবহার করে, তাই factor-টা
   symmetric থাকে না। সেটা PCG-তে দিলে **CG-র convergence theory ভেঙে যায়**।

**আর accuracy-র জয়টা?** ওটা এসেছে একটা refinement pass থেকে **যেটা কোনো baseline পায়নি**।
একই wrapper Gauss-Seidel-এ লাগালে সেটা 5.44e-16 পায় — refined APK (1.45e-14)-এর চেয়ে
প্রায় **২৭ গুণ বেশি accurate**।

---

## ৪. যা টিকে আছে — এবং এটাই আসল paper

আপনাদের "0 only-Jacobi" ফলাফলটা **১৯৪৮ ও ১৯৫৮ সালের দুইটা theorem দিয়েই predicted**:

- **Stein–Rosenberg theorem (1948)** — Jacobi matrix elementwise nonnegative হলে
  (অর্থাৎ প্রতিটা L-matrix / M-matrix-এর জন্য) "Jacobi converge করে কিন্তু GS করে না" —
  এই কেসটা **গাণিতিকভাবে অসম্ভব**।
- **Householder–John theorem (1958)** — A যদি SPD হয়, GS সবসময় converge করে।
- **Young (1950)** — consistently ordered matrix-এ rho(GS) = rho(J)²।

আপনাদের `Convergence_Theory_Deep_Dive.md`-এ grep করে দেখা গেছে: **Stein, Rosenberg, Varga,
H-matrix, M-matrix, property A — একটাও নেই।** যেকোনো numerical venue-র referee প্রথম
লাইনেই বলবে "Varga (1962), Theorem 3.3 দেখুন" — এবং paper আর দাঁড়াতে পারবে না।

**কিন্তু এই আক্রমণটাকেই paper-এর কেন্দ্রীয় measurement বানানো যায়:**

> **বাস্তব matrix-গুলো ঠিক সেই hypothesis class-এ ঘনীভূত যেখানে এই theorem-গুলো base
> paper-এর phenomenon-কে নিষিদ্ধ করে; আর uniform-random ensemble সেই hypothesis-গুলোর
> একটাও মানে না। একই population structure বাস্তব system-এর 81%-এ classical iteration-কে
> অকার্যকর করে দেয়।**

এই census কেউ প্রকাশ করেনি। আপনাদের domain-stratified viability map-ও কেউ করেনি:
structural 76.7%, CFD 16.0%, circuit simulation 1.0%, optimization 0%।

### প্রস্তাবিত Title

> **Real Matrices Are Not Random Matrices: A Hypothesis-Coverage Audit of Classical
> Iterative Solvers on 905 Sparse Systems**

### প্রস্তাবিত Abstract (draft)

> Comparative studies of stationary iterative methods are routinely validated on dense
> uniform-random matrices. We ask whether their conclusions transfer to matrices
> practitioners actually solve. We benchmark ten solvers on 905 SuiteSparse systems
> spanning 26 real application domains (n = 5 to 10,000), plus 100 matrices generated by
> the random scheme of a recent Jacobi/Gauss-Seidel comparison, under a single externally
> verified success criterion applied identically to every solver.
>
> Jacobi and Gauss-Seidel are undefined on 491 systems (zero diagonal entry). On the 414
> remaining we observe 75 systems where both converge, 46 where only Gauss-Seidel
> converges, and zero to one where only Jacobi converges — the count is threshold-sensitive
> — against 1,095 only-Jacobi cases reported at n ≤ 5 on random matrices. We show this is
> hypothesis coverage, not new evidence: 64 of the 75 are H-matrices, and among systems
> with exactly computed spectra 59 of 60 Jacobi-convergent cases satisfy Stein–Rosenberg or
> Young hypotheses; all 100 random matrices have rho(T_J) ≥ 1.26.
>
> Only 170 of 905 systems (18.8%) yield to any classical iterative method, with viability
> set by domain: structural 76.7%, CFD 16.0%, circuit simulation 1.0%, optimization 0%. At
> this scale sparse direct factorization outperforms our best ILU-preconditioned Krylov
> configuration, and symmetry-based solver dispatch adds nothing measurable. Harness and
> all 9,246 measurements are released.

**কেন এই framing?** কারণ প্রতিটা আক্রমণ এখানে একটা measurement-এ রূপান্তরিত হয়:
- "তোমার zero টা ১৯৪৮-এর theorem" → এটাই paper-এর central measurement
- "তুমি 491টা matrix বাদ দিয়েছ" → এটা reported applicability statistic
- "APK ১৯৯৪-এর technology" → এটা honest null result, যা auto-selection literature-এর সাথে কথা বলে

**যে paper-এর thesis-ই হলো "আমাদের নিজের headline অনিবার্য ছিল", সেটাকে referee ভাঙতে পারে না।**

---

## ৫. Venue Ladder (deadline গুলো web-verified)

| # | Venue | Deadline | সম্ভাবনা |
|---|---|---|---|
| ১ | **arXiv (cs.NA) + Zenodo DOI** | নেই — ~২০ সেপ্টে post করুন | 100% |
| ২ | **ICCIT 2026** (IEEE, Cox's Bazar, ১৮-২০ ডিসে) | **২০ সেপ্টেম্বর ২০২৬ — VERIFIED, ১১ দিন বাকি** | reframe করলে 60-70% · বর্তমান APK framing-এ 25-35% |
| ৩ | HiPC 2026 Student Research Symposium (২ পৃষ্ঠা) | ২৪ সেপ্টেম্বর ২০২৬ AoE (verified) | 45-60% |
| ৪ | Numerical Algorithms (Springer) / JCAM (Elsevier) | rolling — নভেম্বরে target | প্রথমবারে 25-35% |
| ৫ | ReScience C / SoftwareX | rolling — ডিসে/জানু | 40-55% |
| ৬ | ACM TOMS (RCR সহ) | সবচেয়ে আগে মে ২০২৭ | 15-20% |
| ৭ | IEEE Access | rolling, দ্রুত review | 50-65% — কিন্তু **APC ~$1,950** |
| ✗ | **IPDPS 2027 — পাঠাবেন না** | ৮ অক্টো ২০২৬ (verified) | 3-5%, parallelism contribution নেই |

**ICCIT কেন সঠিক tier:** এটা আপনাদের home venue, IEEE Xplore-indexed, আসল proceedings।
APK framing বাদ দিলে reframed paper ICCIT-এর গড় submission-এর চেয়ে অনেক ভালো।

**সতর্কতা:** ICCIT আর HiPC — দুই জায়গায় পাঠানোর আগে **dual-submission policy** চেক করুন।

---

## ৬. Roadmap (১১ দিনের পরিকল্পনা)

### Phase 0 — Kill and recount (২ দিন: ৯-১০ সেপ্টে) — সবাই একসাথে
- APK নামটা **সম্পূর্ণ মুছে ফেলুন**। `linear.tex`, notebook, guide, presentation — সব জায়গায়
  find-and-replace: "APK", "Adaptive Preconditioned Krylov", "(ours)", "Our Contribution" →
  **"ILU-PCG/BiCGSTAB (PETSc-style default baseline)"**
- বিদ্যমান CSV থেকেই re-analysis: **একটাই harness-level success predicate** (residual
  externally মাপা, solver-এর নিজের flag না) সব ১০টা solver-এ সমানভাবে প্রয়োগ
- **নতুন compute লাগবে না** — কয়েক ঘণ্টার কাজ

### Phase 1 — Numbers freeze (৪ দিন: ১১-১৪ সেপ্টে) — ৩টা track সমান্তরালে
- **Track A (২ জন):** একটা Kaggle re-run — `spsolve`/SuperLU/UMFPACK **সব size-এ, cap ছাড়া**,
  ILU-GMRES(30), standalone ILU-BiCGSTAB/PCG, আর zero-iteration ILU-only control
- **Track B (২ জন):** সব 905 system-কে hypothesis class-এ ভাগ করা — L-matrix (sign pattern),
  H-matrix, SPD (Cholesky চেষ্টা করে), diagonal dominance। **এটাই আসল contribution**
- **Track C (১ জন):** n ≤ 2000-এর 611টা matrix-এ rho(D⁻¹(L+U)) ও rho((D+L)⁻¹U) সরাসরি হিসাব

### Phase 2 — লেখা ও internal red-team (৪ দিন: ১৫-১৮ সেপ্টে)
৬ পৃষ্ঠা, IEEE format। Structure: (১) base paper-এর synthetic claim, (২) dataset ও protocol
সহ 930→905 reconciliation, (৩) **hypothesis-coverage census = central contribution**,
(৪) domain viability map (n ও Wilson interval সহ), (৫) dispatch-এর null result ও
sparse-direct comparison, (৬) threats to validity

### Phase 3 — Submit (২ দিন: ১৯-২০ সেপ্টে)
ICCIT-এ submit → সাথে সাথে arXiv preprint + Zenodo DOI

### Phase 4-5 — Journal version (অক্টো-ডিসে)
সব confound সরানো, ৫টা RHS replicate, ≥৫ বার timing repetition, একটা installable package,
তারপর Numerical Algorithms/JCAM-এ ~২৫ পৃষ্ঠার version

---

## ৭. প্রতিটা Limitation কীভাবে কাটিয়ে উঠবেন (Fix List)

এটাই ডকুমেন্টের সবচেয়ে কাজের অংশ। ১০টা blocking gap, প্রতিটার **সুনির্দিষ্ট সমাধান**,
সময়, এবং কোন phase-এ করবেন।

> **সবচেয়ে গুরুত্বপূর্ণ কথা:** Gap ১, ২, ৪ — এই তিনটার জন্য **নতুন কোনো compute লাগে না**।
> বিদ্যমান CSV থেকেই ঠিক করা যায়, কয়েক ঘণ্টায়। ICCIT deadline ধরার জন্য এগুলোই যথেষ্ট।

---

### 🔴 GAP ১ — Success criterion উত্তর ঠিক কিনা চেক করে না
**সময়:** ৩-৪ দিন (re-analysis কয়েক ঘণ্টা) · **Phase 0**

**সমাধান:**
- **একটাই harness-level success predicate** বানান, দশটা solver-এ **হুবহু একইভাবে** প্রয়োগ করুন:
  `relative residual ≤ 1e-8`, যা **harness নিজে বাইরে থেকে মাপবে** — solver-এর নিজের
  convergence flag কখনোই বিশ্বাস করবেন না
- Forward error আলাদা column-এ report করুন, কিন্তু success-এর গেট হিসেবে ব্যবহার করবেন না
  (যদি করেন, তাহলে দুইটা variant-ই দেখাতে হবে)
- প্রতিটা table, প্রতিটা figure, ablation count, per-domain rate — **সব আবার হিসাব করুন**
- **Jacobi/GS table-এর 0-vs-1 threshold sensitivity abstract-এ নিজে স্বীকার করুন।**
  নিজে স্বীকার করলে বাঁচা যায়; reviewer ধরে ফেললে যায় না।
- পাশাপাশি `x_true ~ N(0,1)`-এর ৩টা replicate চালান (seed রেকর্ড করে)। কারণ দুইটা
  "converged" label (aft01, Muu) আসলে all-ones RHS + x0=0-এর artifact — random target-এ
  ওগুলো মিলিয়ে যায়

---

### 🔴 GAP ২ — Denominator দুই দিকেই ভুল
**সময়:** ১-২ দিন · **Phase 0**

**সমাধান:**
- প্রতিটা reliability figure-কে **দুইটা কলামে ভাগ করুন, কখনো গুণ করবেন না**:
  - **APPLICABILITY** = corpus-এর কত অংশে method-টা আদৌ defined
  - **CONDITIONAL SUCCESS** = verified success ÷ applicable
- Corpus coverage আলাদা করে, স্পষ্ট label দিয়ে report করুন
- **Abstract-এ 414 denominator লিখে দিন**
- প্রতিটা denominator ৯৩০-এ মিলিয়ে দিন, প্রতিটা বাদ পড়া matrix-এর reason code সহ:
  `930 → 6 load failure → 924 loaded → 19 structurally singular → 905 scored`
- ৯টা condition-number failure যেগুলো এখন কোনো plot-এ নেই, সেগুলোরও হিসাব দিন
- `figures/09_solver_scoreboard.png` আর সব per-domain figure এই ভিত্তিতে নতুন করে বানান

---

### 🔴 GAP ৩ — Spectral ground truth নেই, theorem-class classification নেই
**সময়:** ৪-৬ দিন (compute কয়েক ঘণ্টা, classification+লেখাই আসল খরচ) · **Phase 1 Track B+C**

> ⚠️ **এটাই আপনাদের paper-এর একমাত্র defensible contribution — এবং এটা এখনো অস্তিত্বেই নেই।**

**সমাধান:**
- n ≤ 2000-এর **৬১১টা matrix-এ সরাসরি হিসাব করুন**:
  `rho(D⁻¹(L+U))` এবং `rho((D+L)⁻¹U)`
- Theoretical 2×2 classification table আর empirical table পাশাপাশি দেখান, প্রতিটা
  disagreement ব্যাখ্যা করে
- তারপর **সব ৯০৫টা system-কে hypothesis class-এ ভাগ করুন**:

  | Class | কীভাবে চিনবেন |
  |---|---|
  | L-matrix | sign pattern (a_ii > 0, a_ij ≤ 0) |
  | H-matrix | `rho(|D|⁻¹|L+U|) < 1` |
  | Strict/generalised diagonal dominance | row sum |
  | SPD | Cholesky চেষ্টা করে |
  | Property-A proxy | `rho_GS = rho_J²` ±0.1%-এর মধ্যে |
  | None-of-the-above | বাকি সব |

- **Coverage-টাই paper-এর headline measurement** — এখন পর্যন্ত জানা: 64/75 H-matrix,
  আর exactly-diagonalisable subsample-এ 59/60 union coverage
- একই classification ১০০টা synthetic matrix-এও চালান (সবগুলোর `rho(T_J) ≥ 1.26`) —
  এতে **population gap পরিমাপ** হয়ে যায়

---

### 🔴 GAP ৪ — APK-কে rename ও demote করতে হবে
**সময়:** ২-৩ দিন · **Phase 0 + Phase 1**

**সমাধান:**
- `linear.tex:276` সহ সব জায়গা থেকে মুছুন: "APK", "Adaptive Preconditioned Krylov",
  "(ours)", "Our Contribution"
- নতুন নাম: **"ILU-PCG/BiCGSTAB, the PETSc-style default baseline"**
- Cite করুন: PETSc KSP manual, MATLAB `ilu` documentation example, Barrett et al. (1994),
  Saad (2003), van der Vorst (1992), Meijerink & van der Vorst (1977)
- **যে ablation cell-টা APK-কে justify করত সেটা চালান** — সব ৯০৫-এ:
  ILU-BiCGSTAB-always · ILU-GMRES(30)-always · ILU-PCG-where-applicable · dispatched combination
  — per-matrix win/loss set আর **McNemar exact p-value** সহ
- ফলাফল যে দিকেই যাক report করুন। **একটা plausible heuristic-এর পরিষ্কার null result
  publishable** এবং SolverSet/SALSA-র premise-এর সাথে সরাসরি কথা বলে; কিন্তু সেটা
  লুকালে সেটা **misconduct**
- SPD branch হয় ঠিক করুন (আসল incomplete Cholesky — `ilupp`, `pymatting ichol`),
  নয়তো branch-টা **মুছে দিন** — test data বলছে মুছলে কিছুই হারাবে না

---

### 🔴 GAP ৫ — যে baseline-গুলোর অনুপস্থিতিই সব কাজ করছে
**সময়:** ৪-৫ দিন (বেশিরভাগই একটা Kaggle re-run) · **Phase 1 Track A**

**সমাধান:** নিচের সবগুলো **first-class solver** হিসেবে যোগ করুন — একই ৯০৫ system-এ,
**সব size-এ (cap ছাড়া)**, একই tolerance, একই RHS, একই external success check:

| যোগ করতে হবে | কেন |
|---|---|
| `scipy.sparse.linalg.spsolve` / `splu` (SuperLU) | এটাই আপনাদের হারাচ্ছে |
| `spsolve(use_umfpack=True)` | দ্বিতীয় direct baseline |
| ILU-GMRES(30) এবং ILU-GMRES(50) | PETSc-র আসল default |
| Standalone ILU-BiCGSTAB, ILU-PCG | dispatch ছাড়া baseline |
| `pyamg smoothed_aggregation_solver` | SPD/elliptic domain-এর জন্য |
| **Zero-iteration control**: `x = M.solve(b)` | ILU একাই কতটা করে দেখতে |

- প্রতিটার জন্য fill ratio `nnz(L+U)/nnz(A)` আর factorisation time log করুন
- সবগুলো pip-installable, **নতুন solver code লিখতে হবে না**
- আপনাদের hand-written direct solver-গুলোকে নতুন label দিন:
  **"verification/pedagogical implementations validated against LAPACK"** — performance
  competitor না
- ⚠️ **457 সংখ্যাটা অনেক কমে যাবে। সৎ সংখ্যাটাই লিখুন।**
  "একটা sparse direct solver এই corpus-এর বিপুল সংখ্যাগরিষ্ঠে জেতে, আর দ্রুততর" —
  এটা আগেরটার চেয়ে **ভালো এবং সত্য** headline

---

### 🟠 GAP ৬ — অসম stopping criteria ও অসম budget
**সময়:** ৩-৪ দিন · **Phase 4**

**সমাধান:**
- Refinement-কে **orthogonal experimental factor** বানান: প্রতিটা solver-কে
  `{raw, +1 pass, +2 passes}` — তিনটা condition-এই report করুন, direct method সহ
- সব solver-এর জন্য **একটাই computational budget** ঠিক করুন, হিসাব হবে
  **matrix-vector product + preconditioner apply + setup**-এ — "iteration"-এ না
  (কারণ APK-র এক iteration-এ একটা ILU triangular solve আছে, Jacobi-র নেই)
- "median error per method" বাদ দিয়ে **time-to-solution at common target accuracy**
  (1e-6, 1e-8, 1e-10, 1e-12) — performance profile হিসেবে
- **কখনোই fixed target ছাড়া family-র মধ্যে accuracy ranking দেবেন না**

---

### 🟠 GAP ৭ — ১৬০টা near-singular system error statistics দূষিত করছে
**সময়:** ২-৩ দিন · **Phase 4**

**সমাধান:**
- একটা **numerical-rank screening pass** যোগ করুন
- ফলাফল **দুইটা stratum-এ** ভাগ করে report করুন:
  - **Well-posed stratum** (`cond ≤ 1e15`) — সব headline success rate ও error statistic এখান থেকে
  - **Singular/ill-posed stratum** — success শুধু residual দিয়ে বিচার, forward error = N/A
- Stratum-এর size dataset property হিসেবে report করুন
- যে ৬টা matrix-এর cancellation ratio `||A·1|| / |||A|·1||` < 1e-8 (সর্বনিম্ন 3.5e-17),
  সেগুলো flag করুন — ওখানে b মূলত rounding noise

---

### 🟠 GAP ৮ — সব runtime claim অব্যবহারযোগ্য
**সময়:** মুছতে ১ দিন; ঠিকভাবে করতে ২ সপ্তাহ · **Phase 2 (delete) / Phase 4 (fix)**

**Conference version-এর জন্য সবচেয়ে সস্তা পথ: runtime ranking পুরোপুরি মুছে দিন।**
Defend করার চেয়ে বাদ দেওয়া ভালো।

**Journal version-এর জন্য:**
- GS/SOR-এর জন্য L একবার pre-factor করুন
- সব solver যেন stopping test-এ একই খরচ দেয় — recursive residual ব্যবহার করুন
- যেখানে scipy-র নিজের implementation আছে সেটাই ব্যবহার করুন
- **Primary cost metric = matrix-vector product count**, wall-clock secondary
- ≥৫ বার repetition + একটা discarded warm-up; median ও IQR report করুন
- `OMP_NUM_THREADS` pin করুন, CPU/OS/library version রেকর্ড করুন
- সব ranked median table-এর বদলে **Dolan–Moré performance profile**, failure-কে
  infinite cost ধরে

---

### 🟠 GAP ৯ — Related work-এ মূল theorem ও literature নেই
**সময়:** ৩-৪ দিন পড়া ও লেখা · **Phase 2**

**সমাধান:** এই তিন গুচ্ছ **অবশ্যই cite করতে হবে, শুরুতেই**:

**(ক) যে theorem-গুলো আপনাদের ফলাফল predict করে:**
Stein & Rosenberg (1948) · Householder (1958) · Young (1950, 1971) ·
Varga, *Matrix Iterative Analysis*

**(খ) যে literature-এ আপনাদের "method" আগেই আছে:**
Barrett et al., *Templates* (SIAM 1994) — এবং তার decision flowchart ·
Saad, *Iterative Methods for Sparse Linear Systems* (SIAM 2003) ·
van der Vorst (1992) · Meijerink & van der Vorst (1977) · Benzi (2002, JCP survey) ·
refinement-এর জন্য: Wilkinson (1963), Skeel (1980), Higham ASNA ch.12,
Carson & Higham (SISC 2017, 2018)

**(গ) Auto-selection subfield (২০ বছরের):**
Bhowmick/Eijkhout et al. (2006) · SALSA · Lighthouse · George et al. (2008) ·
Yeom et al. (2016) · **SolverSet (2026)** doi:10.1142/S0218126626500209 ·
arXiv:2606.13255 (621 matrix × 101 PETSc config) ·
arXiv:2406.00809 (867 matrix, 50 domain, Iter-AUC/Time-AUC) ·
Ghai/Lu/Jiao (NLAA 2019) · Ferrari (arXiv:2409.11515) · Tunnell & Gleich (arXiv:2505.20696)

**তারপর আপনাদের সৎ niche-টা বলুন:**
> ২৬টা real domain + একটা controlled synthetic ২৮তম · এমন একটা size range যাতে
> **n < 1000 আছে যা ওই study-গুলো বাদ দেয়** · এবং একটা ১৯৪৮ সালের claim যা ওদের কেউ ছোঁয়নি

---

### 🟡 GAP ১০ — Released source published result reproduce করে না
**সময়:** ৪-৫ দিন · **Phase 5**

**সমাধান:**
- সব code **একটা installable package**-এ একত্র করুন যেটাই single source of truth;
  notebook সেটা **import করবে, নিজে redefine করবে না**
- `requirements.txt`-এ **hash সহ version pin** করুন
- Synthetic domain ও RHS replicate-এর **seed রেকর্ড ও প্রকাশ** করুন
- প্রতিটা skipped/failed matrix **reason code সহ results table-এ log** করুন, যাতে সব
  denominator ৯৩০-এ মেলে
- `except Exception` (blanket) বাদ দিয়ে **narrow catch** — কোন setting কেন fail করল সেটা log করে
- Selection script প্রকাশ করুন, rule declaratively লিখে:
  *square, real, non-pattern, 5 ≤ n ≤ 10,000, প্রতি group-এর সব qualifying matrix
  অথবা seeded sample of k (seed সহ)* — এবং প্রতি domain-এ কতগুলো criteria মিলেছে vs
  কতগুলো ব্যবহার হয়েছে সেটা report করুন
- Code + data **Zenodo-তে DOI সহ deposit** করুন

---

### 📋 সারসংক্ষেপ: কোনটা কখন

| Gap | সমস্যা | সময় | Compute লাগে? | Phase |
|---|---|---|---|---|
| ১ | Success criterion | ৩-৪ দিন | ❌ (re-analysis) | 0 |
| ২ | Denominator | ১-২ দিন | ❌ | 0 |
| ৪ | APK rename + null result | ২-৩ দিন | আংশিক | 0→1 |
| ৩ | **Spectral + hypothesis class** | ৪-৬ দিন | ✅ কয়েক ঘণ্টা | 1 |
| ৫ | Missing baselines | ৪-৫ দিন | ✅ এক re-run | 1 |
| ৯ | Related work | ৩-৪ দিন | ❌ | 2 |
| ৮ | Runtime (conference-এ মুছে দিন) | ১ দিন | ❌ | 2 |
| ৬ | Stopping criteria | ৩-৪ দিন | ✅ | 4 |
| ৭ | Near-singular strata | ২-৩ দিন | ✅ | 4 |
| ৮ | Runtime (ঠিকভাবে) | ২ সপ্তাহ | ✅ | 4 |
| ১০ | Reproducible artifact | ৪-৫ দিন | ❌ | 5 |

**ICCIT (২০ সেপ্টে) ধরতে হলে দরকার:** Gap ১, ২, ৪, ৩, ৫, ৯, আর ৮-এর delete অংশ।
বাকিগুলো journal version-এর জন্য।

---

## ৮. যা কখনো দাবি করা যাবে না (Must NOT Claim)

এই লিস্টটা মুখস্থ রাখুন — এগুলোর যেকোনো একটা paper-কে reject করাতে পারে:

1. ❌ APK-কে novel বা "our method" বলা। Dispatch contributes **0**।
2. ❌ **"924 real matrices-এ ZERO only-Jacobi"** — denominator 414, আর সঠিক predicate-এ
   count 0 না, **1** (fs_680_1)।
3. ❌ APK-এর accuracy সবচেয়ে ভালো — ওটা refinement-এর ফল, যা baseline পায়নি।
   Refined Gauss-Seidel ২৭ গুণ বেশি accurate।
4. ❌ APK ১০টার মধ্যে ২য় দ্রুততম — median গুলো disjoint, self-selected set-এর উপর।
5. ❌ "first", "largest", "at scale" — SolverSet-এ ১২,৬৫১ matrix, arXiv:2406.00809-এ 867টা
   (1K-100K rows, better metrics)। আপনারা mid-pack, size ceiling-এ **সবার শেষে**।
6. ❌ **97.9% direct-method success rate** — accuracy check ছাড়া "ok" মার্ক করা।
7. ❌ 905 দিয়ে ভাগ করা success rate যখন শত শত not_applicable row আছে।
8. ❌ "APK 457টা matrix solve করে যা কোনো classical method পারে না" — baseline set-এ
   GMRES নেই, AMG নেই, sparse direct solver নেই।
9. ❌ Symmetric branch-কে valid PCG বলা — SuperLU factor symmetric না।
10. ❌ Synthetic domain base paper-এর regime test করে বলা — ওদের phenomenon n=2..5-এ,
    আপনাদের 100টার সবগুলোর rho(T_J) ≥ 1.26।
11. ❌ Denominator ও confidence interval ছাড়া per-domain percentage — ২৭টা domain-এর
    ১৬টাতে ২০-এর কম matrix।
12. ❌ "19 structurally singular" আর "6 corrupt files"-কে finding বলা — ওগুলো
    data-preparation note, result না।
13. ❌ "released source published results reproduce করে" — `src/solvebench/`-এ BiCGSTAB
    নেই, APK নেই, আর cap 3000 (notebook-এ ছিল 2000)।

---

## ৯. আমার সৎ মূল্যায়ন

আপনাদের হাতে আছে **~২ সপ্তাহের সুশৃঙ্খল কাজ** একটা বৈধ IEEE-indexed publication পর্যন্ত,
এবং ডিসেম্বরের মধ্যে একটা mid-tier journal-এর বিশ্বাসযোগ্য পথ।

যা আপনাদের হাতে **নেই** সেটা হলো একটা নতুন method।

আর যে paper-টা আপনারা **পেতে পারেন**, সেটা হারানোর দ্রুততম উপায় হলো — পরবর্তী ১১ দিন
সেই paper-টাকে defend করতে ব্যয় করা যেটা আপনারা পাবেন না।

**ভালো খবর:** Phase 0-এর re-analysis-এ নতুন কোনো compute লাগে না। বিদ্যমান CSV থেকেই
এক ঘণ্টার মধ্যে corrected number বের করা সম্ভব।
