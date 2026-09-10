"""Generate the three Kaggle notebooks from the library.

The previous notebook carried its own copy of every solver, which is how it and
``src/solvebench/`` drifted into disagreeing about the direct-method size cap,
the solver set and the dataset layout. Here the library source is read off disk
and embedded verbatim, so the notebook cannot say anything the library does not.

The sweep is split across three notebooks because a single one no longer fits in
Kaggle's 12-hour session limit -- projected at about 14 hours -- and because the
split is better experimental design:

* **main sweep** -- 16 methods over the whole corpus with refinement disabled
  for everyone, so the headline comparison carries no wrapper confound at all.
* **spectral** -- no solving; rho(T_J), rho(T_GS) and hypothesis-class labels,
  which say *why* the sweep came out the way it did.
* **refinement study** -- 0, 1 and 2 passes for all 16 methods on a stratified
  subsample, refinement as a controlled factor rather than one method's perk.

Notebooks emit CSV only. Figures are built locally by ``tools/make_figures.py``
from those CSVs, so fixing a chart never costs another five-hour run.

    python tools/build_notebooks.py
"""
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "src" / "solvebench"
OUT = ROOT / "kaggle_upload"

MODULES = ["config", "metrics", "direct_solvers", "iterative_solvers",
           "reference_solvers", "refinement", "io_utils", "spectral",
           "reordering", "benchmark", "__init__"]

#: The corpus. Owned by the account that first uploaded it and now public, so kernels
#: under either account can attach it without a second 985 MB copy.
DATASET = "mdmarufhasanrubab/solvebench-matrices-large"

#: Account the generated kernels belong to.
KAGGLE_USER = "ijsasif"


# --------------------------------------------------------------------- cells

def cell_threads():
    """Thread pinning must happen before numpy is imported anywhere."""
    return '''# Pin BLAS threads BEFORE numpy is imported, or the setting is ignored.
# Runtime on a shared machine is otherwise unreproducible: numpy's dense
# operations spawn a thread count that depends on the host and its load, and
# that variation lands directly in the timing column.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
print("BLAS threads pinned to 1 for reproducible timing")'''


def cell_library():
    """Write the package to disk from embedded source, then import it."""
    payload = {m: (LIB / f"{m}.py").read_text(encoding="utf-8") for m in MODULES}
    return ('# The solvebench package, embedded verbatim from src/solvebench/ by\n'
            '# tools/build_notebooks.py. Do not edit here: edit the library and\n'
            '# regenerate, or the notebook and the library drift apart again.\n'
            'import sys, pathlib, json\n\n'
            f'_SOURCES = json.loads(r"""{json.dumps(payload)}""")\n\n'
            '_pkg = pathlib.Path("/kaggle/working/solvebench")\n'
            'if not pathlib.Path("/kaggle").exists():\n'
            '    _pkg = pathlib.Path("solvebench")\n'
            '_pkg.mkdir(parents=True, exist_ok=True)\n'
            'for _name, _src in _SOURCES.items():\n'
            '    (_pkg / f"{_name}.py").write_text(_src, encoding="utf-8")\n'
            'sys.path.insert(0, str(_pkg.parent))\n\n'
            'import solvebench\n'
            'from solvebench import benchmark, config, io_utils, metrics, reordering, spectral\n'
            'print(f"library loaded: {len(_SOURCES)} modules, "\n'
            '      f"{len(benchmark.METHODS)} methods")\n'
            'for _m in benchmark.METHODS:\n'
            '    print(f"   {_m.family:<20s} {_m.name}")')


def cell_env():
    return '''import platform, numpy, scipy, pandas, multiprocessing, json, time
from datetime import datetime

ON_KAGGLE = pathlib.Path("/kaggle").exists()
ENV = {
    "started": datetime.now().isoformat(),
    "on_kaggle": ON_KAGGLE,
    "python": platform.python_version(),
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "pandas": pandas.__version__,
    "cpu_count": multiprocessing.cpu_count(),
    "platform": platform.platform(),
}
for k, v in ENV.items():
    print(f"  {k:<12s} {v}")

# scipy version is recorded deliberately. Kaggle runs 1.16.3 where local setups
# often run 1.17.x, and the difference is not cosmetic: spilu on a structurally
# singular matrix raises a catchable error on 1.17 but exhausts memory and gets
# the process OS-killed on 1.16.3. Two full sweeps died that way.'''


def cell_data():
    return '''DATA_ROOT = None
for _candidate in (pathlib.Path("/kaggle/input"), pathlib.Path("dataset_large")):
    if not _candidate.exists():
        continue
    _hits = list(_candidate.rglob("*.mtx"))
    if _hits:
        # Walk up from any matrix to the directory holding the domain folders.
        DATA_ROOT = _hits[0].parent.parent
        break

if DATA_ROOT is None:
    # A dataset attached seconds ago may not have propagated yet; that race
    # produced an empty /kaggle/input and a FileNotFoundError on an early run.
    print("NO MATRICES FOUND. Contents of /kaggle/input:")
    for _p in sorted(pathlib.Path("/kaggle/input").rglob("*"))[:40]:
        print("   ", _p)
    raise FileNotFoundError("dataset not attached or not yet propagated")

MATRICES = io_utils.discover_matrices(DATA_ROOT)
print(f"corpus root : {DATA_ROOT}")
print(f"matrices    : {len(MATRICES)}")
print(f"domains     : {len({m['domain'] for m in MATRICES})}")

OUT_DIR = pathlib.Path("/kaggle/working/output" if ON_KAGGLE else "results/output")
(OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
print(f"output      : {OUT_DIR}")'''


def cell_finish(name, tables):
    listing = ", ".join(f'"{t}"' for t in tables)
    return f'''ENV["finished"] = datetime.now().isoformat()
ENV["runtime_minutes"] = (time.perf_counter() - T0) / 60
ENV["notebook"] = "{name}"
ENV["config"] = {{k: getattr(config, k) for k in dir(config)
                 if k.isupper() and not k.startswith("_")}}
(OUT_DIR / "logs" / "run_metadata.json").write_text(json.dumps(ENV, indent=2, default=str))

print("=" * 70)
print(f"{name} COMPLETE in {{ENV['runtime_minutes']:.1f}} min")
print("=" * 70)
for _t in [{listing}]:
    _f = OUT_DIR / "tables" / _t
    print(f"  {{_t:<34s}} {{_f.stat().st_size/1024:>9.1f}} KB" if _f.exists()
          else f"  {{_t:<34s}} MISSING")'''


# ------------------------------------------------------------- notebook bodies

MAIN_SWEEP = '''T0 = time.perf_counter()
import pandas as pd

# Refinement is disabled for every method here. The headline comparison must
# carry no wrapper advantage for anyone; refinement is measured separately, as a
# controlled factor, in the refinement-study notebook.
rows, specs = [], []
results_csv = OUT_DIR / "tables" / "benchmark_results.csv"

order = sorted(MATRICES, key=lambda e: e["path"].stat().st_size)
for i, entry in enumerate(order, 1):
    print(f"\\n[{i}/{len(order)}] {entry['domain']}/{entry['name']}"
          f"   ({(time.perf_counter()-T0)/60:.1f} min)")
    r, _ = benchmark.run_one_matrix(entry["domain"], entry["name"], entry["path"],
                                    refinement_passes=(0,), with_spectral=False,
                                    verbose=True)
    rows.extend(r)
    pd.DataFrame(rows).to_csv(results_csv, index=False)   # checkpoint every matrix

results = pd.DataFrame(rows)
print(f"\\nrows {len(results):,}  matrices {results['matrix'].nunique()}")
print(results["status"].value_counts().to_string())'''

MAIN_SUMMARY = '''summary = benchmark.summarise(results, passes=0)
summary.to_csv(OUT_DIR / "tables" / "method_summary.csv", index=False)
print("Applicability and conditional success are separate columns and are never")
print("multiplied: doing that reported Conjugate Gradient at 10.4% when its")
print("conditional success is 83.9%.\\n")
print(summary.to_string(index=False))

# Per-domain, with the denominator carried so a rate can always be checked.
per_domain = (results[results.refinement_passes == 0]
              .groupby(["domain", "method"])
              .agg(n=("status", "size"),
                   applicable=("status", lambda s: s.isin(metrics.APPLICABLE_STATUSES).sum()),
                   solved=("status", lambda s: (s == metrics.STATUS_SOLVED).sum()))
              .reset_index())
per_domain.to_csv(OUT_DIR / "tables" / "per_domain.csv", index=False)
print(f"\\nper-domain rows: {len(per_domain)}")'''

SPECTRAL = '''T0 = time.perf_counter()
import pandas as pd

# No solving happens here. This computes the quantity the base paper is about
# and the first sweep never calculated: the spectral radius of each iteration
# matrix, plus which classical theorem covers each matrix.
specs = []
spectral_csv = OUT_DIR / "tables" / "spectral.csv"

order = sorted(MATRICES, key=lambda e: e["path"].stat().st_size)
for i, entry in enumerate(order, 1):
    t0 = time.perf_counter()
    row = {"domain": entry["domain"], "matrix": entry["name"]}
    try:
        A = io_utils.load_matrix(entry["path"])
        row["n"], row["nnz"] = A.shape[0], A.nnz
        zr, zc = io_utils.structural_singularity(A)
        if zr or zc:
            row["status"] = "structurally_singular"
        else:
            rho_j, how_j = spectral.spectral_radius(A, "jacobi")
            rho_g, how_g = spectral.spectral_radius(A, "gauss_seidel")
            row.update(spectral.classify(A, rho_j, rho_g),
                       rho_jacobi_method=how_j, rho_gs_method=how_g,
                       status="analysed")
    except Exception as e:
        row["status"] = f"failed: {type(e).__name__}: {e}"
    row["seconds"] = time.perf_counter() - t0
    specs.append(row)
    print(f"[{i}/{len(order)}] {entry['name']:<24s} "
          f"rho_J={row.get('rho_jacobi', float('nan')):>12.6f} "
          f"rho_GS={row.get('rho_gauss_seidel', float('nan')):>12.6f} "
          f"{row.get('hypothesis_class', row['status'])}  [{row['seconds']:.1f}s]")
    pd.DataFrame(specs).to_csv(spectral_csv, index=False)

spec_df = pd.DataFrame(specs)
print(f"\\nanalysed {len(spec_df)} matrices")'''

SPECTRAL_SUMMARY = '''ok = spec_df[spec_df.status == "analysed"]
print("=== hypothesis-class coverage ===")
print(ok.hypothesis_class.value_counts().to_string())

print("\\n=== convergence verdicts (three-way, not the textbook binary) ===")
for col in ("jacobi_verdict", "gs_verdict"):
    print(f"\\n{col}:")
    print(ok[col].value_counts().to_string())

print("\\n=== the base paper's question, answered by theory rather than by running ===")
jc = ok.jacobi_verdict == "converges"
gc = ok.gs_verdict == "converges"
print(f"  both converge      : {(jc & gc).sum()}")
print(f"  only Gauss-Seidel  : {(~jc & gc).sum()}")
print(f"  only Jacobi        : {(jc & ~gc).sum()}")
print(f"  neither            : {(~jc & ~gc).sum()}")
print("\\nStein-Rosenberg (1948) forbids the only-Jacobi case for M-matrices, and")
print("Householder-John (1958) guarantees Gauss-Seidel for SPD. Coverage of those")
print("classes is what explains the count above.")

ok.groupby("hypothesis_class").size().to_frame("matrices").to_csv(
    OUT_DIR / "tables" / "hypothesis_coverage.csv")'''

REFINEMENT = '''T0 = time.perf_counter()
import pandas as pd, numpy as np

# A stratified subsample: refinement at three levels for all 16 methods over the
# whole corpus would not fit in a session. Sampling by size decile keeps the
# spread of the corpus rather than favouring the small, fast matrices.
SAMPLE_SIZE = 200
sizes = np.array([e["path"].stat().st_size for e in MATRICES])
deciles = np.quantile(sizes, np.linspace(0, 1, 11))
rng = np.random.default_rng(20260909)      # seed published with the results
sample = []
for lo, hi in zip(deciles[:-1], deciles[1:]):
    bucket = [e for e, s in zip(MATRICES, sizes) if lo <= s <= hi]
    take = min(SAMPLE_SIZE // 10, len(bucket))
    idx = rng.choice(len(bucket), size=take, replace=False)
    sample += [bucket[j] for j in idx]
print(f"stratified sample: {len(sample)} matrices across 10 size deciles (seed 20260909)")

rows = []
csv = OUT_DIR / "tables" / "refinement_study.csv"
for i, entry in enumerate(sample, 1):
    print(f"\\n[{i}/{len(sample)}] {entry['domain']}/{entry['name']}"
          f"   ({(time.perf_counter()-T0)/60:.1f} min)")
    r, _ = benchmark.run_one_matrix(entry["domain"], entry["name"], entry["path"],
                                    refinement_passes=config.REFINEMENT_STUDY_PASSES,
                                    with_spectral=False, verbose=False)
    rows.extend(r)
    pd.DataFrame(rows).to_csv(csv, index=False)

study = pd.DataFrame(rows)
print(f"\\nrows {len(study):,}")'''

REFINEMENT_SUMMARY = '''solved = study[study.status == metrics.STATUS_SOLVED]
pivot = solved.pivot_table(index="method", columns="refinement_passes",
                           values="error_rel", aggfunc="median")
pivot.to_csv(OUT_DIR / "tables" / "refinement_effect.csv")
print("=== median relative forward error by refinement passes ===")
print("Every method gets the same wrapper. In the first sweep only one did, and")
print("that is where its accuracy advantage came from.\\n")
print(pivot.to_string(float_format=lambda v: f"{v:.3e}"))

cost = solved.pivot_table(index="method", columns="refinement_passes",
                          values="matvecs", aggfunc="median")
print("\\n=== median matrix-vector products (the cost of that accuracy) ===")
print(cost.to_string(float_format=lambda v: f"{v:.0f}"))
cost.to_csv(OUT_DIR / "tables" / "refinement_cost.csv")'''


REORDERING = '''T0 = time.perf_counter()
import numpy as np
import pandas as pd

# The question this notebook exists to answer: how many systems does *choosing* the
# diagonal move from "no stationary method works" to "solved"?
#
#   none  -- the matrix as it arrives, the baseline every earlier result uses
#   mc64  -- the established permutation, maximising the product of |diagonal| entries
#   best  -- compute mc64, min-sum and bottleneck, keep whichever gives the smallest
#            estimated rho(T_GS). "none" is among the candidates, so this can never do
#            worse than leaving the matrix alone.
CONDITIONS = ("none", "mc64", "best")
STATIONARY = {"Jacobi": solvebench.iterative_solvers.jacobi,
              "Gauss-Seidel": solvebench.iterative_solvers.gauss_seidel,
              "SOR": solvebench.iterative_solvers.sor}

# Set by the generator. The probe variant trades corpus coverage and the exact spectra
# for a run short enough to confirm the pipeline works before the full sweep finishes.
PROBE_N = __PROBE_N__
WITH_EXACT_SPECTRA = __WITH_SPECTRA__

_SHORT = {"solved": "ok", "not_applicable": "n/a", "diverged": "div",
          "inaccurate": "bad", "error": "err"}


def _outcome(r):
    """One compact token per method, e.g. ``Jac:ok(37)`` or ``SOR:cap(10000)``.

    ``cap`` and ``stop`` are both did_not_converge but mean opposite things: cap ran the
    full iteration budget and was still going, stop was abandoned early by the
    divergence guard. Collapsing them hides which one happened.
    """
    name = str(r.get("method", "?"))[:3]
    it = r.get("iterations")
    shown = int(it) if isinstance(it, (int, float)) and it == it else "-"
    status = str(r.get("status"))
    if status == "did_not_converge":
        tag = "cap" if shown != "-" and shown >= config.MAX_ITERATIONS else "stop"
    else:
        tag = _SHORT.get(status, status[:3])
    return "{}:{}({})".format(name, tag, shown)


rows = []
csv = OUT_DIR / "tables" / "reordering_study.csv"
order = sorted(MATRICES, key=lambda e: e["path"].stat().st_size)

if PROBE_N and PROBE_N < len(order):
    # Stratified by size decile so the sample keeps the corpus's spread rather than
    # filling up with small, fast matrices.
    rng = np.random.default_rng(20260910)
    edges = np.linspace(0, len(order), 11).astype(int)
    picked = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        bucket = order[lo:hi]
        take = min(PROBE_N // 10, len(bucket))
        picked += [bucket[j] for j in rng.choice(len(bucket), size=take, replace=False)]
    order = sorted(picked, key=lambda e: e["path"].stat().st_size)
    print(f"probe: {len(order)} matrices sampled across 10 size deciles (seed 20260910)")

# Kaggle kills a session at 12 hours without warning. Stop cleanly before that, write
# the results, and say so -- a run that reports "budget reached at matrix 812" is worth
# far more than one that simply vanishes. The previous attempt was cancelled at the
# limit and only survived because Kaggle happened to publish the checkpoint.
TIME_BUDGET_H = 11.0
_total_nnz = sum(e["path"].stat().st_size for e in order)

print("=" * 78)
print("REORDERING STUDY -- convergence-oriented diagonal selection")
print("=" * 78)
print(f"  matrices          : {len(order)}")
print(f"  conditions        : {', '.join(CONDITIONS)}")
print(f"  methods           : {', '.join(STATIONARY)}")
print(f"  iteration cap     : {config.MAX_ITERATIONS:,}   tolerance {config.TOLERANCE:g}")
print(f"  exact spectra     : {'yes, n <= %d' % config.SPECTRAL_EXACT_CAP if WITH_EXACT_SPECTRA else 'no'}")
print(f"  wall-clock budget : {TIME_BUDGET_H} h  (Kaggle cancels at 12 h)")
print(f"  expected runtime  : 3.5-4.5 h typical, 9 h worst case")
print(f"  checkpoint        : {csv.name} rewritten after every matrix")
print("=" * 78, flush=True)

for i, entry in enumerate(order, 1):
    base = {"domain": entry["domain"], "matrix": entry["name"]}
    try:
        A = io_utils.load_matrix(entry["path"])
    except Exception as e:
        rows.append({**base, "status": f"load_failed: {type(e).__name__}"})
        continue
    n = A.shape[0]
    base.update(n=n, nnz=A.nnz)
    zr, zc = io_utils.structural_singularity(A)
    if zr or zc:
        rows.append({**base, "status": "structurally_singular"})
        pd.DataFrame(rows).to_csv(csv, index=False)
        continue

    x_true, b = io_utils.make_ground_truth(A)
    bn = np.linalg.norm(b) or 1.0
    xn = np.linalg.norm(x_true) or 1.0

    # Progress is printed BEFORE the work, and again after each condition. The previous
    # run printed one line per matrix only once all three conditions were done, so when
    # it stalled inside a permutation the log simply stopped -- twelve hours of silence
    # and no way to tell which matrix or which step was responsible. Announcing the
    # matrix first means the last line in the log always names whatever is stuck.
    elapsed = (time.perf_counter() - T0) / 60
    rate = i / max(elapsed, 1e-9)
    eta = (len(order) - i) / rate if rate > 0 else float("nan")
    print(f"[{i:>4}/{len(order)}] {100.0 * i / len(order):5.1f}%  {entry['name']:<24} "
          f"n={n:<6} nnz={A.nnz:<9} | {elapsed / 60:5.2f}h elapsed"
          f" | ETA {eta / 60:5.2f}h | projected total {(elapsed + eta) / 60:5.2f}h",
          flush=True)

    for cond in CONDITIONS:
        print(f"    {cond:<6} ...", flush=True, end="")
        t0 = time.perf_counter()
        try:
            A2, b2, info = reordering.select_diagonal(A, b, objective=cond)
        except Exception as e:
            rows.append({**base, "condition": cond,
                         "status": f"reorder_failed: {type(e).__name__}"})
            continue
        setup = time.perf_counter() - t0
        zeros = int((np.abs(A2.diagonal()) < 1e-14).sum())

        # Exact spectra only for the two conditions the write-up compares, and only
        # where the dense route is affordable. Selection itself uses a cheap estimate.
        rho_j = rho_g = np.nan
        if (WITH_EXACT_SPECTRA and cond in ("none", "best")
                and n <= config.SPECTRAL_EXACT_CAP and zeros == 0):
            try:
                rho_j, _ = spectral.spectral_radius(A2, "jacobi")
                rho_g, _ = spectral.spectral_radius(A2, "gauss_seidel")
            except Exception:
                pass

        for mname, fn in STATIONARY.items():
            # worst_ratio_* is the quantity the method is actually optimising, and the
            # one the convergence guarantee is stated in: below 1 the permuted matrix is
            # strictly diagonally dominant, so Jacobi AND Gauss-Seidel are guaranteed to
            # converge. Recording it for every matrix is what turns "the permutation
            # helped" into "the permutation bought a guarantee, on this many matrices".
            row = {**base, "condition": cond, "method": mname,
                   "chosen": info.get("chosen", cond), "setup_sec": setup,
                   "zero_diagonal": zeros, "rho_jacobi": rho_j, "rho_gauss_seidel": rho_g,
                   "permuted": info.get("permuted", False),
                   "perm_cost": info.get("cost", float("nan")),
                   "worst_ratio_before": info.get("worst_ratio_before", float("nan")),
                   "worst_ratio_after": info.get("worst_ratio_after", float("nan")),
                   **{f"ratio_{k}": v
                      for k, v in info.get("ratios", {}).items()}}
            if zeros:
                row.update(metrics.blank(metrics.STATUS_NOT_APPLICABLE, "zero diagonal"))
                rows.append(row)
                continue
            try:
                t1 = time.perf_counter()
                x, its, conv, work = fn(A2, b2)
                row.update(runtime_sec=time.perf_counter() - t1, iterations=its,
                           **work.as_dict(),
                           **metrics.score(A2, x, b2, x_true, bn, xn, conv))
            except solvebench.SolverNotApplicable as e:
                row.update(metrics.blank(metrics.STATUS_NOT_APPLICABLE, str(e)))
            except (MemoryError, RuntimeError, ValueError, ZeroDivisionError) as e:
                row.update(metrics.blank(metrics.STATUS_ERROR, f"{type(e).__name__}: {e}"))
            rows.append(row)

        made = rows[-len(STATIONARY):]
        solved = sum(1 for r in made if r.get("status") == metrics.STATUS_SOLVED)
        detail = " ".join(_outcome(r) for r in made)
        wr = info.get("worst_ratio_after", info.get("worst_ratio_before", float("nan")))
        print(f" {solved}/{len(STATIONARY)} solved | {detail} | perm {setup:6.2f}s"
              f" | total {time.perf_counter() - t0:7.2f}s"
              f" | ratio {wr:.3g}" + (f" | chose {info.get('chosen', cond)}"
                                      if cond == "best" else ""), flush=True)

    pd.DataFrame(rows).to_csv(csv, index=False)

    # A standing scoreboard every 25 matrices: the log should answer "is it working?"
    # without waiting for the end, and "will it finish?" without doing arithmetic.
    if i % 25 == 0 or i == len(order):
        d = pd.DataFrame(rows)
        if "condition" in d and "status" in d:
            tally = {c: int((d[(d.condition == c)].status == metrics.STATUS_SOLVED).sum())
                     for c in CONDITIONS if (d.condition == c).any()}
            spent = (time.perf_counter() - T0) / 3600
            print(f"    ---- after {i}/{len(order)}: solved "
                  + ", ".join(f"{c}={v}" for c, v in tally.items())
                  + f" | {spent:.2f}h spent, {spent / max(i, 1) * len(order):.2f}h projected"
                  + f" | {len(d):,} rows", flush=True)

    # Kaggle cancels at 12 h with no warning. Stop first, and say where we stopped.
    if (time.perf_counter() - T0) / 3600 > TIME_BUDGET_H:
        print(f"\\n*** WALL-CLOCK BUDGET {TIME_BUDGET_H} h REACHED at matrix {i}"
              f"/{len(order)} ({entry['name']}). Stopping cleanly; "
              f"{len(rows):,} rows written to {csv.name}. ***", flush=True)
        break

study = pd.DataFrame(rows)
print(f"\\nrows {len(study):,}")'''

REORDERING_SUMMARY = '''ok = study[study.method.notna()]
piv = (ok.assign(solved=ok.status == metrics.STATUS_SOLVED)
         .pivot_table(index=["matrix", "method"], columns="condition",
                      values="solved", aggfunc="max"))

print("=== systems solved, by condition ===")
for c in CONDITIONS:
    if c in piv:
        print(f"  {c:<6} {int(piv[c].sum()):>5}")

if "none" in piv and "best" in piv:
    gained = piv[(piv["none"] == 0) & (piv["best"] == 1)]
    lost = piv[(piv["none"] == 1) & (piv["best"] == 0)]
    print(f"\\n  RESCUED (failed as given, solved after choosing the diagonal): {len(gained)}")
    print(f"  LOST    (solved as given, failed after)                      : {len(lost)}")
    if "mc64" in piv:
        mc = piv[(piv["none"] == 0) & (piv["mc64"] == 1)]
        print(f"  MC64 rescues                                                 : {len(mc)}")
        print(f"  ours rescues where MC64 does not                             : "
              f"{len(set(gained.index) - set(mc.index))}")
    print("\\n  by method:")
    for m in STATIONARY:
        lvl = piv.index.get_level_values("method")
        if m not in set(lvl):
            continue
        sub = piv.xs(m, level="method")
        g = ((sub["none"] == 0) & (sub["best"] == 1)).sum()
        print(f"    {m:<14} {int(sub['none'].sum()):>4} -> {int(sub['best'].sum()):>4}"
              f"   (+{int(g)} rescued)")

print("\\n=== which objective the selector picked ===")
print(ok[ok.condition == "best"].drop_duplicates("matrix").chosen.value_counts().to_string())

sp_rows = ok[(ok.condition.isin(["none", "best"])) & ok.rho_gauss_seidel.notna()]
if len(sp_rows):
    w = sp_rows.drop_duplicates(["matrix", "condition"]).pivot(
        index="matrix", columns="condition", values="rho_gauss_seidel").dropna()
    if {"none", "best"} <= set(w.columns):
        crossed = ((w["none"] >= 1) & (w["best"] < 1)).sum()
        print(f"\\n=== rho(T_GS) crossing below 1 (n <= {config.SPECTRAL_EXACT_CAP}) ===")
        print(f"  matrices measured both ways  : {len(w)}")
        print(f"  rho >= 1 as given, < 1 after : {int(crossed)}")
        print(f"  median change in rho         : {(w['best'] - w['none']).median():.4g}")

one = ok[ok.condition == "best"].drop_duplicates("matrix")
have = [c for c in ("ratio_none", "ratio_mc64", "ratio_minsum", "ratio_bottleneck")
        if c in one.columns]
if have:
    print("\\n=== strict diagonal dominance, bought by permutation ===")
    print("A worst row ratio below 1 IS strict diagonal dominance, which guarantees both")
    print("Jacobi and Gauss-Seidel converge. The bottleneck objective minimises exactly")
    print("that ratio, so if any row permutation makes the matrix dominant, it finds one:")
    print("the guarantee is reachable precisely when ratio_bottleneck < 1. MC64 maximises")
    print("the product of |diagonal| instead and carries no such statement.")
    for col in have:
        v = pd.to_numeric(one[col], errors="coerce")
        fin = v.replace([np.inf, -np.inf], np.nan)
        print(f"  {col[6:]:<11} dominant {int((v < 1).sum()):>4} / {int(v.notna().sum()):>4}"
              f"   median ratio {fin.median():.4g}"
              f"   undefined {int(np.isinf(v).sum()):>4}")
    if {"ratio_none", "ratio_bottleneck"} <= set(one.columns):
        a = pd.to_numeric(one.ratio_none, errors="coerce")
        c = pd.to_numeric(one.ratio_bottleneck, errors="coerce")
        print(f"\\n  not dominant as given, dominant after bottleneck : "
              f"{int(((a >= 1) & (c < 1)).sum())}")
        print(f"  dominant as given, lost by bottleneck            : "
              f"{int(((a < 1) & (c >= 1)).sum())}")

piv.to_csv(OUT_DIR / "tables" / "reordering_pivot.csv")'''

# The full sweep, and a probe short enough to confirm the pipeline and show the signal
# early. The probe drops the exact spectra -- roughly two hours of the cost -- and
# samples the corpus; everything else, including the 10,000-iteration cap, is identical,
# so the two are directly comparable.
REORDERING_FULL = (REORDERING.replace("__PROBE_N__", "None")
                             .replace("__WITH_SPECTRA__", "True"))
REORDERING_PROBE = (REORDERING.replace("__PROBE_N__", "200")
                              .replace("__WITH_SPECTRA__", "False"))


PIPELINE = '''T0 = time.perf_counter()
import pandas as pd

# Only the pipeline arm runs here. Every baseline it is compared against was measured
# in the main sweep, on this same corpus and through this same scoring, so repeating
# the sixteen would cost hours and add nothing.
rows = []
results_csv = OUT_DIR / "tables" / "pipeline_results.csv"
order = sorted(MATRICES, key=lambda e: e["path"].stat().st_size)
TIME_BUDGET_H = 11.0

print("=" * 78)
print("PIPELINE ARM -- convergence-oriented preprocessing")
print("=" * 78)
print(f"  matrices        : {len(order)}")
print(f"  methods         : {len(benchmark.PIPELINE_METHODS)}")
for _m in benchmark.PIPELINE_METHODS:
    print(f"      {_m.name}")
print(f"  iteration cap   : {config.MAX_ITERATIONS:,}   tolerance {config.TOLERANCE:g}")
print(f"  refinement      : off, as in the main sweep")
print(f"  budget          : {TIME_BUDGET_H} h  (Kaggle cancels at 12 h)")
print(f"  checkpoint      : {results_csv.name} after every matrix")
print("=" * 78, flush=True)

for i, entry in enumerate(order, 1):
    elapsed = (time.perf_counter() - T0) / 60
    eta = (len(order) - i) / max(i / max(elapsed, 1e-9), 1e-9)
    print(f"[{i:>4}/{len(order)}] {100.0 * i / len(order):5.1f}%  {entry['name']:<24}"
          f" | {elapsed / 60:5.2f}h elapsed | ETA {eta / 60:5.2f}h", flush=True)
    r, _ = benchmark.run_one_matrix(entry["domain"], entry["name"], entry["path"],
                                    refinement_passes=(0,), with_spectral=False,
                                    verbose=True,
                                    methods=benchmark.PIPELINE_METHODS)
    rows.extend(r)
    pd.DataFrame(rows).to_csv(results_csv, index=False)

    if i % 25 == 0 or i == len(order):
        d = pd.DataFrame(rows)
        got = int((d.status == metrics.STATUS_SOLVED).sum())
        spent = (time.perf_counter() - T0) / 3600
        print(f"    ---- after {i}/{len(order)}: {got:,} solved rows"
              f" | {spent:.2f}h spent, {spent / max(i, 1) * len(order):.2f}h projected",
              flush=True)

    if (time.perf_counter() - T0) / 3600 > TIME_BUDGET_H:
        print(f"\\n*** BUDGET {TIME_BUDGET_H} h REACHED at matrix {i}"
              f"/{len(order)} ({entry['name']}). Stopping cleanly; {len(rows):,} rows "
              f"written. ***", flush=True)
        break

results = pd.DataFrame(rows)
print(f"\\nrows {len(results):,}  matrices {results['matrix'].nunique()}")'''

PIPELINE_SUMMARY = '''APP = metrics.APPLICABLE_STATUSES
summary = (results.groupby("method")
           .agg(corpus=("status", "size"),
                applicable=("status", lambda s: s.isin(APP).sum()),
                solved=("status", lambda s: (s == metrics.STATUS_SOLVED).sum()))
           .reset_index())
summary["applicability"] = 100 * summary.applicable / summary.corpus
summary["conditional_success"] = 100 * summary.solved / summary.applicable.clip(lower=1)
summary.to_csv(OUT_DIR / "tables" / "pipeline_summary.csv", index=False)

print("Applicability and conditional success stay separate columns here for the same")
print("reason as everywhere else: multiplying them is what reported CG at 10.4%.")
print(summary.to_string(index=False))

print("\\nThe comparison against the sixteen baselines is made locally, against the")
print("main sweep's own results -- both arms ran the same corpus through the same")
print("scoring, so the rows line up on (matrix, status) without any adjustment.")'''


NOTEBOOKS = {
    "solvebench-pipeline": {
        "title": "SolveBench Pipeline",
        "intro": ("# SolveBench -- Pipeline Arm" + chr(92) + "n" + chr(92) + "n"
                  "Two preprocessing steps in front of unmodified solvers: choose the "
                  "diagonal by assignment, then choose omega from an estimate of "
                  "rho(T_J)." + chr(92) + "n" + chr(92) + "n"
                  "Each step is also run alone, because *which* step helps is the "
                  "result -- 'the pipeline helps' is not. Baselines come from the main "
                  "sweep and are not repeated here."),
        "body": [PIPELINE, PIPELINE_SUMMARY],
        "tables": ["pipeline_results.csv", "pipeline_summary.csv"],
    },
    "solvebench-main-sweep": {
        "title": "SolveBench Main Sweep",
        "intro": ("# SolveBench -- Main Sweep\\n\\n"
                  "16 solvers across 930 real sparse matrices from 27 domains.\\n\\n"
                  "Refinement is **disabled for every method**, so the comparison "
                  "carries no wrapper advantage. Success is decided by the harness "
                  "from the residual it measures itself, never by a solver's own flag."),
        "body": [MAIN_SWEEP, MAIN_SUMMARY],
        "tables": ["benchmark_results.csv", "method_summary.csv", "per_domain.csv"],
    },
    "solvebench-spectral": {
        "title": "SolveBench Spectral",
        "intro": ("# SolveBench -- Spectral Analysis\\n\\n"
                  "No solving. Computes rho(T_J) and rho(T_GS) for every matrix, "
                  "plus the classical hypothesis class each one falls in.\\n\\n"
                  "This is the quantity the base paper exists to characterise, and "
                  "it says *why* the main sweep came out as it did."),
        "body": [SPECTRAL, SPECTRAL_SUMMARY],
        "tables": ["spectral.csv", "hypothesis_coverage.csv"],
    },
    "solvebench-reordering-study": {
        "title": "SolveBench Reordering Study",
        "intro": ("# SolveBench -- Reordering Study\\n\\n"
                  "How many systems does *choosing* the diagonal move from unsolvable "
                  "to solved?\\n\\nA stationary method divides by a_ii, so it depends "
                  "entirely on which entries sit on the diagonal -- and that was decided "
                  "by the order the rows happened to arrive in. Three conditions: the "
                  "matrix as given, the established MC64 permutation, and a selection "
                  "among the MC64, min-sum and bottleneck objectives."),
        "body": [REORDERING_FULL, REORDERING_SUMMARY],
        "tables": ["reordering_study.csv", "reordering_pivot.csv"],
    },
    "solvebench-reordering-probe": {
        "title": "SolveBench Reordering Probe",
        "intro": ("# SolveBench -- Reordering Probe\\n\\n"
                  "The same experiment as the full reordering study, on 200 matrices "
                  "sampled across the corpus's size deciles and without the exact "
                  "spectra.\\n\\nIt exists to show the signal early: the full sweep runs "
                  "for hours because the method working is what makes it expensive -- "
                  "491 systems that short-circuit as undefined suddenly have a usable "
                  "diagonal and iterate to the cap."),
        "body": [REORDERING_PROBE, REORDERING_SUMMARY],
        "tables": ["reordering_study.csv", "reordering_pivot.csv"],
    },
    "solvebench-refinement-study": {
        "title": "SolveBench Refinement Study",
        "intro": ("# SolveBench -- Refinement Study\\n\\n"
                  "0, 1 and 2 refinement passes for all 16 methods on a stratified "
                  "subsample.\\n\\nIn the first sweep refinement was applied to one "
                  "method only, which is what produced its accuracy result. Here "
                  "every method gets the same wrapper."),
        "body": [REFINEMENT, REFINEMENT_SUMMARY],
        "tables": ["refinement_study.csv", "refinement_effect.csv", "refinement_cost.csv"],
    },
}


def build(name, spec):
    cells = [("markdown", spec["intro"].replace("\\n", "\n")),
             ("code", cell_threads()),
             ("code", cell_library()),
             ("code", cell_env()),
             ("code", cell_data())]
    cells += [("code", b) for b in spec["body"]]
    cells.append(("code", cell_finish(name, spec["tables"])))

    nb = {
        "cells": [
            {"cell_type": t, "metadata": {},
             **({"source": s.splitlines(keepends=True)} if t == "markdown" else
                {"source": s.splitlines(keepends=True), "outputs": [], "execution_count": None})}
            for t, s in cells
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }

    # One directory per kernel: the Kaggle CLI pushes a folder and requires the
    # metadata to be named exactly kernel-metadata.json inside it.
    kdir = OUT / name
    kdir.mkdir(parents=True, exist_ok=True)
    nb_path = kdir / f"{name}.ipynb"
    nb_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")

    meta = {
        "id": f"{KAGGLE_USER}/{name}",
        "title": spec["title"],
        "code_file": f"{name}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,          # pure CPU: scipy sparse, SuperLU, ARPACK
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [DATASET],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return nb_path, len(cells)


if __name__ == "__main__":
    print(f"generating from {LIB} at {datetime.now():%Y-%m-%d %H:%M}\n")
    for name, spec in NOTEBOOKS.items():
        path, n = build(name, spec)
        print(f"  {path.name:<42s} {n} cells  {path.stat().st_size/1024:>7.1f} KB")
    print(f"\nlibrary embedded: {len(MODULES)} modules")
    print("Notebooks emit CSV only; figures are built locally by tools/make_figures.py")
