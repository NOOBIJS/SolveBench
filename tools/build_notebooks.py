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
           "benchmark", "__init__"]

DATASET = "mdmarufhasanrubab/solvebench-matrices-large"


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
            'from solvebench import benchmark, config, io_utils, metrics, spectral\n'
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


NOTEBOOKS = {
    "solvebench-main-sweep": {
        "title": "SolveBench: Main Sweep (16 solvers, 930 matrices)",
        "intro": ("# SolveBench -- Main Sweep\\n\\n"
                  "16 solvers across 930 real sparse matrices from 27 domains.\\n\\n"
                  "Refinement is **disabled for every method**, so the comparison "
                  "carries no wrapper advantage. Success is decided by the harness "
                  "from the residual it measures itself, never by a solver's own flag."),
        "body": [MAIN_SWEEP, MAIN_SUMMARY],
        "tables": ["benchmark_results.csv", "method_summary.csv", "per_domain.csv"],
    },
    "solvebench-spectral": {
        "title": "SolveBench: Spectral Analysis and Hypothesis Coverage",
        "intro": ("# SolveBench -- Spectral Analysis\\n\\n"
                  "No solving. Computes rho(T_J) and rho(T_GS) for every matrix, "
                  "plus the classical hypothesis class each one falls in.\\n\\n"
                  "This is the quantity the base paper exists to characterise, and "
                  "it says *why* the main sweep came out as it did."),
        "body": [SPECTRAL, SPECTRAL_SUMMARY],
        "tables": ["spectral.csv", "hypothesis_coverage.csv"],
    },
    "solvebench-refinement-study": {
        "title": "SolveBench: Iterative Refinement as a Controlled Factor",
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

    OUT.mkdir(parents=True, exist_ok=True)
    nb_path = OUT / f"{name}.ipynb"
    nb_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")

    meta = {
        "id": f"mdmarufhasanrubab/{name}",
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
    (OUT / f"{name}.kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return nb_path, len(cells)


if __name__ == "__main__":
    print(f"generating from {LIB} at {datetime.now():%Y-%m-%d %H:%M}\n")
    for name, spec in NOTEBOOKS.items():
        path, n = build(name, spec)
        print(f"  {path.name:<42s} {n} cells  {path.stat().st_size/1024:>7.1f} KB")
    print(f"\nlibrary embedded: {len(MODULES)} modules")
    print("Notebooks emit CSV only; figures are built locally by tools/make_figures.py")
