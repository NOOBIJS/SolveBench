"""Is the 166-matrix ILU hole real, or is it our retry ladder?

build_ilu retries spilu at (1e-3,5), (1e-2,10), (1e-2,20) -- that is, with *increasing*
drop_tol, which drops more entries and makes a zero pivot more likely, not less. It never
touches diag_pivot_thresh or permc_spec, which are SuperLU's actual levers against a zero
pivot. Before the ILU-hole result goes into a paper we have to know whether stock spilu
factors these matrices under a ladder that pushes the other way.

A referee will run exactly this. Better that we run it first.

    python tools/audit_ilu_hole.py [--limit N] [--max-n N]

Writes results/tables/ilu_hole_audit.csv.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402

from solvebench import corpus, io_utils  # noqa: E402

# Each entry is a name and the kwargs handed to spilu. "ours" reproduces the ladder the
# benchmark actually used; everything after it pushes toward robustness instead of away.
LADDERS = {
    "ours_1":      dict(drop_tol=1e-3, fill_factor=5),
    "ours_2":      dict(drop_tol=1e-2, fill_factor=10),
    "ours_3":      dict(drop_tol=1e-2, fill_factor=20),
    "low_drop":    dict(drop_tol=1e-6, fill_factor=20),
    "no_drop":     dict(drop_tol=0.0,  fill_factor=50),
    "full_pivot":  dict(drop_tol=1e-3, fill_factor=10, diag_pivot_thresh=1.0),
    "permc_mmd":   dict(drop_tol=1e-4, fill_factor=20, permc_spec="MMD_AT_PLUS_A"),
    "permc_natural": dict(drop_tol=1e-4, fill_factor=20, permc_spec="NATURAL"),
}


def try_ilu(A, kwargs, rng):
    """Return (built, usable, seconds). usable means it maps a vector to finite values.

    A factorization that "succeeds" on a structurally singular matrix can still hand back
    inf or nan, and counting that as a win is exactly the mistake this audit exists to
    catch.
    """
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ilu = spla.spilu(A.tocsc(), **kwargs)
    except Exception:
        return False, False, time.perf_counter() - t0
    dt = time.perf_counter() - t0
    try:
        y = ilu.solve(rng.standard_normal(A.shape[0]))
        return True, bool(np.all(np.isfinite(y))), dt
    except Exception:
        return True, False, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "dataset_large"))
    ap.add_argument("--results", default=str(ROOT / "results" / "tables"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-n", type=int, default=10_000)
    args = ap.parse_args()

    res = corpus.apply(pd.read_csv(Path(args.results) / "benchmark_results.csv"))
    res = res[res.refinement_passes == 0]
    ilu = res[res.method == "ILU-BiCGSTAB"]
    targets = set(ilu[ilu.status == "not_applicable"].matrix)
    print(f"matrices with no usable ILU in the main sweep: {len(targets)}", flush=True)

    entries = [e for e in io_utils.discover_matrices(args.data) if e["name"] in targets]
    entries.sort(key=lambda e: e["path"].stat().st_size)
    if args.limit:
        entries = entries[:args.limit]
    print(f"found locally: {len(entries)}\n", flush=True)

    rng = np.random.default_rng(0)
    rows, t0 = [], time.perf_counter()
    for i, e in enumerate(entries, 1):
        try:
            A = io_utils.load_matrix(e["path"])
        except Exception as exc:
            print(f"[{i}/{len(entries)}] {e['name']:<22} load failed: "
                  f"{type(exc).__name__}", flush=True)
            continue
        n = A.shape[0]
        if n > args.max_n:
            continue
        struct_sing, _ = io_utils.structural_singularity(A)
        row = {"matrix": e["name"], "domain": e["domain"], "n": n, "nnz": A.nnz,
               "zero_diag": int((np.abs(A.diagonal()) < 1e-14).sum()),
               "structurally_singular": bool(struct_sing)}
        for name, kw in LADDERS.items():
            built, usable, dt = try_ilu(A, kw, rng)
            row[f"{name}_built"] = built
            row[f"{name}_usable"] = usable
            row[f"{name}_sec"] = dt
        rows.append(row)
        won = [k for k in LADDERS if row[f"{k}_usable"]]
        print(f"[{i}/{len(entries)}] {e['name']:<22} n={n:<6} "
              f"sing={'Y' if struct_sing else 'n'}  usable: "
              f"{','.join(won) if won else '(none)'}", flush=True)

    out = Path(args.results) / "ilu_hole_audit.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(rows)} matrices, "
          f"{time.perf_counter() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
