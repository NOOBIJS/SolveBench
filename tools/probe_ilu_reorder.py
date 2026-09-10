"""Does choosing the diagonal first make ILU buildable where it currently is not?

The main sweep found 166 matrices on which no usable ILU factorization exists. On those
the whole preconditioned family is unavailable at once, and what remains solves almost
nothing: BiCGSTAB 12 of 166, GMRES 11, Gauss-Seidel 0, against sparse direct's 148.

build_ilu tries spilu at three drop/fill settings and, failing those, falls back to
diagonal scaling -- which needs a nonzero diagonal. So a zero on the diagonal closes the
last door. That is exactly what a row permutation can fix, which makes this a hypothesis
worth one local run before spending a Kaggle session.

    python tools/probe_ilu_reorder.py [--limit N]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from solvebench import (corpus, io_utils, iterative_solvers as it, metrics,  # noqa: E402
                        reference_solvers as ref, reordering)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "dataset_large"))
    ap.add_argument("--results", default=str(ROOT / "results" / "tables"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    res = corpus.apply(pd.read_csv(Path(args.results) / "benchmark_results.csv"))
    res = res[res.refinement_passes == 0]
    ilu = res[res.method == "ILU-BiCGSTAB"]
    targets = set(ilu[ilu.status == "not_applicable"].matrix)
    print(f"matrices with no usable ILU: {len(targets)}", flush=True)

    entries = [e for e in io_utils.discover_matrices(args.data) if e["name"] in targets]
    entries.sort(key=lambda e: e["path"].stat().st_size)
    if args.limit:
        entries = entries[:args.limit]
    print(f"found locally: {len(entries)}\n", flush=True)

    rows, t0 = [], time.perf_counter()
    for i, e in enumerate(entries, 1):
        try:
            A = io_utils.load_matrix(e["path"])
        except Exception as exc:
            print(f"[{i}/{len(entries)}] {e['name']:<22} load failed: "
                  f"{type(exc).__name__}", flush=True)
            continue
        n = A.shape[0]
        zeros = int((np.abs(A.diagonal()) < 1e-14).sum())
        row = {"matrix": e["name"], "domain": e["domain"], "n": n, "nnz": A.nnz,
               "zero_diag": zeros}

        line = (f"[{i}/{len(entries)}] {e['name']:<22} n={n:<6} zeros={zeros:<6}"
                f" | {(time.perf_counter() - t0) / 60:5.1f} min")
        # Building the factorization is not the claim; solving the system is. Both are
        # recorded, because "ILU became constructible" and "the system was solved" are
        # different results, and conflating them is the error this project exists to
        # avoid repeating.
        x_true, b = io_utils.make_ground_truth(A)
        bn = np.linalg.norm(b) or 1.0
        xn = np.linalg.norm(x_true) or 1.0
        for cond in ("none", "mc64", "best"):
            t = time.perf_counter()
            built, z2, status = False, -1, "error"
            try:
                A2, b2, info = reordering.select_diagonal(A, b, objective=cond)
                z2 = int((np.abs(A2.diagonal()) < 1e-14).sum())
                built = it.build_ilu(A2) is not None
                if built:
                    x, iters, conv, work = ref.ilu_bicgstab(A2, b2)
                    sc = metrics.score(A2, x, b2, x_true, bn, xn, conv)
                    status = sc["status"]
                    row[cond + "_residual"] = sc.get("residual_rel")
                    row[cond + "_iterations"] = iters
                else:
                    status = "not_applicable"
            except Exception as exc:
                row[cond + "_error"] = "{}: {}".format(type(exc).__name__, exc)[:120]
            row[cond] = bool(built)
            row[cond + "_status"] = status
            row[cond + "_zeros_after"] = z2
            row[cond + "_sec"] = time.perf_counter() - t
            tag = "SOLVED" if status == "solved" else ("built" if built else "  -   ")
            line += "  {}:{:<6}".format(cond, tag)
        print(line, flush=True)
        rows.append(row)

        if i % 20 == 0 or i == len(entries):
            d = pd.DataFrame(rows)
            parts = []
            for c in ("none", "mc64", "best"):
                if c in d:
                    sv = int((d.get(c + "_status") == "solved").sum())
                    parts.append("{}: built {}, solved {}".format(
                        c, int(d[c].sum()), sv))
            print("    ---- after {} of {} | ".format(i, len(entries))
                  + " | ".join(parts), flush=True)

    d = pd.DataFrame(rows)
    out = ROOT / "results" / "tables" / "ilu_reorder_probe.csv"
    d.to_csv(out, index=False)
    print(f"\n{'=' * 70}")
    print(f"ILU builds after choosing the diagonal, on {len(d)} matrices where it did not")
    print(f"{'=' * 70}")
    print("  {:<10}{:>12}{:>9}".format("condition", "ILU builds", "solved"))
    for c in ("none", "mc64", "best"):
        if c in d:
            sv = int((d.get(c + "_status") == "solved").sum())
            print("  {:<10}{:>12}{:>9}".format(c, int(d[c].sum()), sv))
    if {"none", "best"} <= set(d.columns):
        newly = d[(~d.none) & d.best]
        gained = d[(d.get("none_status") != "solved")
                   & (d.get("best_status") == "solved")]
        print("")
        print("  newly buildable : {}".format(len(newly)))
        print("  newly SOLVED    : {}   <- the number that matters"
              .format(len(gained)))
        if len(gained):
            print("")
            print("  by domain:")
            print(gained.domain.value_counts().to_string())

    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
