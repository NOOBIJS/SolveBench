"""Does a searched rho below 1 actually become a solved system?

probe_rho_search reports 22 matrices where a pairwise-swap descent pushes the estimated
rho(T_GS) under 1 after the portfolio had left it above. That is a prediction. This
project's rule is that success is decided from the residual by the harness, never from an
estimate, so the prediction has to be cashed in before it counts.

    python tools/verify_rho_search.py
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from solvebench import io_utils, iterative_solvers as it, metrics, reordering  # noqa: E402
from probe_rho_search import search  # noqa: E402


def run_gs(A, b, x_true):
    try:
        x, mv, conv, work = it.gauss_seidel(A, b)
    except Exception as exc:
        return metrics.blank("error", f"{type(exc).__name__}"), np.nan
    sc = metrics.score(A, x, b, x_true, reported_converged=conv)
    return sc, work.matvecs if hasattr(work, "matvecs") else np.nan


def main():
    probe = pd.read_csv(ROOT / "results" / "tables" / "rho_search_probe.csv")
    want = set(probe[probe.crossed].matrix)
    print(f"matrices whose searched rho crossed below 1: {len(want)}\n", flush=True)

    entries = {e["name"]: e for e in io_utils.discover_matrices(ROOT / "dataset_large")}
    rows, t0 = [], time.perf_counter()
    for k, name in enumerate(sorted(want), 1):
        e = entries.get(name)
        if e is None:
            print(f"[{k}] {name:<18} not found locally", flush=True)
            continue
        A = io_utils.load_matrix(e["path"])
        x_true, b = io_utils.make_ground_truth(A)

        C, bc, info = reordering.select_diagonal(A, b, objective="best")
        before, _ = run_gs(C, bc, x_true)

        C2, rho1, evals, _, perm = search(C, budget=400, track=True)
        after, _ = run_gs(C2, bc[perm], x_true)

        rows.append({"matrix": name, "n": A.shape[0],
                     "rho_portfolio": reordering._estimate_rho_gs(C), "rho_searched": rho1,
                     "status_portfolio": before["status"], "status_searched": after["status"],
                     "resid_portfolio": before.get("residual_rel"),
                     "resid_searched": after.get("residual_rel"),
                     "gained": before["status"] != "solved" and after["status"] == "solved",
                     "lost": before["status"] == "solved" and after["status"] != "solved"})
        r = rows[-1]
        flag = "  <== GAINED" if r["gained"] else ("  <== LOST" if r["lost"] else "")
        print(f"[{k}/{len(want)}] {name:<18} n={A.shape[0]:<5} "
              f"{r['status_portfolio']:<17} -> {r['status_searched']:<17}{flag}", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "tables" / "rho_search_verified.csv"
    df.to_csv(out, index=False)
    if len(df):
        print(f"\n  predicted crossings : {len(df)}")
        print(f"  actually solved     : {int((df.status_searched == 'solved').sum())}")
        print(f"  gained (was not solved, now is): {int(df.gained.sum())}")
        print(f"  lost                           : {int(df.lost.sum())}")
    print(f"wrote {out}  ({time.perf_counter()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
