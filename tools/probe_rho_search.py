"""If the row ratio is the wrong objective, does targeting rho directly do better?

The bottleneck objective minimises the worst row ratio. It is solved exactly and it beats
MC64 365-0 on that quantity -- and it converts two extra systems, because the row-ratio
bound certifies convergence only below 1 and real matrices sit orders of magnitude above
it (median ratio 597 among the wins). Optimising a bound you never reach buys nothing.

So target the quantity that actually decides. rho(T_GS) is not an assignment problem and
cannot be minimised exactly, but it can be searched: start from the portfolio's choice and
swap pairs of rows while the estimate falls.

The reachable population is small and known in advance -- 40 matrices sit in
rho(T_GS) in [1, 1.2] after the portfolio -- so this probe says yes or no cheaply.

    python tools/probe_rho_search.py [--limit N] [--max-n N] [--budget N]

Writes results/tables/rho_search_probe.csv.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from solvebench import corpus, io_utils, reordering  # noqa: E402

TOL = 1e-14


def feasible_pairs(C):
    """All (i, j) whose rows may swap without putting a zero on either new diagonal.

    Swapping rows i and j moves C[j, i] to (i, i) and C[i, j] to (j, j), so both have to
    be nonzero. That is exactly the pattern of C .* C^T off the diagonal, which is far
    smaller than n^2 and is what makes an exhaustive pass over single swaps affordable.
    """
    S = C.multiply(C.transpose()).tocoo()
    return [(int(i), int(j)) for i, j, v in zip(S.row, S.col, S.data)
            if i < j and abs(v) > TOL]


def swap_rows(C, i, j):
    p = np.arange(C.shape[0])
    p[i], p[j] = j, i
    return C[p, :].tocsr()


def search(C, budget=300, rounds=12, track=False):
    """Greedy first-improvement descent on the estimated rho(T_GS).

    Candidate swaps are ranked by the row ratios they touch: a swap that leaves the two
    worst rows alone is unlikely to move the spectral radius, so those are tried first and
    the budget is spent where it has a chance.
    """
    best_rho = reordering._estimate_rho_gs(C)
    evals = 1
    history = [best_rho]
    perm = np.arange(C.shape[0])          # cumulative, so b can follow the rows
    for _ in range(rounds):
        if evals >= budget or not np.isfinite(best_rho):
            break
        pairs = feasible_pairs(C)
        if not pairs:
            break
        M = abs(C).tocsr()
        rowsum = np.asarray(M.sum(axis=1)).ravel()
        d = np.abs(C.diagonal())
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = (rowsum - d) / d
        finite = ratios[np.isfinite(ratios)]
        big = float(finite.max()) if finite.size else 1.0
        ratios = np.where(np.isfinite(ratios), ratios, big)
        score = [max(ratios[i], ratios[j]) for i, j in pairs]
        pairs = [p for _, p in sorted(zip(score, pairs), key=lambda t: -t[0])]
        improved = False
        for i, j in pairs:
            if evals >= budget:
                break
            cand = swap_rows(C, i, j)
            rho = reordering._estimate_rho_gs(cand)
            evals += 1
            if rho < best_rho - 1e-12:
                best_rho, C, improved = rho, cand, True
                perm[i], perm[j] = perm[j], perm[i]
                history.append(best_rho)
                break
        if not improved:
            break
    return (C, best_rho, evals, history, perm) if track else (C, best_rho, evals, history)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "dataset_large"))
    ap.add_argument("--results", default=str(ROOT / "results" / "tables"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-n", type=int, default=3000)
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--rho-hi", type=float, default=1.2)
    args = ap.parse_args()

    r = corpus.apply(pd.read_csv(Path(args.results) / "reordering_study.csv"))
    gs = r[r.method == "Gauss-Seidel"].dropna(subset=["rho_gauss_seidel"])
    piv = gs.pivot_table(index="matrix", columns="condition",
                         values="rho_gauss_seidel", aggfunc="first")
    piv = piv.replace([np.inf, -np.inf], np.nan)
    reach = piv[(piv["best"] >= 1.0) & (piv["best"] < args.rho_hi)]
    targets = set(reach.index)
    print(f"matrices left just above the threshold by the portfolio "
          f"(1 <= rho < {args.rho_hi}): {len(targets)}", flush=True)

    entries = [e for e in io_utils.discover_matrices(args.data) if e["name"] in targets]
    entries.sort(key=lambda e: e["path"].stat().st_size)
    if args.limit:
        entries = entries[:args.limit]
    print(f"found locally: {len(entries)}\n", flush=True)

    rows, t0 = [], time.perf_counter()
    for k, e in enumerate(entries, 1):
        try:
            A = io_utils.load_matrix(e["path"])
        except Exception as exc:
            print(f"[{k}] {e['name']:<20} load failed: {type(exc).__name__}", flush=True)
            continue
        if A.shape[0] > args.max_n:
            print(f"[{k}] {e['name']:<20} n={A.shape[0]} over cap, skipped", flush=True)
            continue
        C, _, info = reordering.select_diagonal(A, None, objective="best")
        rho0 = reordering._estimate_rho_gs(C)
        t1 = time.perf_counter()
        C2, rho1, evals, hist = search(C, budget=args.budget)
        dt = time.perf_counter() - t1
        rows.append({"matrix": e["name"], "domain": e["domain"], "n": A.shape[0],
                     "nnz": A.nnz, "chosen": info.get("chosen"),
                     "rho_portfolio": rho0, "rho_searched": rho1,
                     "crossed": bool(rho0 >= 1.0 > rho1),
                     "evals": evals, "search_sec": dt,
                     "worst_ratio_portfolio": reordering.worst_row_ratio(C),
                     "worst_ratio_searched": reordering.worst_row_ratio(C2)})
        flag = "  <-- CROSSED" if rows[-1]["crossed"] else ""
        print(f"[{k}/{len(entries)}] {e['name']:<20} n={A.shape[0]:<6} "
              f"rho {rho0:.6g} -> {rho1:.6g}  ({evals} evals, {dt:.1f}s){flag}",
              flush=True)

    out = Path(args.results) / "rho_search_probe.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    if len(df):
        print(f"\ncrossed below 1: {int(df.crossed.sum())} / {len(df)}")
    print(f"wrote {out}  ({time.perf_counter()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
