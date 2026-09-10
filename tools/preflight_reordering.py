"""Run every permutation objective over the whole corpus before spending a Kaggle session.

Two different hangs have already been found inside
scipy.sparse.csgraph.min_weight_full_bipartite_matching -- one with raw ratios on
nnc261, one under a log transform on west0067, a 67x67 matrix with 294 nonzeros. The
second consumed a twelve-hour session that finished 30 matrices of 930. A hang cannot be
interrupted from Python, so the only defence is to try every matrix here first.

    python tools/preflight_reordering.py [--limit N]
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from solvebench import io_utils, reordering  # noqa: E402

SLOW = 30.0     # seconds; anything past this is a problem worth naming


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "dataset_large"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    entries = sorted(io_utils.discover_matrices(args.data),
                     key=lambda e: e["path"].stat().st_size)
    if args.limit:
        entries = entries[:args.limit]

    objectives = [("mc64", reordering.mc64_permutation),
                  ("minsum", reordering.minsum_permutation),
                  ("bottleneck", reordering.bottleneck_permutation)]
    worst, failures, t_start = {}, [], time.perf_counter()

    for i, e in enumerate(entries, 1):
        try:
            A = io_utils.load_matrix(e["path"])
        except Exception as exc:
            failures.append((e["name"], "load", type(exc).__name__))
            continue
        line = f"[{i}/{len(entries)}] {e['name'][:24]:<24} n={A.shape[0]:<6}"
        for name, fn in objectives:
            t0 = time.perf_counter()
            try:
                fn(A)
                dt = time.perf_counter() - t0
            except Exception as exc:
                failures.append((e["name"], name, type(exc).__name__))
                dt = time.perf_counter() - t0
            if dt > worst.get(name, (0, ""))[0]:
                worst[name] = (dt, e["name"])
            line += f"  {name} {dt:6.2f}s"
        if any(w > SLOW for w, _ in worst.values()):
            line += "   <-- SLOW"
        print(line, flush=True)

    print(f"\ncompleted {len(entries)} matrices in "
          f"{(time.perf_counter() - t_start) / 60:.1f} min")
    print("slowest call per objective:")
    for name, (dt, mat) in worst.items():
        print(f"  {name:<11} {dt:7.2f}s   on {mat}")
    if failures:
        print(f"\n{len(failures)} failures:")
        for f in failures[:20]:
            print("   ", f)
    else:
        print("\nno failures -- every matrix completes every objective")


if __name__ == "__main__":
    main()
