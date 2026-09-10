"""Is a per-matrix omega worth anything over the fixed 1.25?

Young (1950): for a consistently ordered matrix with property A the optimal relaxation
factor is  omega* = 2 / (1 + sqrt(1 - rho(T_J)^2)).  Most of this corpus is neither
consistently ordered nor property-A, so the formula is a heuristic here, not a theorem --
which is exactly why it has to be measured rather than assumed.
"""
import sys, time
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, "src")
from solvebench import io_utils, iterative_solvers as it, metrics


def rho_jacobi_estimate(A, iters=60, seed=0):
    """Power iteration on T_J = I - D^-1 A, using only matvecs."""
    d = A.diagonal()
    if np.any(np.abs(d) < 1e-14):
        return np.nan
    rng = np.random.default_rng(seed)
    v = rng.normal(size=A.shape[0])
    v /= np.linalg.norm(v) or 1.0
    lam = 0.0
    for _ in range(iters):
        w = v - (A @ v) / d
        nw = np.linalg.norm(w)
        if nw < 1e-300:
            return 0.0
        lam = nw
        v = w / nw
    return float(lam)


def omega_from_rho(rho):
    if not np.isfinite(rho) or rho >= 1.0:
        return 1.0                      # no useful estimate: fall back to Gauss-Seidel
    return float(np.clip(2.0 / (1.0 + np.sqrt(max(1.0 - rho * rho, 0.0))), 1.0, 1.95))


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    ents = sorted(io_utils.discover_matrices("dataset_large"),
                  key=lambda e: e["path"].stat().st_size)
    tally = {k: 0 for k in ("gs", "fixed", "adaptive")}
    recs = []
    seen = 0
    print(f"{'matrix':<20}{'n':>6}{'rho_J':>9}{'omega':>7}   "
          f"{'GS':>10}{'SOR 1.25':>10}{'adaptive':>10}")
    t0 = time.perf_counter()
    for e in ents:
        if seen >= limit:
            break
        try:
            A = io_utils.load_matrix(e["path"])
        except Exception:
            continue
        if np.any(np.abs(A.diagonal()) < 1e-14) or A.shape[0] < 4:
            continue                     # stationary methods undefined; not this question
        x_true, b = io_utils.make_ground_truth(A)
        bn = np.linalg.norm(b) or 1.0
        xn = np.linalg.norm(x_true) or 1.0
        rho = rho_jacobi_estimate(A)
        om = omega_from_rho(rho)
        out = {}
        for tag, w in (("gs", 1.0), ("fixed", 1.25), ("adaptive", om)):
            try:
                x, iters, conv, _ = it.sor(A, b, omega=w)
                sc = metrics.score(A, x, b, x_true, bn, xn, conv)
                out[tag] = (sc["status"], iters)
            except Exception:
                out[tag] = ("error", 0)
            if out[tag][0] == "solved":
                tally[tag] += 1
        seen += 1
        recs.append({'matrix': e['name'], 'n': A.shape[0], 'rho': rho, 'omega': om,
                     **{t + '_status': out[t][0] for t in out},
                     **{t + '_iters': out[t][1] for t in out}})
        f = lambda t: (f"{out[t][1]}" if out[t][0] == "solved" else "-")
        print(f"{e['name']:<20}{A.shape[0]:>6}{rho:>9.4f}{om:>7.3f}   "
              f"{f('gs'):>10}{f('fixed'):>10}{f('adaptive'):>10}", flush=True)

    print(f"\n{'=' * 64}")
    print(f"solved, out of {seen} matrices with a usable diagonal "
          f"({(time.perf_counter() - t0) / 60:.1f} min)")
    for k, label in (("gs", "Gauss-Seidel (omega = 1)"), ("fixed", "SOR, fixed 1.25"),
                     ("adaptive", "SOR, omega from rho(T_J)")):
        print(f"  {label:<28} {tally[k]:>4}")

    import pandas as pd
    d = pd.DataFrame(recs)
    d.to_csv("results/tables/adaptive_sor_probe.csv", index=False)
    both = d[(d.fixed_status == "solved") & (d.adaptive_status == "solved")]
    if len(both):
        r = both.fixed_iters / both.adaptive_iters.replace(0, np.nan)
        print(f"\n  both solve it            : {len(both)}")
        print(f"  adaptive needs fewer     : {int((r > 1).sum())}")
        print(f"  adaptive needs more      : {int((r < 1).sum())}")
        print(f"  identical                : {int((r == 1).sum())}")
        print(f"  median speedup           : {r.median():.2f}x")
        print(f"  total iterations  fixed {int(both.fixed_iters.sum()):>9,}"
              f"   adaptive {int(both.adaptive_iters.sum()):>9,}")
    only = d[(d.fixed_status != "solved") & (d.adaptive_status == "solved")]
    lost = d[(d.fixed_status == "solved") & (d.adaptive_status != "solved")]
    print(f"\n  adaptive rescues         : {len(only)}")
    print(f"  adaptive loses           : {len(lost)}")


if __name__ == "__main__":
    main()
