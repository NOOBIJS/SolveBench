# how much of b survives cancellation in x_true=ones, b=A@x_true; 6 matrices fall below 1e-8
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent


def cancellation_ratios(data_root):
    rows = []
    for path in sorted(glob.glob(os.path.join(data_root, "*", "*.mtx"))):
        name = os.path.basename(path)[:-4]
        try:
            A = sio.mmread(path)
            A = (sp.csr_matrix(A) if not sp.issparse(A) else A.tocsr()).astype(float)
        except Exception:
            rows.append((name, None, "load_failed"))
            continue
        if A.shape[0] != A.shape[1]:
            rows.append((name, None, "not_square"))
            continue
        one = np.ones(A.shape[0])
        b = A @ one                       # exactly the right-hand side the harness uses
        magnitude = abs(A) @ one          # the same sum with no cancellation possible
        denom = np.linalg.norm(magnitude)
        ratio = np.linalg.norm(b) / denom if denom > 0 else np.nan
        rows.append((name, float(ratio), "ok"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "dataset_large"))
    ap.add_argument("--out", default=str(ROOT / "results" / "cancellation.json"))
    args = ap.parse_args()

    rows = cancellation_ratios(args.data)
    ok = sorted([r for r in rows if r[2] == "ok" and r[1] == r[1]], key=lambda r: r[1])

    print(f"matrices measured: {len(ok)}")
    for t in (1e-8, 1e-10, 1e-12, 1e-14):
        print(f"  ratio < {t:.0e}: {sum(1 for r in ok if r[1] < t)}")
    print("\n--- 10 worst ---")
    for name, ratio, _ in ok[:10]:
        print(f"  {name:<22} {ratio:.3e}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({n: r for n, r, _ in ok}, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
