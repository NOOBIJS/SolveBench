"""Quick correctness check on one small matrix before running the full 49x8 sweep."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from solvebench.benchmark import run_one_matrix  # noqa: E402

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    mtx_path = project_root / "dataset" / "structural" / "bcsstk01" / "bcsstk01.mtx"
    rows = run_one_matrix("structural", "bcsstk01", mtx_path)
    for r in rows:
        runtime = f"{r['runtime_sec']:.6f}" if r['runtime_sec'] is not None else "-"
        res_rel = f"{r['residual_rel']:.3e}" if r['residual_rel'] is not None else "-"
        err_rel = f"{r['error_rel']:.3e}" if r['error_rel'] is not None else "-"
        iters = r['iterations'] if r['iterations'] is not None else "-"
        print(f"{r['method']:<20s} status={r['status']:<20s} "
              f"runtime={runtime:<12} iters={iters!s:<8} "
              f"rel_residual={res_rel:<12} rel_error={err_rel:<12} "
              f"cond={r['condition_number']:.3e}")
