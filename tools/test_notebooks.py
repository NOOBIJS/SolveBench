"""Execute every generated notebook locally against a handful of matrices.

Nothing goes to Kaggle until this passes. Two full sweeps have already been lost
to faults that only appeared at run time -- an empty ``/kaggle/input`` because a
freshly created dataset had not propagated, and a structurally singular matrix
that got the kernel OS-killed -- and both would have shown up here.

This runs the real cells in order, not a paraphrase of them, so a cell that
fails to parse or references a name that no longer exists fails the test.

    python tools/test_notebooks.py [n_matrices]
"""
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "kaggle_upload"


def run_notebook(path, n_matrices):
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    ns = {"__name__": "__main__"}

    for i, cell in enumerate(cells):
        source = "".join(cell["source"])
        try:
            exec(compile(source, f"{path.name}#cell{i}", "exec"), ns)
        except Exception:
            print(f"\n  FAILED at cell {i} of {path.name}")
            print("  " + "\n  ".join(traceback.format_exc().splitlines()[-6:]))
            return False

        # Straight after the data cell, cut the corpus down so the sweep cells
        # exercise every code path without running the actual benchmark.
        if "MATRICES" in ns and len(ns["MATRICES"]) > n_matrices:
            ns["MATRICES"] = ns["MATRICES"][:n_matrices]
            ns["SAMPLE_SIZE"] = n_matrices
            print(f"  [test] corpus reduced to {n_matrices} matrices")
    return True


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    notebooks = sorted(NB_DIR.glob("solvebench-*.ipynb"))
    if not notebooks:
        raise SystemExit("no notebooks found -- run tools/build_notebooks.py first")

    results = {}
    for path in notebooks:
        print(f"\n{'=' * 70}\n{path.name}\n{'=' * 70}")
        results[path.name] = run_notebook(path, n)

    print(f"\n{'=' * 70}")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("=" * 70)
    if not all(results.values()):
        raise SystemExit(1)
    print("All notebooks execute. Safe to push.")


if __name__ == "__main__":
    main()
