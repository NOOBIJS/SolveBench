# run every generated notebook locally on a few matrices before it goes anywhere near Kaggle
import ast
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"


def check_grammar(path, version=(3, 12)):
    # parse against Kaggle's actual python version (3.12.13, measured) not the local one
    nb = json.loads(path.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        try:
            ast.parse(src, feature_version=version)
        except SyntaxError as e:
            print(f"\n  {path.name} cell {i}: needs a newer Python than "
                  f"{version[0]}.{version[1]}")
            print(f"    line {e.lineno}: {e.msg}")
            return False
    return True


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

        if "MATRICES" in ns and len(ns["MATRICES"]) > n_matrices:  # shrink corpus post-load
            ns["MATRICES"] = ns["MATRICES"][:n_matrices]
            ns["SAMPLE_SIZE"] = n_matrices
            print(f"  [test] corpus reduced to {n_matrices} matrices")
    return True


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    notebooks = sorted(NB_DIR.glob("solvebench-*/solvebench-*.ipynb"))
    if not notebooks:
        raise SystemExit("no notebooks found -- run tools/build_notebooks.py first")

    results = {}
    for path in notebooks:
        print(f"\n{'=' * 70}\n{path.name}\n{'=' * 70}")
        results[path.name] = check_grammar(path) and run_notebook(path, n)

    print(f"\n{'=' * 70}")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("=" * 70)
    if not all(results.values()):
        raise SystemExit(1)
    print("All notebooks execute. Safe to push.")


if __name__ == "__main__":
    main()
