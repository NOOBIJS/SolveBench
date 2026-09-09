"""Entry point: run the full SolveBench sweep (49 matrices x 8 methods)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from solvebench.benchmark import run_full_benchmark  # noqa: E402
from solvebench.visualize import build_dashboard  # noqa: E402

if __name__ == "__main__":
    project_root = Path(__file__).parent
    dataset_root = project_root / "dataset"
    output_csv = project_root / "results" / "benchmark_results.csv"
    run_full_benchmark(dataset_root, output_csv)
    build_dashboard(output_csv, project_root / "results")
