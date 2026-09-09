"""Turns results/benchmark_results.csv into the "Results dashboard" from
linear.tex's methodology diagram: heatmaps, plots, and a plain-English
solver decision guide.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _method_order(df: pd.DataFrame) -> list[str]:
    direct = sorted(df[df["type"] == "direct"]["method"].unique())
    iterative = sorted(df[df["type"] == "iterative"]["method"].unique())
    return direct + iterative


def plot_error_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """Matrix x method grid, colored by log10(relative error). Grey = not
    applicable or failed to converge.
    """
    methods = _method_order(df)
    matrices = df.sort_values("n")["matrix"].unique()
    grid = np.full((len(matrices), len(methods)), np.nan)

    for i, mat in enumerate(matrices):
        for j, meth in enumerate(methods):
            sub = df[(df["matrix"] == mat) & (df["method"] == meth)]
            if len(sub) and sub.iloc[0]["status"] == "ok" and pd.notna(sub.iloc[0]["error_rel"]):
                err = sub.iloc[0]["error_rel"]
                grid[i, j] = np.log10(max(err, 1e-16))

    fig, ax = plt.subplots(figsize=(1.1 * len(methods) + 2, 0.22 * len(matrices) + 2))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_yticks(range(len(matrices)))
    ax.set_yticklabels(matrices, fontsize=6)
    ax.set_title("log10(relative error) -- grey = not applicable / did not converge")
    fig.colorbar(im, ax=ax, label="log10(relative error)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_runtime_vs_size(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in _method_order(df):
        sub = df[(df["method"] == method) & (df["status"] == "ok")].sort_values("n")
        if len(sub) == 0:
            continue
        ax.loglog(sub["n"], sub["runtime_sec"], marker="o", markersize=3, label=method)
    ax.set_xlabel("matrix size n")
    ax.set_ylabel("runtime (seconds)")
    ax.set_title("Runtime vs. matrix size (log-log)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_condition_vs_error(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in _method_order(df):
        sub = df[(df["method"] == method) & (df["status"] == "ok")]
        sub = sub[pd.notna(sub["condition_number"]) & pd.notna(sub["error_rel"])]
        if len(sub) == 0:
            continue
        ax.loglog(sub["condition_number"], sub["error_rel"].clip(lower=1e-16),
                   marker="o", linestyle="none", markersize=4, label=method, alpha=0.7)
    ax.set_xlabel("condition number")
    ax.set_ylabel("relative error")
    ax.set_title("Relative error vs. condition number")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_decision_guide(df: pd.DataFrame, out_path: Path) -> None:
    """Plain-English 'which solver, for which real system' summary."""
    lines = ["# SolveBench decision guide\n"]
    ok = df[df["status"] == "ok"].copy()

    for domain in sorted(df["domain"].unique()):
        lines.append(f"\n## {domain}\n")
        sub = ok[ok["domain"] == domain]
        if len(sub) == 0:
            lines.append("No successful solves recorded for this domain yet.\n")
            continue

        best_direct = sub[sub["type"] == "direct"].groupby("method")["runtime_sec"].mean().sort_values()
        best_iter = sub[sub["type"] == "iterative"].groupby("method")["runtime_sec"].mean().sort_values()
        most_accurate = sub.groupby("method")["error_rel"].mean().sort_values()

        if len(best_direct):
            lines.append(f"- Fastest direct method (avg runtime): **{best_direct.index[0]}**\n")
        if len(best_iter):
            lines.append(f"- Fastest iterative method (avg runtime): **{best_iter.index[0]}**\n")
        if len(most_accurate):
            lines.append(f"- Most accurate overall (avg relative error): **{most_accurate.index[0]}**\n")

        non_conv = df[(df["domain"] == domain) & (df["status"] == "did_not_converge")]
        if len(non_conv):
            failed_methods = ", ".join(sorted(non_conv["method"].unique()))
            lines.append(f"- Methods that failed to converge on at least one matrix here: {failed_methods}\n")

    out_path.write_text("".join(lines), encoding="utf-8")


def build_dashboard(results_csv: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(results_csv)
    plot_error_heatmap(df, output_dir / "heatmap_error.png")
    plot_runtime_vs_size(df, output_dir / "runtime_vs_size.png")
    plot_condition_vs_error(df, output_dir / "condition_vs_error.png")
    write_decision_guide(df, output_dir / "decision_guide.md")
    print(f"Dashboard written to {output_dir}")
