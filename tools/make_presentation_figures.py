"""Six extra figures built specifically for Presentation/present.tex.

These are additive to the 17 figures tools/make_figures.py already draws from the
result tables. They fill three gaps the presentation needs and the main figure set
does not cover on its own: what the matrices actually look like, the size of the
scipy hangs that were fixed, and two results (the staged reordering gain and the
adaptive-omega ceiling) that are easier to sell as a dedicated chart than as a line
in a table.

Reuses the exact palette and rcParams from tools/make_figures.py, so a slide with
one figure from each script still reads as one consistent deck.

    python tools/make_presentation_figures.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from solvebench import corpus, io_utils  # noqa: E402
import make_figures as mf                # noqa: E402  (palette + rcParams, reused as-is)

OUT = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"
DATA = ROOT / "dataset_large"

C1, C2, C3 = mf.C1, mf.C2, mf.C3
STATUS = mf.STATUS
SURFACE, INK, INK2, MUTED = mf.SURFACE, mf.INK, mf.INK2, mf.MUTED
GRID, AXIS = mf.GRID, mf.AXIS


def save(fig, name, caption):
    path = OUT / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path.name:<32s} {path.stat().st_size / 1024:>7.1f} KB")
    return caption


# --------------------------------------------------------------- 18: matrix gallery

def fig_matrix_gallery():
    """Four real matrices, four different domains, drawn as sparsity patterns.

    A spy plot shows only where the nonzero entries sit, ignoring their size. It is
    the standard way to show what a sparse matrix's structure looks like, and it is
    the fastest way to make the word "domain" concrete: a circuit matrix, a
    structural mesh, a fluid-dynamics discretisation and a small 2D/3D problem do
    not just differ in numbers, they differ in shape.

    All four matrices named here reappear elsewhere in the story: oscil_dcop_23 and
    bcsstk19 are two of the four matrices whose scipy hang is fixed in figure 19;
    mcca is one of the four matrices the bottleneck objective rescues that MC64 does
    not (RESULTS.md section 8).
    """
    picks = [
        ("oscil_dcop_23", "circuit simulation"),
        ("bcsstk19", "structural"),
        ("mcca", "2D/3D problem"),
        ("ex21", "fluid dynamics"),
    ]
    ents = {e["name"]: e for e in io_utils.discover_matrices(str(DATA))}

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.3))
    for ax, (name, label) in zip(axes, picks):
        e = ents[name]
        A = io_utils.load_matrix(e["path"])
        n = A.shape[0]
        coo = A.tocoo()
        ax.scatter(coo.col, coo.row, s=2.2, c=C1, marker="s", linewidths=0)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(AXIS)
        ax.grid(False)
        ax.set_title(f"{name}\n{label}", fontsize=9.5, color=INK)
        ax.set_xlabel(f"n = {n:,}   nnz = {A.nnz:,}", fontsize=8, color=INK2)

    fig.suptitle("Four domains, four different matrix shapes", fontsize=12,
                 color=INK, y=1.04)
    fig.tight_layout()
    return save(fig, "18_matrix_gallery",
                "Sparsity pattern only -- every dot is one nonzero entry, position "
                "only, no magnitude. Circuit and structural matrices show band and "
                "block structure inherited from how components are physically wired "
                "or meshed; the corpus spans 26 such domains, and no two look alike.")


# --------------------------------------------------------------- 19: hang fixes

def fig_hang_fixes():
    """The four scipy hangs, before the fix and after.

    Two of the four never returned in a bounded time -- west0067 hung inside a
    12.7-hour Kaggle session until the wall-clock limit force-cancelled it, and
    nnc261 had not returned after 240 seconds when it was killed by hand. Those two
    "before" points are marked as lower bounds (open circles, an outward-pointing
    arrow) rather than completed measurements, because that is what they are: a
    process that was still stuck when observation stopped, not a process that
    finished slowly. The other two (bcsstk19, oscil_dcop_23) did complete, just
    far too slowly to be usable at the scale of a 927-matrix sweep.

    Log scale is unavoidable here -- the smallest gap is about 1,300x, the largest
    is unmeasurable-versus-instant, and no linear axis holds both.
    """
    rows = [
        # label,           domain,                       before_s,  before_is_bound, after_s
        ("nnc261",          "2D/3D problem",              240,       True,   0.01),
        ("west0067",        "chemical process sim.",      45720,     True,   0.006),
        ("bcsstk19",        "structural",                 90,        False,  0.07),
        ("oscil_dcop_23",   "circuit simulation",          9.7,      False,  0.006),
    ]
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    ys = np.arange(len(rows))[::-1]

    def human(v):
        if v < 10:
            return f"{v:g}s"
        if v < 120:
            return f"{v:.0f}s"
        if v < 3600:
            return f"{v / 60:.0f}m" if (v / 60) % 1 == 0 else f"{v / 60:.1f}m"
        return f"{v / 3600:.1f}h" if (v / 3600) % 1 else f"{v / 3600:.0f}h"

    seen_fixed = seen_bound = seen_measured = False
    for y, (name, dom, before, is_bound, after) in zip(ys, rows):
        ax.plot([after, before], [y, y], color=AXIS, lw=1.6, zorder=1)
        ax.scatter([after], [y], s=85, color=STATUS["solved"], zorder=3,
                   label=None if seen_fixed else "fixed")
        seen_fixed = True
        if is_bound:
            # Never actually returned -- a hollow marker signals "still stuck when we
            # stopped watching", not a completed measurement.
            ax.scatter([before], [y], s=95, facecolor="none", edgecolor=C2,
                       linewidth=2.2, zorder=3,
                       label=None if seen_bound else "never returned (session ended it)")
            seen_bound = True
            btxt = f"{human(before)}+, still stuck"
        else:
            ax.scatter([before], [y], s=85, color=C2, zorder=3,
                       label=None if seen_measured else "before the fix (completed)")
            seen_measured = True
            btxt = human(before)
        ax.annotate(btxt, xy=(before, y), xytext=(0, 13), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.3, color=C2)
        ax.annotate(human(after), xy=(after, y), xytext=(0, 13),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.3,
                    color=STATUS["solved"])

    ax.set_yticks(ys)
    ax.set_yticklabels([f"{name}\n{dom}" for name, dom, *_ in rows], fontsize=9)
    ax.set_ylim(-0.6, len(rows) - 0.15)
    ax.set_xscale("log")
    ax.set_xlim(0.003, 2e5)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: human(v)))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("time for one bottleneck-selection call (log scale)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", fontsize=8.3, bbox_to_anchor=(0.5, -0.30), ncol=3,
              frameon=False)
    ax.set_title("Four scipy hangs, fixed by switching to a dense assignment", pad=12)
    fig.subplots_adjust(left=0.19, bottom=0.30, top=0.88)
    return save(fig, "19_hang_fixes",
                "west0067 and nnc261 never returned inside their measurement window -- "
                "their arrows mark a lower bound on how long they were stuck, not a "
                "completed time. bcsstk19 and oscil_dcop_23 did return, just 1,300x "
                "and 1,600x slower than the dense replacement. A fourth hang "
                "(rw5151, not shown) was caught by the local preflight check before it "
                "could cost another Kaggle session.")


# --------------------------------------------------------------- 20: staged gain

def fig_incremental_gains():
    """The reordering result as a bridge: where each stage's gain actually lands.

    Three stacked segments rather than three independent bars, specifically so the
    height of the middle segment (MC64's own contribution) is compared directly
    against the height of the top segment (what the group's own objective adds on
    top of MC64) -- the comparison the whole honesty section of RESULTS.md rests on.
    """
    stages = [("as given", 0, 317, MUTED),
              ("+ MC64", 317, 514, C2),
              ("+ bottleneck\n(ours)", 514, 516, C1)]
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for i, (label, lo, hi, colour) in enumerate(stages):
        # The third stage is a real gain of 2 out of 927, and drawn at its true
        # height it would vanish entirely -- a floating callout says what it is
        # without pretending the bar is any taller than it really is.
        if i == 2:
            ax.plot([i - 0.31, i + 0.31], [hi, hi], color=colour, lw=4, zorder=4,
                    solid_capstyle="round")
            ax.annotate("+2 (ours)", xy=(i, hi), xytext=(0, 26),
                        textcoords="offset points", ha="center", fontsize=9.5,
                        color=colour, fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color=colour, lw=1.0,
                                        shrinkA=0, shrinkB=4))
        else:
            ax.bar(i, hi - lo, bottom=lo, width=0.62, color=colour, zorder=3)
        ax.text(i, hi + 8, f"{hi}", ha="center", fontsize=11, fontweight="bold",
                color=INK, zorder=5)
        if 0 < i < 2:
            ax.plot([i - 1 + 0.31, i - 0.31], [lo, lo], color=AXIS, lw=1.1,
                    linestyle=(0, (3, 2)), zorder=2)
            ax.text(i, lo + (hi - lo) / 2, f"+{hi - lo}", ha="center", va="center",
                    fontsize=9.5, color="white" if (hi - lo) > 15 else INK,
                    fontweight="bold")
        elif i == 2:
            ax.plot([i - 1 + 0.31, i - 0.31], [514, 514], color=AXIS, lw=1.1,
                    linestyle=(0, (3, 2)), zorder=2)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([s[0] for s in stages], fontsize=9.5)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, 590)
    ax.set_ylabel("systems solved (Jacobi + Gauss-Seidel + SOR, 927 matrices)")
    ax.set_title("Choosing the diagonal: where the gain actually comes from", pad=10)
    mf._hide_grid_x(ax)
    return save(fig, "20_incremental_gains",
                "317 to 514 to 516, out of 2,781 (matrix, method) pairs across the "
                "three stationary methods. The MC64 segment is 197 systems; the "
                "group's own bottleneck objective adds 2 more on top of it. Both "
                "numbers are real and both are reported -- the idea of reordering is "
                "the large effect, the choice of objective is the small one.")


# --------------------------------------------------------------- 21: adaptive omega

def fig_adaptive_omega():
    """SOR's relaxation factor, fixed for every matrix versus chosen per matrix.

    The ceiling line is the point of the chart: 20 systems in the 400-matrix probe
    have their outcome decided by omega alone (Gauss-Seidel beats fixed SOR on 11 of
    them, fixed SOR beats Gauss-Seidel on the other 9), so no adaptive rule could
    ever recover more than 20. Reporting 8 net without that ceiling reads as a small
    number; reporting it as 8 of 20 reads as what it is, 40% of everything available.
    """
    labels = ["Gauss-Seidel\n(omega = 1)", "SOR\nfixed omega = 1.25", "SOR\nomega per matrix"]
    values = [122, 119, 127]
    colours = [MUTED, C2, C1]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    bars = ax.bar(labels, values, color=colours, width=0.56, zorder=3)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 3, str(v), ha="center",
                fontsize=11, fontweight="bold", color=INK)
    ax.set_ylim(0, 145)
    ax.set_ylabel("systems solved (400-matrix probe)")
    ax.set_title("Choosing omega per matrix instead of fixing it at 1.25", pad=10)
    mf._hide_grid_x(ax)
    fig.subplots_adjust(bottom=0.30, top=0.87)
    fig.text(0.5, 0.06,
              "20 systems in this sample have their outcome decided by omega alone.\n"
              "Adaptive omega recovers 8 of them net (16 rescued, 8 lost) -- 40% of "
              "the ceiling.",
              ha="center", va="bottom", fontsize=8.6, color=INK2)
    return save(fig, "21_adaptive_omega",
                "Fixing omega at 1.25 for every matrix is the wrong call in both "
                "directions: plain Gauss-Seidel solves 11 systems fixed SOR misses, "
                "and fixed SOR solves 9 Gauss-Seidel misses. Estimating rho(T_Jacobi) "
                "per matrix and applying Young's 1950 formula recovers 8 of the "
                "resulting 20-system ceiling.")


# --------------------------------------------------------------- 22: domain composition

def fig_domain_composition():
    """How many matrices each of the 26 domains contributes.

    Sorted, not alphabetical, so the reader's eye lands on the imbalance immediately:
    one domain has 200 matrices, eight domains have fewer than 5. Every per-domain
    claim in this project is made with that imbalance in view.
    """
    d = corpus.apply(pd.read_csv(TABLES / "benchmark_results.csv"))
    d = d[d.refinement_passes == 0].drop_duplicates("matrix")
    counts = d.domain.value_counts().sort_values(ascending=True)
    labels = [c.replace("_", " ").replace(" problem", "") for c in counts.index]

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    colours = [C1 if v >= 5 else STATUS["not_applicable"] for v in counts.values]
    ax.barh(range(len(counts)), counts.values, color=colours, height=0.68, zorder=3)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(labels, fontsize=8)
    for i, v in enumerate(counts.values):
        ax.text(v + 3, i, str(v), va="center", fontsize=7.6, color=INK2)
    ax.set_xlabel("matrices in the corpus")
    ax.set_title(f"{len(counts)} domains, {int(counts.sum())} unique matrices",
                 pad=10)
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, counts.max() * 1.14)
    return save(fig, "22_domain_composition",
                "Sorted by size. Grey bars are the 8 domains with fewer than 5 "
                "matrices -- verified against SuiteSparse's own listing to be the "
                "collection's own scarcity, not under-sampling on our part. Every "
                "per-domain rate in this project is reported with its own n for "
                "exactly this reason.")


# --------------------------------------------------------------- 23: random matrices

def fig_random_matrices():
    """The base paper's own test bed, on our 100 self-generated random matrices.

    Jacobi, Gauss-Seidel and SOR are the methods the base paper studies; a sparse
    direct solver is the ceiling. All four ran on the identical 100 matrices, sizes
    5 to 300 -- exactly the regime the base paper's own statistical validation uses.
    """
    labels = ["Jacobi", "Gauss-Seidel", "SOR", "sparse direct\n(spsolve)"]
    values = [0, 0, 0, 100]
    colours = [STATUS["not_applicable"], STATUS["not_applicable"],
               STATUS["not_applicable"], STATUS["solved"]]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    bars = ax.bar(labels, values, color=colours, width=0.6, zorder=3)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 2.5, f"{v}/100", ha="center",
                fontsize=11, fontweight="bold", color=INK)
    ax.set_ylim(0, 118)
    ax.set_ylabel("solved, out of 100 random matrices (n = 5 to 300)")
    ax.set_title("The base paper's own test bed, run at slightly larger sizes")
    mf._hide_grid_x(ax)
    return save(fig, "23_random_matrices",
                "The same kind of matrix the base paper validates its theory on, "
                "generated ourselves at sizes 5 to 300. rho(T_Jacobi) grows roughly "
                "in proportion to matrix size on random matrices, so the paper's own "
                "validation method stops working almost as soon as the matrices are "
                "made bigger than the paper itself tested.")


def main():
    print(f"writing new presentation figures to {OUT}\n")
    figs = [fig_matrix_gallery(), fig_hang_fixes(), fig_incremental_gains(),
            fig_adaptive_omega(), fig_domain_composition(), fig_random_matrices()]
    print(f"\n{len(figs)} figures written")


if __name__ == "__main__":
    main()
