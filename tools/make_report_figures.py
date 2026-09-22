"""Five figures for the written report that no earlier script had reason to draw.

Each one visualises a finding from the 2026-09-16 pre-submission audit
(docs/AUDIT_2026-09-16.md) that exists in that document only as prose and a
plain-text table. They reuse the exact palette and rcParams of
tools/make_figures.py and tools/make_presentation_figures.py, so a report page
carrying one of these next to an older figure still reads as one document.

    python tools/make_report_figures.py
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

from solvebench import corpus  # noqa: E402
import make_figures as mf       # noqa: E402

OUT = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"

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


# --------------------------------------------------------------- 26: the ILU ladder audit

def fig_ilu_ladder_audit():
    """What actually happened when the 166-matrix hole was checked against stock spilu.

    The point of this figure is the contrast between the three bars on the left,
    which are what the project's own retry ladder achieved, and the two bars on
    the right, which are what a correctly ordered ladder achieves with no
    permutation at all. The gap between them is the bug, not a property of ILU.
    """
    settings = [
        ("ours_1\n(1e-3, 5)", 0, MUTED),
        ("ours_2\n(1e-2, 10)", 0, MUTED),
        ("ours_3\n(1e-2, 20)", 0, MUTED),
        ("permc_natural\n(1e-4, 20)", 14, MUTED),
        ("full_pivot\n(1e-3, 10)", 9, MUTED),
        ("permc_mmd\n(1e-4, 20)", 44, MUTED),
        ("low_drop\n(1e-6, 20)", 113, C2),
        ("no_drop\n(0, 50)", 150, C1),
    ]
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    xs = np.arange(len(settings))
    for x, (label, val, colour) in zip(xs, settings):
        ax.bar(x, val, width=0.6, color=colour, zorder=3)
        ax.text(x, val + 3, str(val), ha="center", fontsize=10, fontweight="bold",
                color=INK)
    ax.axhline(166, color=AXIS, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(len(settings) - 0.5, 166, " all 166 matrices", va="bottom", ha="right",
            fontsize=8, color=MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels([s[0] for s in settings], fontsize=8.3)
    ax.set_ylim(0, 178)
    ax.set_ylabel("of 166 matrices, a usable ILU factorization\n(no permutation applied)")
    ax.set_title("The 166-matrix hole, checked against stock spilu settings")
    mf._hide_grid_x(ax)
    fig.subplots_adjust(bottom=0.22)
    return save(fig, "26_ilu_ladder_audit",
                "The project's own retry ladder (grey, left three bars) never varied "
                "drop_tol in the direction that helps: dropping more entries makes a "
                "zero pivot likelier, not less, so all three of its settings factor "
                "zero of the 166. A correctly ordered ladder factors 150 with drop_tol "
                "= 0 and no permutation at all (blue), and 113 with a genuinely "
                "incomplete factorization at drop_tol = 1e-6 (orange). The real hole "
                "is 16 matrices, about a tenth of what was originally reported.")


# --------------------------------------------------------------- 27: threshold blindness

def fig_bottleneck_threshold():
    """Why beating MC64 365-0 on the row ratio converts only two systems.

    Left: the wins, binned by how large MC64's own ratio already was. Right: the
    same wins as a before/after pair on a log scale, with the ratio = 1 threshold
    marked. The story is that every winning matrix sits so far above the line that
    a one-third reduction in the ratio changes nothing about whether the bound
    can certify convergence.
    """
    bins = [("2 to 10", 20, C1), ("10 to 1e3", 175, C2), ("above 1e3", 170, C3)]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.2),
                                  gridspec_kw={"width_ratios": [1, 1.15]})

    xs = np.arange(len(bins))
    vals = [b[1] for b in bins]
    ax.bar(xs, vals, width=0.55, color=[b[2] for b in bins], zorder=3)
    for x, v in zip(xs, vals):
        ax.text(x, v + 3, str(v), ha="center", fontsize=11, fontweight="bold",
                color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bins], fontsize=9)
    ax.set_xlabel("MC64's own worst row ratio, among the 365 matrices\nbottleneck beats it on")
    ax.set_ylabel("matrices")
    ax.set_ylim(0, 200)
    ax.set_title("Every win starts far above the threshold", fontsize=11)
    mf._hide_grid_x(ax)

    d = pd.read_csv(TABLES / "reordering_study.csv")
    one = d[d.condition == "best"].drop_duplicates("matrix")
    a = pd.to_numeric(one.ratio_mc64, errors="coerce")
    c = pd.to_numeric(one.ratio_bottleneck, errors="coerce")
    m = np.isfinite(a) & np.isfinite(c) & (a > 0) & (c > 0) & (c < a)
    a, c = a[m], c[m]
    rng = np.random.default_rng(3)
    idx = rng.choice(len(a), size=min(220, len(a)), replace=False)
    ax2.plot([a.min() * 0.5, a.max() * 2], [a.min() * 0.5, a.max() * 2],
             color=AXIS, lw=1.0, zorder=1)
    ax2.scatter(a.values[idx], c.values[idx], s=10, color=C1, alpha=0.55,
               edgecolors="none", zorder=3)
    ax2.axvline(1.0, color=STATUS["inaccurate"], lw=1.1, ls=(0, (4, 3)), zorder=2)
    ax2.axhline(1.0, color=STATUS["inaccurate"], lw=1.1, ls=(0, (4, 3)), zorder=2)
    ax2.text(1.15, ax2.get_ylim()[1] if False else 1.4, "ratio = 1", color=STATUS["inaccurate"],
             fontsize=8, rotation=0, va="bottom")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("worst row ratio under MC64")
    ax2.set_ylabel("under bottleneck")
    ax2.set_title("Median 597 to 368: real, and nowhere near 1", fontsize=11)

    fig.suptitle("The bottleneck objective wins 365-0 on a bound the data never approaches",
                 fontsize=12.5, y=1.03)
    fig.tight_layout()
    return save(fig, "27_bottleneck_threshold",
                "Left: MC64's own worst row ratio on the 365 matrices where bottleneck "
                "achieves a strictly smaller one. Right: the same matrices, MC64's ratio "
                "against bottleneck's, both axes logarithmic, with the ratio = 1 "
                "convergence threshold marked. Every point sits orders of magnitude to "
                "the right and above that line. A 35.8% median reduction in a "
                "continuous quantity does nothing to a bound that only fires at a "
                "single point, which is why 365 wins convert two systems rather than "
                "hundreds.")


# --------------------------------------------------------------- 28: rho search

def fig_rho_search():
    """Targeting rho directly instead of a bound on rho: how far it reaches, and
    how much of that reach survives contact with the harness.

    Left: the distribution of the power-iteration rho estimate before and after
    the swap search, on a log axis, with the crossing threshold marked. Right:
    what happened when the 22 predicted crossings were checked against an actual
    solve, which is the step a spectral estimate cannot substitute for.
    """
    p = pd.read_csv(TABLES / "rho_search_probe.csv")
    v = pd.read_csv(TABLES / "rho_search_verified.csv")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.2),
                                  gridspec_kw={"width_ratios": [1.15, 1]})

    before = np.sort(p.rho_portfolio.values)
    after = np.sort(p.rho_searched.values)
    ax.plot(before, np.arange(1, len(before) + 1) / len(before), color=MUTED, lw=1.8,
            label="before the search (portfolio)")
    ax.plot(after, np.arange(1, len(after) + 1) / len(after), color=C1, lw=1.8,
            label="after the search")
    ax.axvline(1.0, color=STATUS["inaccurate"], lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("estimated rho(T_GS)  (dashed: rho = 1)")
    ax.set_ylabel("fraction of 195 matrices at or below")
    ax.set_title("Descent moves the median from 1.30 to 1.01", fontsize=11)
    ax.legend(loc="upper left", fontsize=8.5)

    stages = ["predicted\ncrossings", "actually\nsolved"]
    vals = [22, 2]
    cols = [C2, STATUS["solved"]]
    ax2.bar(stages, vals, width=0.45, color=cols, zorder=3)
    for i, val in enumerate(vals):
        ax2.text(i, val + 0.5, str(val), ha="center", fontsize=13, fontweight="bold",
                 color=INK)
    ax2.set_ylim(0, 26)
    ax2.set_ylabel("matrices")
    ax2.set_title("A predicted crossing is not a solved system", fontsize=11)
    mf._hide_grid_x(ax2)

    fig.suptitle("Targeting rho directly reaches further, and still mostly does not land",
                 fontsize=12.3, y=1.03)
    fig.tight_layout()
    return save(fig, "28_rho_search",
                f"Left: the estimated rho(T_GS) for {len(p)} matrices before and after a "
                "pairwise-swap descent seeded from the portfolio's own choice, both axes "
                "logarithmic. The search lowers rho on 154 of 195 and pushes 22 from "
                "above 1 to below it. Right: only 2 of those 22 predicted crossings turn "
                "into an actual solve within the iteration budget once checked against "
                "the harness, one of them a synthetic 5x5. The crossings land at 0.97 to "
                "0.9996, asymptotically guaranteed but unreachable at the tolerance and "
                "cap this project uses, and the 40-iteration power estimate cannot "
                "resolve rho precisely enough near 1 to be trusted as a threshold test.")


# --------------------------------------------------------------- 29: applicability vs convergence

def fig_decomposition():
    """Decomposing the pipeline's 199 recovered systems by whether the method was
    even defined on that matrix before reordering ran.

    This is the figure the whole methodological story compresses into: three
    independent attempts to improve convergence (bottleneck, rho search, min-sum)
    each move only a handful of systems, while the same 199 that motivated the
    project mostly turn out to be about applicability rather than convergence at
    all.
    """
    methods = ["Jacobi", "Gauss-Seidel", "SOR"]
    empty_before = [44, 75, 63]
    full_before = [4, 11, 2]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    xs = np.arange(len(methods))
    b1 = ax.bar(xs, empty_before, width=0.55, color=C1, zorder=3,
               label="diagonal empty before (applicability)")
    b2 = ax.bar(xs, full_before, width=0.55, bottom=empty_before, color=C2, zorder=3,
               label="diagonal full before (convergence)")
    for x, e, f in zip(xs, empty_before, full_before):
        ax.text(x, e / 2, str(e), ha="center", va="center", fontsize=10,
                color="white", fontweight="bold")
        ax.text(x, e + f + 2, str(f), ha="center", va="bottom", fontsize=9.5,
                color=INK2)
        ax.text(x, e + f + 9, str(e + f), ha="center", va="bottom", fontsize=11,
                color=INK, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_ylabel("systems recovered by the pipeline")
    ax.set_title("Where the 199 recovered systems actually came from")
    ax.legend(loc="upper right", fontsize=8.7)
    mf._hide_grid_x(ax)
    return save(fig, "29_decomposition",
                "199 systems recovered across the three stationary methods, split by "
                "whether the diagonal was already usable before reordering ran. 182 of "
                "199 (91%) had a zero diagonal beforehand and any perfect matching "
                "would have supplied one; only 17 (9%) already had a full diagonal and "
                "were made to converge by the choice of permutation. The headline "
                "percentages stay true and stay measured; this figure is what their "
                "cause looks like.")


# --------------------------------------------------------------- 30: the near-1 wall

def fig_theorem_wall():
    """Classical convergence theorems, scored on the population they can actually be
    checked on, sorted by how much of the corpus each one covers.

    Denominators here exclude structurally singular and load-failed matrices, since
    a system with no solution or no data cannot be attributed to the solver at all.
    The point of the ordering is the cliff at the right-hand end: coverage and
    Gauss-Seidel's success rate fall together, and 679 matrices carry no applicable
    theorem and solve at 3%.
    """
    rows = [
        ("strict diagonal\ndominance", 43, 88),
        ("weak diagonal\ndominance", 46, 83),
        ("H-matrix", 83, 81),
        ("M-matrix", 34, 76),
        ("L-matrix", 38, 71),
        ("SPD", 102, 51),
        ("property A\n(proxy)", 117, 29),
        ("symmetric", 298, 22),
        ("no theorem\napplies", 679, 3),
    ]
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    xs = np.arange(len(rows))
    colours = [C1] * (len(rows) - 1) + [STATUS["inaccurate"]]
    bars = ax.bar(xs, [r[2] for r in rows], width=0.6, color=colours, zorder=3)
    for x, (label, n, pct) in zip(xs, rows):
        ax.text(x, pct + 1.8, f"{pct}%", ha="center", fontsize=10, fontweight="bold",
                color=INK)
        ax.text(x, -6, f"n={n}", ha="center", va="top", fontsize=7.8, color=MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], fontsize=8.4)
    ax.set_ylim(-14, 100)
    ax.set_ylabel("Gauss-Seidel success rate, of matrices\nthe theorem applies to")
    ax.set_title("Coverage and success fall together: the corpus's largest class carries no guarantee")
    ax.grid(axis="y", visible=True)
    ax.grid(axis="x", visible=False)
    ax.axhline(0, color=AXIS, lw=0.8)
    return save(fig, "30_theorem_wall",
                "Gauss-Seidel's success rate, restricted each time to the solvable "
                "matrices a given classical sufficient condition actually applies to, "
                "ordered from the strongest guarantee to none. n is the size of each "
                "class; a matrix can belong to several. Where some theorem applies "
                "(223 solvable matrices, the union of the first eight classes), "
                "Gauss-Seidel solves 46%; where none applies (679 matrices), it solves "
                "3%. Every theorem here is correct on its own terms: the five strictly "
                "diagonally dominant matrices that still fail sit at rho(T_GS) above "
                "0.99, guaranteed to converge and unreachable within budget regardless.")


def main():
    print(f"writing new report figures to {OUT}\n")
    figs = [fig_ilu_ladder_audit(), fig_bottleneck_threshold(), fig_rho_search(),
            fig_decomposition(), fig_theorem_wall()]
    print(f"\n{len(figs)} figures written")


if __name__ == "__main__":
    main()
