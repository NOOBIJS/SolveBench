"""Build every figure and summary table locally from the notebooks' CSVs.

Figures used to live inside the Kaggle notebook, which meant a wrong chart cost
another multi-hour run to correct -- and the first set was wrong: two of the ten
methods were missing from every figure because the plotting code carried a
hard-coded four-method list. Keeping this separate means a chart can be fixed in
seconds against data that is already on disk.

    python tools/make_figures.py [--results DIR] [--out DIR]

Runs against whatever is present, so it can be used as soon as the first
notebook finishes rather than waiting for all three.

Design notes, since these end up in a report:
* Colour is assigned by the job it does -- categorical hues for identity,
  a single-hue ramp for magnitude, status colours only for outcomes that
  genuinely mean good/bad. Outcome charts pair colour with a labelled legend so
  nothing is carried by hue alone.
* Applicability and conditional success are always plotted as two series on one
  0-1 axis, never multiplied together and never on two y-scales.
* Cost by family is drawn as small multiples rather than five overlapping
  colours, which no palette separates safely for scatter.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from solvebench import corpus

ROOT = Path(__file__).resolve().parent.parent

# --- palette -----------------------------------------------------------------
# Categorical slots in fixed order; identity never depends on rank or row order.
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
# Four real outcomes in status colours, then two neutrals. The neutrals are a full
# lightness step apart (OKLCH L about 0.62 against 0.88) so they separate under any
# colour vision; the four greys they replace did not.
STATUS = {"solved": "#0ca30c",            # good
          "inaccurate": "#d03b3b",        # critical: claimed success, wrong answer
          "did_not_converge": "#fab219",  # warning: admitted failure
          "diverged": "#ec835a",          # serious: blew up
          "not_applicable": "#8a8880",    # the method is undefined on this matrix
          "excluded": "#dcdbd3"}          # corpus/harness, says nothing about the solver

#: Reasons that are properties of the corpus or the harness rather than results.
EXCLUDED_AS = {"structurally_singular": "excluded", "skipped_too_large": "excluded",
               "load_failed": "excluded", "error": "excluded"}

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.grid": True, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 160, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 11, "legend.fontsize": 8,
})

#: Statuses meaning "the method was defined here and we let it try". Every rate in
#: every figure divides by this, never by the corpus -- dividing by the corpus is the
#: error that reported Conjugate Gradient at 10.4% when it solves 84% of what it can.
APPLICABLE = ["solved", "inaccurate", "did_not_converge", "diverged"]

FIGURES = []


def save(fig, name, caption):
    path = OUT / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    FIGURES.append({"file": path.name, "caption": caption})
    print(f"  {path.name:<44s} {path.stat().st_size/1024:>7.1f} KB")


def _hide_grid_x(ax):
    ax.grid(axis="y", visible=False)


# --------------------------------------------------------------- figures

def fig_scoreboard(res):
    """The corrected scoreboard: two rates, one axis, never multiplied.

    Reporting solved/corpus instead of solved/applicable is what put Conjugate
    Gradient at 10.4% when it solves 83.9% of the systems it is defined on.
    """
    from collections import OrderedDict
    order = list(OrderedDict.fromkeys(res["method"]))
    rows = []
    for m in order:
        sub = res[res.method == m]
        applicable = sub.status.isin(APPLICABLE).sum()
        solved = (sub.status == "solved").sum()
        rows.append((m, applicable / len(sub) if len(sub) else np.nan,
                     solved / applicable if applicable else np.nan, applicable, solved))
    d = pd.DataFrame(rows, columns=["method", "applicability", "conditional", "n_app", "n_ok"])
    d = d.sort_values("conditional", ascending=True, na_position="first")

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(d) + 1.6))
    y = np.arange(len(d))
    h = 0.36
    ax.barh(y + h / 2 + 0.02, d.applicability, height=h, color=C1, label="Applicability  (defined on / corpus)")
    ax.barh(y - h / 2 - 0.02, d.conditional, height=h, color=C2, label="Conditional success  (solved / applicable)")

    for yi, r in zip(y, d.itertuples()):
        if np.isfinite(r.conditional):
            # A long bar's label goes inside its end, so it never collides with
            # the denominator column pinned to the right margin.
            inside = r.conditional > 0.86
            ax.text(r.conditional - 0.012 if inside else r.conditional + 0.012,
                    yi - h / 2 - 0.02, f"{r.conditional:.0%}",
                    va="center", ha="right" if inside else "left",
                    fontsize=7.5, color=SURFACE if inside else INK2)
        ax.text(1.03, yi, f"n={int(r.n_app)}", va="center", fontsize=7, color=MUTED,
                transform=ax.get_yaxis_transform())

    ax.set_yticks(y, d.method, fontsize=8.5)
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    # No x-label: the ticks are already percentages and the legend names both
    # series, so a "share" caption would only collide with the legend below.
    ax.set_title("Applicability and conditional success are separate measurements",
                 loc="left", pad=12)
    # Outside the axes: with the full corpus every row carries bars, so a legend
    # placed inside would sit on top of the data.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncols=2)
    _hide_grid_x(ax)
    save(fig, "01_method_scoreboard",
         "Applicability (how much of the corpus a method is defined on) against "
         "conditional success (how much of that it solves). Multiplying the two "
         "into one rate is what understated several baselines.")


def fig_outcomes(res):
    """Every attempt accounted for, including the ones that never ran.

    Eight colour classes was one or two too many: `structurally_singular`,
    `skipped_too_large` and `load_failed` were three light greys nobody could tell
    apart. They are also the three that say nothing about the solver -- a broken file
    and a matrix with no unique solution are properties of the corpus, not results.
    They collapse into one "excluded" class, leaving six: four real outcomes,
    "not applicable" where the method is undefined, and "excluded".
    """
    from collections import OrderedDict
    order = list(OrderedDict.fromkeys(res["method"]))
    grouped = res.assign(status=res.status.map(lambda s: EXCLUDED_AS.get(s, s)))
    counts = (grouped.groupby(["method", "status"]).size().unstack(fill_value=0)
              .reindex(order))
    present = [s for s in STATUS if s in counts.columns]
    counts = counts[present]

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(counts) + 1.8))
    left = np.zeros(len(counts))
    y = np.arange(len(counts))
    for s in present:
        v = counts[s].values
        ax.barh(y, v, left=left, height=0.62, color=STATUS[s],
                edgecolor=SURFACE, linewidth=1.2, label=s.replace("_", " "))
        left += v
    ax.set_yticks(y, counts.index, fontsize=8.5)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("attempts")
    ax.set_title("Outcome of every (matrix, method) attempt", loc="left", pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncols=4)
    _hide_grid_x(ax)
    save(fig, "02_outcomes",
         "'inaccurate' is the category that did not exist before: the solver "
         "returned without complaint and the residual says the answer is wrong.")


def fig_domain_heatmap(res):
    """Conditional success per domain and method. One hue, light to dark."""
    app = res.status.isin(APPLICABLE)
    g = (res.assign(_app=app, _ok=res.status == "solved")
           .groupby(["domain", "method"])[["_app", "_ok"]].sum())
    rate = (g["_ok"] / g["_app"].replace(0, np.nan)).unstack()
    from collections import OrderedDict
    rate = rate.reindex(columns=[m for m in OrderedDict.fromkeys(res["method"])
                                 if m in rate.columns])
    rate = rate.loc[rate.notna().sum(axis=1).sort_values(ascending=False).index]

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    cmap.set_bad("#f4f3ef")
    fig, ax = plt.subplots(figsize=(0.55 * rate.shape[1] + 4.5, 0.34 * len(rate) + 2.2))
    im = ax.imshow(rate.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(rate.shape[1]), rate.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rate)), [d.replace("_problem", "") for d in rate.index], fontsize=8)
    ax.set_title("Conditional success by domain", loc="left", pad=12)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.015)
    cb.set_label("solved / applicable", color=INK2, fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=MUTED, labelcolor=MUTED, labelsize=7.5)
    save(fig, "03_domain_heatmap",
         "Grey cells are domains where a method is defined on nothing. Rates are "
         "conditional; per-domain denominators are in per_domain.csv.")


def fig_cost_profile(res):
    """Work against size, one small multiple per family.

    Five families cannot be separated safely as five scatter colours, so this is
    faceted rather than overlaid. Matrix-vector products rather than seconds:
    wall-clock on a shared machine is not reproducible.
    """
    # Iterative families only. A direct method never forms A*v -- it factorises and
    # back-substitutes -- so its matvec count is exactly 0 for every solve, and on a log
    # axis those points land on whatever floor the plot clips to. That was a straight
    # line of artefacts, not a measurement. Direct cost lives in the factorisation.
    ok = res[(res.status == "solved") & res.matvecs.notna() & (res.n > 0)
             & (res.matvecs > 0)]
    if ok.empty:
        return
    fams = [f for f in ["stationary", "krylov", "preconditioned"] if f in set(ok.family)]
    fig, axes = plt.subplots(1, len(fams), figsize=(3.2 * len(fams), 3.4),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, fams):
        s = ok[ok.family == fam]
        ax.scatter(s.n, s.matvecs, s=9, color=C1, alpha=0.5,
                   linewidths=0.4, edgecolors=SURFACE)
        ax.axhline(s.matvecs.median(), color=C2, lw=1.2, zorder=3)
        # Top-left: the bottom-right corner is dense with data in every panel.
        ax.text(0.03, 0.95, f"median {s.matvecs.median():.0f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8, color=C2)
        ax.text(0.03, 0.88, f"n={len(s)} solves",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=MUTED)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(fam, fontsize=9.5, color=INK2)
        ax.set_xlabel("matrix size n")
    axes[0].set_ylabel("matrix-vector products")
    fig.suptitle("Cost of a successful solve, iterative families only",
                 x=0.02, ha="left", fontsize=11, color=INK)
    save(fig, "04_cost_profile",
         "Counted work rather than wall-clock time, which is not reproducible on shared "
         "hardware. Direct methods are absent by construction: they perform no "
         "matrix-vector products at all, so their cost is the factorisation instead.")


def fig_conditioning(res):
    """Accuracy against conditioning, split into two strata rather than pooled."""
    ok = res[(res.status == "solved") & res.condition_number.notna()
             & res.error_rel.notna() & (res.error_rel > 0)]
    if ok.empty:
        return
    well = ok[ok.condition_number <= 1e15]
    ill = ok[ok.condition_number > 1e15]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.scatter(well.condition_number, well.error_rel, s=8, color=C1, alpha=0.45,
               linewidths=0, label=f"well-posed, cond <= 1e15  (n={len(well)})")
    ax.scatter(ill.condition_number, ill.error_rel, s=8, color=C2, alpha=0.55,
               linewidths=0, label=f"numerically singular, cond > 1e15  (n={len(ill)})")
    ax.axvline(1e15, color=AXIS, linewidth=1.0)
    ax.text(1.15e15, ax.get_ylim()[1], " double-precision limit", fontsize=7.5,
            color=MUTED, va="top")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("condition number"); ax.set_ylabel("relative forward error")
    ax.set_title("Forward error against conditioning", loc="left", pad=12)
    ax.legend(loc="lower right")
    save(fig, "05_conditioning",
         "Pooling the two strata is what let dataset ill-conditioning masquerade "
         "as solver inaccuracy in the earlier error medians.")


def fig_dispatch_ablation(res):
    """Does the symmetry dispatch rule contribute anything measurable?"""
    fams = ["ILU only", "ILU-BiCGSTAB", "ILU-GMRES(30)", "ILU-Krylov (dispatched)"]
    have = [m for m in fams if m in set(res.method)]
    if len(have) < 2:
        return
    counts = [(res[(res.method == m) & (res.status == "solved")].shape[0]) for m in have]
    colours = [C1 if m != "ILU-Krylov (dispatched)" else C2 for m in have]

    applicable = int(res[res.method == have[0]].status.isin(APPLICABLE).sum())
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    bars = ax.bar(range(len(have)), counts, width=0.6, color=colours)
    ax.set_ylim(0, max(counts) * 1.18)      # headroom so the labels clear the legend
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c}", ha="center", va="bottom",
                fontsize=9, color=INK2)
    ax.set_xticks(range(len(have)), [m.replace("ILU-", "ILU-\n") for m in have], fontsize=8.5)
    ax.set_ylabel(f"matrices solved  (of {applicable} applicable)")
    ax.set_title("The dispatch rule against always running one Krylov method", loc="left", pad=12)
    # Below the axes: inside, it sat on top of the tallest bar's value label.
    ax.legend(handles=[Patch(color=C2, label="dispatched (the rule proposed as novel)"),
                       Patch(color=C1, label="fixed choice, no dispatch")],
              loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=2)
    _hide_grid_x(ax); ax.grid(axis="x", visible=False); ax.grid(axis="y", visible=True)
    save(fig, "06_dispatch_ablation",
         "If the dispatched bar matches always-BiCGSTAB, the dispatch rule adds "
         "nothing. Earlier probes put both at 51/70.")


def fig_spectral(spec):
    """Where the spectral radii actually fall, against the rho = 1 threshold."""
    ok = spec[spec.get("status") == "analysed"] if "status" in spec else spec

    # A cumulative distribution rather than a histogram. rho(T_GS) reaches 1e77 on the
    # worst systems, and binning across eighty decades collapses everything into a
    # single spike. An ECDF lets the extreme tail simply flatten out, and the height
    # where each curve crosses rho = 1 reads directly as the fraction that converges.
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    notes = []
    for col, name, colour in (("rho_jacobi", "Jacobi", C1),
                              ("rho_gauss_seidel", "Gauss-Seidel", C2)):
        raw = pd.to_numeric(ok.get(col), errors="coerce")
        undefined = int(raw.isna().sum())        # zero diagonal: T does not exist
        finite = raw[np.isfinite(raw)].sort_values()
        if finite.empty:
            continue
        below = int((finite < 1).sum())
        # rho == 0 means a nilpotent iteration matrix: the exact answer in one step.
        # Those cannot sit on a log axis, so the curve starts at their share.
        exact = int((finite == 0).sum())
        v = finite[finite > 0]
        y = (np.arange(len(v)) + 1 + exact) / len(finite)
        ax.step(v, y, where="post", color=colour, lw=2,
                label=f"{name}   {below} of {len(finite)} below 1  ({below/len(finite):.0%})")
        ax.plot([v.iloc[0]], [(exact + 1) / len(finite)], "o", color=colour, ms=5,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        notes.append(f"{name}: {undefined} undefined (zero diagonal)"
                     + (f", {exact} with $\\rho=0$" if exact else ""))

    ax.axvline(1.0, color="#d03b3b", lw=1.4)
    ax.text(1.15, 0.03, r"$\rho=1$", color="#d03b3b", fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1e4)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    # Say what the clipped window hides rather than letting the curve run off.
    for col, name, colour, dy in (("rho_jacobi", "Jacobi", C1, 0.0),
                                  ("rho_gauss_seidel", "Gauss-Seidel", C2, 0.06)):
        raw = pd.to_numeric(ok.get(col), errors="coerce")
        f = raw[np.isfinite(raw)]
        beyond = int((f > 1e4).sum())
        if beyond:
            ax.text(0.985, 0.30 - dy, f"{beyond} {name} systems beyond $10^4$",
                    transform=ax.transAxes, ha="right", fontsize=7.5, color=colour)
    ax.set_xlabel(r"spectral radius $\rho(T)$")
    ax.set_ylabel("share of systems with $\\rho$ at or below x")
    ax.legend(loc="upper left")
    ax.text(0.02, -0.20, "   ·   ".join(notes), transform=ax.transAxes,
            fontsize=7.5, color=MUTED, va="top")
    fig.suptitle("Convergence is decided by whether rho falls left of 1",
                 x=0.02, ha="left", fontsize=11, color=INK)
    save(fig, "07_spectral_radius",
         "The quantity the base paper exists to characterise, computed here for "
         "every matrix in the corpus.")


def fig_hypothesis_coverage(spec):
    """Which classical theorem covers each matrix -- nominal, so one colour."""
    if "hypothesis_class" not in spec:
        return
    counts = spec.hypothesis_class.dropna().value_counts()
    if counts.empty:
        return
    fig, ax = plt.subplots(figsize=(6.8, 0.42 * len(counts) + 1.8))
    y = np.arange(len(counts))[::-1]
    ax.barh(y, counts.values, height=0.6, color=C1)
    for yi, v in zip(y, counts.values):
        ax.text(v, yi, f" {v}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, counts.index, fontsize=9)
    ax.set_xlabel("matrices")
    ax.set_title("Hypothesis-class coverage across the corpus", loc="left", pad=12)
    _hide_grid_x(ax)
    save(fig, "08_hypothesis_coverage",
         "Stein-Rosenberg (1948) forbids Jacobi-converges-while-Gauss-Seidel-does-not "
         "for M-matrices; Householder-John (1958) guarantees Gauss-Seidel for SPD. "
         "Coverage of these classes is what explains the benchmark's own headline.")


def fig_prediction(res, spec):
    """Theory against observation, including the third case theory cannot state."""
    if "jacobi_verdict" not in spec:
        return
    rows = []
    for meth, col in (("Jacobi", "jacobi_verdict"), ("Gauss-Seidel", "gs_verdict")):
        sub = res[res.method == meth].set_index("matrix")
        # Same restriction as the base-paper figure: a method that was never defined
        # on a matrix has no observation there, and counting it as "did not solve"
        # would blame the solver for a system it was never given.
        sub = sub[sub.status.isin(APPLICABLE)]
        obs = sub.status == "solved"
        pred = spec.set_index("matrix")[col]
        joined = pd.concat([pred.rename("pred"), obs.rename("obs")], axis=1).dropna()
        for verdict in ("converges", "too_slow", "diverges"):
            s = joined[joined.pred == verdict]
            if len(s):
                # astype(bool) matters. pd.concat widens a boolean column to int, and
                # "~" is then bitwise negation rather than logical: ~1 == -2. Without
                # this the "not solved" bars were negative, reaching -300.
                solved = int(s.obs.astype(bool).sum())
                rows.append((meth, verdict, solved, len(s) - solved))
    if not rows:
        return
    d = pd.DataFrame(rows, columns=["method", "verdict", "solved", "not_solved"])

    # One panel per method and a stacked bar per verdict. Side-by-side pairs were
    # ambiguous: where one bar is near zero the pair visually breaks apart and the
    # label stops obviously belonging to either. Stacking makes the bar height the
    # number of matrices predicted that way, and the split inside it the outcome.
    methods = list(dict.fromkeys(d.method))
    fig, axes = plt.subplots(1, len(methods), figsize=(4.4 * len(methods), 4.0),
                             sharey=True)
    axes = np.atleast_1d(axes)
    order = ["converges", "too_slow", "diverges"]
    for ax, meth in zip(axes, methods):
        sub = d[d.method == meth].set_index("verdict").reindex(order).fillna(0)
        x = np.arange(len(order))
        ax.bar(x, sub.solved, width=0.55, color=C1, label="observed: solved")
        ax.bar(x, sub.not_solved, width=0.55, bottom=sub.solved, color=C2,
               edgecolor=SURFACE, linewidth=1.2, label="observed: not solved")
        span = d.groupby("method")[["solved", "not_solved"]].sum().sum(axis=1).max()
        for xi, r in zip(x, sub.itertuples()):
            total = r.solved + r.not_solved
            for value, base, colour in ((r.solved, 0, C1),
                                        (r.not_solved, r.solved, C2)):
                if not value:
                    continue
                # A sliver cannot hold a label. Below a few percent of the axis the
                # number goes beside the bar in the segment's own colour instead of
                # inside it in white, where it was invisible.
                if value < 0.045 * span:
                    ax.text(xi + 0.31, base + value / 2, f"{int(value)}", ha="left",
                            va="center", fontsize=7.5, color=colour)
                else:
                    ax.text(xi, base + value / 2, f"{int(value)}", ha="center",
                            va="center", fontsize=8, color=SURFACE)
            ax.text(xi, total, f"n={int(total)}", ha="center", va="bottom",
                    fontsize=7.5, color=MUTED)
        ax.set_xticks(x, order, fontsize=9)
        ax.set_title(meth, fontsize=10, color=INK2, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("matrices  (where the method is defined)")
    axes[0].set_ylim(0, d.groupby("method")[["solved", "not_solved"]].sum().sum(axis=1).max() * 0.85)
    fig.suptitle("Predicted verdict against observed outcome", x=0.02, ha="left",
                 fontsize=11, color=INK)
    axes[-1].legend(loc="upper right", fontsize=8)
    save(fig, "09_prediction_vs_observation",
         "'too_slow' is the case the textbook criterion cannot express: rho < 1, so "
         "convergence is guaranteed, but not within any usable iteration budget.")


def fig_refinement(study):
    """Refinement given to every method, not to one."""
    ok = study[study.status == "solved"]
    if ok.empty:
        return
    err = ok.pivot_table(index="method", columns="refinement_passes",
                         values="error_rel", aggfunc="median").dropna(how="all")
    cost = ok.pivot_table(index="method", columns="refinement_passes",
                          values="matvecs", aggfunc="median").reindex(err.index)
    if err.empty:
        return

    # Dots, not bars. A bar encodes its value by length measured from zero, and a log
    # axis has no zero -- so the bars' lengths were set by wherever the axis happened to
    # be cut, not by the numbers. Position reads correctly on a log scale; length does
    # not. The two panels put the trade-off side by side: refinement buys accuracy and
    # is paid for in work.
    cols = sorted(err.columns)
    order = err[cols[1]].sort_values(ascending=False).index if len(cols) > 1 else err.index
    err, cost = err.reindex(order), cost.reindex(order)

    fig, axes = plt.subplots(1, 2, figsize=(11, 0.40 * len(err) + 2.2), sharey=True)
    y = np.arange(len(err))
    for ax, table, xlabel in ((axes[0], err, "median relative forward error"),
                              (axes[1], cost, "median matrix-vector products")):
        for yi, row in zip(y, table.itertuples(index=False)):
            vals = [v for v in row if np.isfinite(v) and v > 0]
            if len(vals) > 1:
                ax.plot([min(vals), max(vals)], [yi, yi], color=GRID, lw=1.4, zorder=1,
                        solid_capstyle="round")
        for c, colour in zip(cols, [C1, C2, C3]):
            v = table[c] if c in table else None
            if v is None:
                continue
            m = np.isfinite(v) & (v > 0)
            ax.scatter(v[m], y[m.values], s=44, color=colour, zorder=3,
                       edgecolors=SURFACE, linewidths=1.4,
                       label=f"{c} pass{'es' if c != 1 else ''}")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, err.index, fontsize=8.5)
    axes[0].invert_xaxis()          # better to the right in both panels
    axes[0].set_title("accuracy  (further right is better)", fontsize=9, color=INK2, loc="left")
    axes[1].set_title("cost  (further left is better)", fontsize=9, color=INK2, loc="left")
    fig.suptitle("Refinement applied to every method, on equal terms",
                 x=0.02, ha="left", fontsize=11, color=INK)
    axes[1].legend(loc="upper center", bbox_to_anchor=(-0.05, -0.09), ncols=3)
    save(fig, "10_refinement_effect",
         "Refinement applied to every method, not to one. Given the same wrapper, Jacobi "
         "becomes the most accurate method in the benchmark and the dispatched ILU-Krylov "
         "does not -- but Jacobi pays roughly twenty times the work for it, which is why "
         "cost is plotted beside accuracy rather than left to a footnote. In the first "
         "sweep only one method received refinement, and that is where its accuracy "
         "advantage came from.")


def fig_base_paper_test(res, spec):
    """The base paper's own question, asked of real matrices."""
    jac = res[res.method == "Jacobi"].set_index("matrix")
    gs = res[res.method == "Gauss-Seidel"].set_index("matrix")
    # Restrict to systems where BOTH methods are actually defined. Comparing on the
    # whole corpus instead puts the 491 systems with a zero diagonal into "neither
    # converges", which is the very denominator error this project is about.
    applicable = jac.status.isin(APPLICABLE) & gs.status.isin(APPLICABLE)
    d = pd.concat([(jac.status == "solved").rename("J"),
                   (gs.status == "solved").rename("G")], axis=1)[applicable].dropna()
    if d.empty:
        return
    counts = [int((d.J & d.G).sum()), int((~d.J & d.G).sum()),
              int((d.J & ~d.G).sum()), int((~d.J & ~d.G).sum())]
    labels = ["both\nconverge", "only\nGauss-Seidel", "only\nJacobi", "neither"]
    colours = [C1, C1, C2, C1]

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    bars = ax.bar(range(4), counts, width=0.6, color=colours)
    ax.set_ylim(0, max(counts) * 1.22)          # headroom so labels clear the top
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c}", ha="center", va="bottom",
                fontsize=9.5, color=INK2)
    # The "only Jacobi" bar is zero, so a legend swatch for it would explain a colour
    # nothing on the chart shows -- and it collided with the tallest bar's value.
    # Annotate the empty column directly instead.
    ax.annotate("Stein-Rosenberg (1948)\nforbids this for M-matrices",
                xy=(2, 0), xytext=(2, max(counts) * 0.42), ha="center", fontsize=7.5,
                color=C2, arrowprops=dict(arrowstyle="-", color=C2, lw=1.0,
                                          shrinkA=2, shrinkB=16))
    ax.set_xticks(range(4), labels, fontsize=8.5)
    ax.set_ylabel(f"matrices  (n={len(d)} where both are defined)")
    ax.set_title("The base paper's comparison, re-run on real matrices", loc="left", pad=12)
    ax.grid(axis="x", visible=False)
    save(fig, "11_jacobi_vs_gauss_seidel",
         "The denominator is matrices where both methods are defined, not the "
         "whole corpus -- Jacobi and Gauss-Seidel are undefined wherever the "
         "diagonal carries a zero.")


# --------------------------------------------------------------- driver

def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results" / "output" / "tables"))
    ap.add_argument("--out", default=str(ROOT / "results" / "figures"))
    args = ap.parse_args()

    tables = Path(args.results)
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    def load(name):
        p = tables / name
        if not p.exists():
            print(f"  (skipped: {name} not present yet)")
            return None
        df = pd.read_csv(p)
        # Merge the split CFD domain and drop the three exact duplicates before
        # anything is plotted, so no figure double-counts a matrix or splits a domain.
        fixed = corpus.apply(df)
        if len(fixed) != len(df):
            print(f"  {name}: {len(df) - len(fixed)} duplicate rows removed")
        return fixed

    print(f"reading {tables}")
    res = load("benchmark_results.csv")
    spec = load("spectral.csv")
    study = load("refinement_study.csv")

    print(f"\nwriting figures to {OUT}")
    if res is not None:
        fig_scoreboard(res)
        fig_outcomes(res)
        fig_domain_heatmap(res)
        fig_cost_profile(res)
        fig_conditioning(res)
        fig_dispatch_ablation(res)
        fig_base_paper_test(res, spec)
    if spec is not None:
        fig_spectral(spec)
        fig_hypothesis_coverage(spec)
        if res is not None:
            fig_prediction(res, spec)
    if study is not None:
        fig_refinement(study)

    if FIGURES:
        index = "\n".join(f"- `{f['file']}` -- {f['caption']}" for f in FIGURES)
        (OUT / "index.md").write_text(f"# Figures\n\n{index}\n", encoding="utf-8")
        print(f"\n  {len(FIGURES)} figures + index.md")
    else:
        print("\n  nothing produced -- no input tables found")


if __name__ == "__main__":
    main()
