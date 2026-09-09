"""Corrections applied to the corpus at analysis time.

Two defects were found in the downloaded collection. Both are fixed here rather than by
re-uploading the 985 MB dataset, so the raw files, the Kaggle runs and every table stay
consistent and the correction is visible in one place instead of buried in a download
script nobody reads.

**One domain arrived split in two.** Folders were named by slugifying SuiteSparse's
``kind`` field, and that field spells the same physical domain several ways --
``computational fluid dynamics problem``, ``subsequent computational fluid dynamics
problem``, ``computational fluid dynamics problem sequence``. The grouping folded those
together correctly everywhere, but the Goodwin group carries the kind ``computational
fluid dynamics`` with no "problem" suffix at all, so its four matrices landed in a folder
of their own. Every other domain in the corpus checks out.

**Three matrices are exact duplicates of three others.** SuiteSparse marks them with a
``duplicate ...`` kind, and comparing the CSR structure and values byte for byte confirms
it: they are the same matrix under a second name, so every result for them was counted
twice. Two further matrices carry a duplicate kind but their originals are not in this
corpus, so they are unique here and are kept.
"""

#: Folders that are the same domain under different SuiteSparse spellings.
DOMAIN_ALIASES = {
    "computational_fluid_dynamics": "computational_fluid_dynamics_problem",
}

#: name -> the matrix it duplicates. Verified by SHA-256 over indptr, indices and data.
DUPLICATES = {
    "bcsstk07": "bcsstk06",     # 420 x 420, 7860 nnz
    "bcsstk12": "bcsstk11",     # 1473 x 1473, 34241 nnz
    "t2dal_a": "t2dal",         # 4257 x 4257, 37465 nnz
}

#: Marked duplicate by SuiteSparse but with no counterpart in this corpus, so kept.
KEPT_DESPITE_DUPLICATE_KIND = ("nasa1824", "t2dal_e")

#: Below this a per-domain rate is not a statistic. Reported, never silently dropped.
SMALL_DOMAIN = 5


def canonical_domain(domain):
    return DOMAIN_ALIASES.get(domain, domain)


def apply(df, drop_duplicates=True):
    """Return `df` with domains merged and duplicated matrices removed.

    Works on any table carrying a ``domain`` and/or ``matrix`` column, so the same
    correction covers the benchmark results, the spectral table and the studies.
    """
    out = df.copy()
    if "domain" in out.columns:
        out["domain"] = out["domain"].map(canonical_domain)
    if drop_duplicates and "matrix" in out.columns:
        out = out[~out["matrix"].isin(DUPLICATES)]
    return out


def summary(df):
    """What the correction changed, for reporting alongside the numbers."""
    matrices = set(df["matrix"]) if "matrix" in df.columns else set()
    return {
        "matrices_before": len(matrices),
        "duplicates_removed": sorted(matrices & set(DUPLICATES)),
        "matrices_after": len(matrices - set(DUPLICATES)),
        "domains_merged": {k: v for k, v in DOMAIN_ALIASES.items()
                           if "domain" in df.columns and k in set(df["domain"])},
    }
