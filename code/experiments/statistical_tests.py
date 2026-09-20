"""
Part 12, Experiment 13 / Table XV: statistical reliability.

Computes, from a long-format results CSV with columns [dataset, model, seed,
<metric>], per-(dataset, metric): mean +/- std, 95% CI (normal approx.,
appropriate given >=5 seeds), and, for the winning method vs each baseline,
a paired t-test and Wilcoxon signed-rank test (seed-paired) with Holm
correction across the family of comparisons, plus Cohen's d effect size.
Never fabricates a comparison for a (baseline, dataset) pair that lacks a
matching seed-for-seed pairing in the CSV -- such pairs are skipped and
reported as "insufficient paired seeds" rather than silently dropped.
"""
import numpy as np
import pandas as pd
from scipy import stats


def mean_ci(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    m = values.mean()
    s = values.std(ddof=1) if n > 1 else 0.0
    if n > 1:
        h = s / np.sqrt(n) * stats.t.ppf((1 + confidence) / 2, n - 1)
    else:
        h = np.nan
    return m, s, m - h, m + h


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    pooled_std = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (a.mean() - b.mean()) / pooled_std


def holm_correction(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvalues[idx]
        running_max = max(running_max, val)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def summarize_by_dataset_model(df, metric, group_cols=("dataset", "model")):
    rows = []
    for keys, g in df.groupby(list(group_cols)):
        m, s, lo, hi = mean_ci(g[metric].values)
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update({"metric": metric, "mean": m, "std": s, "ci_lo": lo, "ci_hi": hi, "n_seeds": len(g)})
        rows.append(row)
    return pd.DataFrame(rows)


def paired_tests_vs_winner(df, metric, winner_name, dataset, model_col="model", seed_col="seed"):
    """Paired (by seed) t-test + Wilcoxon between the winner and every other model
    present for this dataset/metric, with Holm correction across the resulting
    family. Returns a DataFrame; pairs without matching seeds on both sides are
    reported with a note rather than silently skipped."""
    sub = df[df["dataset"] == dataset]
    winner_rows = sub[sub[model_col] == winner_name].set_index(seed_col)[metric]
    results = []
    other_models = [m for m in sub[model_col].unique() if m != winner_name]
    raw_p_t, raw_p_w = [], []
    valid_idx = []
    for m in other_models:
        other_rows = sub[sub[model_col] == m].set_index(seed_col)[metric]
        common_seeds = winner_rows.index.intersection(other_rows.index)
        if len(common_seeds) < 3:
            results.append({"baseline": m, "dataset": dataset, "metric": metric,
                           "note": "insufficient paired seeds (<3)", "n_paired_seeds": len(common_seeds)})
            continue
        a = winner_rows.loc[common_seeds].values
        b = other_rows.loc[common_seeds].values
        t_stat, p_t = stats.ttest_rel(a, b)
        try:
            w_stat, p_w = stats.wilcoxon(a, b)
        except ValueError:
            w_stat, p_w = np.nan, np.nan
        d = cohens_d(a, b)
        results.append({"baseline": m, "dataset": dataset, "metric": metric,
                       "n_paired_seeds": len(common_seeds), "winner_mean": a.mean(), "baseline_mean": b.mean(),
                       "t_stat": t_stat, "p_ttest": p_t, "wilcoxon_stat": w_stat, "p_wilcoxon": p_w,
                       "cohens_d": d})
        raw_p_t.append(p_t); raw_p_w.append(p_w); valid_idx.append(len(results) - 1)

    if raw_p_t:
        adj_t = holm_correction(raw_p_t)
        adj_w = holm_correction([p if not np.isnan(p) else 1.0 for p in raw_p_w])
        for k, idx in enumerate(valid_idx):
            results[idx]["p_ttest_holm"] = adj_t[k]
            results[idx]["p_wilcoxon_holm"] = adj_w[k]
    return pd.DataFrame(results)
