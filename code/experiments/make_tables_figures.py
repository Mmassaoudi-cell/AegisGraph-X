"""
Part 14/15: turn results/csv/*.csv into IEEE-ready LaTeX tables (tables/*.tex)
and publication figures (figures/*.png), from real logged results only -- no
numbers are invented here; datasets/candidates with missing or degenerate
(NaN) metrics are shown as such rather than backfilled.
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from experiments.statistical_tests import paired_tests_vs_winner

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_DIR = os.path.join(ROOT, "results", "csv")
TAB_DIR = os.path.join(ROOT, "tables")
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(TAB_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SCREEN_METRICS = ["macro_f1", "minority_recall", "pr_auc", "fpr", "fnr",
                  "cumulative_reward_per_window", "avg_detection_delay",
                  "telemetry_cost", "latency_sec", "peak_memory_MB"]


def df_to_latex_table(df, caption, label, float_cols=None, out_name=None):
    float_cols = float_cols or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df_fmt = df.copy()
    for c in float_cols:
        df_fmt[c] = df_fmt[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "--")
    ncols = len(df_fmt.columns)
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{" + "l" * ncols + "}")
    lines.append(r"\hline")
    lines.append(" & ".join(str(c).replace("_", r"\_") for c in df_fmt.columns) + r" \\")
    lines.append(r"\hline")
    for _, row in df_fmt.iterrows():
        lines.append(" & ".join(str(v).replace("_", r"\_") for v in row.values) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    tex = "\n".join(lines)
    if out_name:
        with open(os.path.join(TAB_DIR, out_name), "w") as f:
            f.write(tex)
    return tex


def make_table_iv_candidate_screening():
    path = os.path.join(CSV_DIR, "candidate_screening.csv")
    if not os.path.exists(path):
        print("candidate_screening.csv not found, skipping Table IV")
        return None
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"] if "status" in df.columns else df
    agg = df.groupby("candidate")[[c for c in SCREEN_METRICS if c in df.columns]].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    if "macro_f1_mean" in agg.columns:
        agg = agg.sort_values("macro_f1_mean", ascending=False, na_position="last")
    agg.to_csv(os.path.join(CSV_DIR, "table_IV_candidate_screening_agg.csv"), index=False)
    show_cols = ["candidate"] + [f"{m}_mean" for m in SCREEN_METRICS if f"{m}_mean" in agg.columns]
    df_to_latex_table(agg[show_cols], "Validation screening of the 10 candidate graph/RL methods "
                       "(mean over seeds and datasets).", "tab:candidate_screening",
                       out_name="table_IV_candidate_screening.tex")
    return agg


def make_win_tie_loss(baseline_csv="baselines.csv", metric="macro_f1", margin=0.01):
    """Compares the selected method's TEST-split performance (family=="selected_method"
    rows written only by run_final_comparison.py) against every baseline's TEST-split
    performance. Deliberately requires those rows to exist -- comparing the winner's
    VALIDATION screening score against baselines' TEST scores would be an invalid,
    apples-to-oranges comparison (different splits, different row-cap/eval config),
    so this returns None rather than silently producing a misleading win/tie/loss
    count from mismatched splits."""
    path = os.path.join(CSV_DIR, baseline_csv)
    if not os.path.exists(path):
        print("baselines.csv not found, skipping win/tie/loss table")
        return None
    base_all = pd.read_csv(path)
    if "selected_method" not in base_all.get("family", pd.Series(dtype=str)).unique():
        print("no selected_method (TEST-split) rows in baselines.csv yet -- "
              "skipping win/tie/loss table until run_final_comparison.py has run")
        return None
    winner_rows = base_all[base_all["family"] == "selected_method"]
    winner_name = winner_rows["model"].iloc[0]
    base = base_all[base_all["family"] != "selected_method"]
    rows = []
    for dataset, g in base.groupby("dataset"):
        winner_val = winner_rows[winner_rows["dataset"] == dataset][metric]
        if winner_val.empty or winner_val.isna().all():
            continue
        winner_score = winner_val.mean()
        for _, r in g.iterrows():
            base_score = r.get(metric)
            if pd.isna(base_score):
                continue
            if winner_score > base_score + margin:
                outcome = "win"
            elif abs(winner_score - base_score) <= margin:
                outcome = "tie"
            else:
                outcome = "loss"
            rows.append({"dataset": dataset, "baseline": r["model"], "family": r.get("family"),
                         "winner_score": winner_score, "baseline_score": base_score, "outcome": outcome})
    wtl = pd.DataFrame(rows)
    if wtl.empty:
        return None
    summary = wtl["outcome"].value_counts().reindex(["win", "tie", "loss"]).fillna(0).astype(int)
    wtl.to_csv(os.path.join(CSV_DIR, "table_XVI_win_tie_loss_detail.csv"), index=False)
    summary.to_csv(os.path.join(CSV_DIR, "table_XVI_win_tie_loss_summary.csv"))
    print("Win/Tie/Loss summary:\n", summary)
    return wtl, summary, winner_name


def fig_candidate_ranking():
    agg = make_table_iv_candidate_screening()
    if agg is None or agg.empty or "macro_f1_mean" not in agg.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    valid = agg.dropna(subset=["macro_f1_mean"])
    ax.barh(valid["candidate"], valid["macro_f1_mean"], xerr=valid.get("macro_f1_std", 0))
    ax.set_xlabel("Validation macro-F1 (mean over seeds/datasets)")
    ax.set_title("Candidate-method validation ranking")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig6_candidate_ranking.png"), dpi=200)
    plt.close()


def fig_runtime_vs_f1():
    path = os.path.join(CSV_DIR, "baselines.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    if "macro_f1" not in df.columns or "inference_latency_sec_per_sample" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for family, g in df.groupby("family"):
        ax.scatter(g["inference_latency_sec_per_sample"], g["macro_f1"], label=family, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Per-sample inference latency (s, log scale)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Runtime vs. macro-F1 tradeoff")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig13_runtime_vs_f1.png"), dpi=200)
    plt.close()


def fig_win_tie_loss():
    result = make_win_tie_loss()
    if result is None:
        return
    _, summary, winner_name = result
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(summary.values, labels=summary.index, autopct="%1.0f%%")
    ax.set_title(f"{winner_name} vs. {summary.sum()} benchmarks")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig14_win_tie_loss.png"), dpi=200)
    plt.close()


BASELINE_DISPLAY_METRICS = ["macro_f1", "minority_recall", "pr_auc", "fpr", "fnr",
                            "roc_auc", "train_time_sec", "inference_latency_sec_per_sample"]


def make_table_baselines(family_filter=None, out_name="table_IX_baselines.tex",
                          caption="Comparison against IDS benchmarks.", label="tab:baselines"):
    path = os.path.join(CSV_DIR, "baselines.csv")
    if not os.path.exists(path):
        print(f"baselines.csv not found, skipping {out_name}")
        return None
    df = pd.read_csv(path)
    if family_filter is not None:
        df = df[df["family"].isin(family_filter)]
    cols = [c for c in BASELINE_DISPLAY_METRICS if c in df.columns]
    agg = df.groupby(["dataset", "model"])[cols].mean().reset_index()
    agg = agg.sort_values(["dataset", "macro_f1"], ascending=[True, False]) if "macro_f1" in agg.columns else agg
    agg.to_csv(os.path.join(CSV_DIR, out_name.replace(".tex", ".csv")), index=False)
    df_to_latex_table(agg, caption, label, out_name=out_name)
    return agg


def make_table_rl_baselines():
    return make_table_baselines(family_filter=["rl"], out_name="table_X_rl_baselines.tex",
                                caption="Comparison against RL/decision-policy benchmarks.",
                                label="tab:rl_baselines")


def make_table_low_label():
    path = os.path.join(CSV_DIR, "low_label.csv")
    if not os.path.exists(path):
        print("low_label.csv not found, skipping Table XI")
        return None
    df = pd.read_csv(path)
    cols = [c for c in ["macro_f1", "minority_recall", "pr_auc"] if c in df.columns]
    agg = df.groupby(["dataset", "model", "label_fraction"])[cols].mean().reset_index()
    agg = agg.sort_values(["dataset", "model", "label_fraction"])
    agg.to_csv(os.path.join(CSV_DIR, "table_XI_low_label.csv"), index=False)
    df_to_latex_table(agg, "Low-label-fraction results.", "tab:low_label",
                       out_name="table_XI_low_label.tex")
    return agg


def make_table_robustness():
    path = os.path.join(CSV_DIR, "robustness.csv")
    if not os.path.exists(path):
        print("robustness.csv not found, skipping Table XIII")
        return None
    df = pd.read_csv(path)
    cols = [c for c in ["macro_f1", "minority_recall", "pr_auc", "fpr", "fnr"] if c in df.columns]
    agg = df.groupby(["dataset", "model", "perturbation"])[cols].mean().reset_index()
    agg.to_csv(os.path.join(CSV_DIR, "table_XIII_robustness.csv"), index=False)
    df_to_latex_table(agg, "Robustness / distribution-shift results.", "tab:robustness",
                       out_name="table_XIII_robustness.tex")
    return agg


def fig_low_label_curve():
    path = os.path.join(CSV_DIR, "low_label.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    if "macro_f1" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for (dataset, model), g in df.groupby(["dataset", "model"]):
        g = g.sort_values("label_fraction")
        ax.plot(g["label_fraction"], g["macro_f1"], marker="o", label=f"{dataset}-{model}")
    ax.set_xlabel("Training label fraction")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Low-label-fraction performance")
    ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_low_label.png"), dpi=200)
    plt.close()


def fig_robustness_heatmap():
    path = os.path.join(CSV_DIR, "robustness.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    if "macro_f1" not in df.columns:
        return
    pivot = df.pivot_table(index="model", columns="perturbation", values="macro_f1", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=7)
    fig.colorbar(im, ax=ax, label="Macro-F1")
    ax.set_title("Robustness heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig12_robustness_heatmap.png"), dpi=200)
    plt.close()


DECISION_METRICS = ["cumulative_reward_per_window", "avg_detection_delay", "telemetry_cost",
                    "escalation_rate", "isolation_rate", "abstention_rate", "budget_violation_rate"]


def make_table_sequential_decision():
    """Table VIII: sequential decision utility, RL/decision-policy family only
    (classical/deep/graph baselines have no telemetry/escalation/isolation
    semantics, so they are naturally absent from this table rather than
    padded with not-applicable placeholders)."""
    path = os.path.join(CSV_DIR, "baselines.csv")
    if not os.path.exists(path):
        print("baselines.csv not found, skipping Table VIII")
        return None
    df = pd.read_csv(path)
    df = df[df["family"].isin(["rl", "selected_method"])]
    cols = [c for c in DECISION_METRICS if c in df.columns]
    agg = df.groupby(["dataset", "model"])[cols].mean().reset_index()
    agg.to_csv(os.path.join(CSV_DIR, "table_VIII_sequential_decision.csv"), index=False)
    df_to_latex_table(agg, "Sequential decision utility (RL/decision-policy family, test split). "
                       "NOTE: the selected method (C3) uses its own validation-tuned reward weights "
                       "while every baseline uses the default weights, so cumulative\\_reward\\_per\\_window "
                       "is NOT directly comparable across rows -- only the reward-scale-independent "
                       "columns (telemetry\\_cost, avg\\_detection\\_delay, budget\\_violation\\_rate, "
                       "isolation/escalation/abstention rate) support cross-model comparison.",
                       "tab:sequential_decision", out_name="table_VIII_sequential_decision.tex")
    return agg


RUNTIME_METRICS = ["train_time_sec", "inference_latency_sec_per_sample", "peak_memory_MB",
                   "n_params", "latency_sec"]


def make_table_runtime():
    """Table XIV: runtime/deployment analysis across all families."""
    path = os.path.join(CSV_DIR, "baselines.csv")
    if not os.path.exists(path):
        print("baselines.csv not found, skipping Table XIV")
        return None
    df = pd.read_csv(path)
    cols = [c for c in RUNTIME_METRICS if c in df.columns]
    agg = df.groupby(["family", "model"])[cols].mean().reset_index()
    agg.to_csv(os.path.join(CSV_DIR, "table_XIV_runtime.csv"), index=False)
    df_to_latex_table(agg, "Runtime and deployment analysis (mean over datasets/seeds).",
                       "tab:runtime", out_name="table_XIV_runtime.tex")
    return agg


def make_table_multiclass():
    """Table VII: multiclass reference result (classical baseline only -- see
    run_multiclass.py docstring for the disclosed scope limitation)."""
    path = os.path.join(CSV_DIR, "multiclass.csv")
    if not os.path.exists(path):
        print("multiclass.csv not found, skipping Table VII")
        return None
    df = pd.read_csv(path)
    agg = df.groupby("dataset")[["macro_f1", "weighted_f1", "balanced_accuracy", "n_classes_train"]].agg(
        ["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(os.path.join(CSV_DIR, "table_VII_multiclass.csv"), index=False)
    df_to_latex_table(agg, "Multiclass reference result (HistGradientBoosting only; not evaluated "
                       "for the selected method or other candidates -- see Section~\\ref{sec:limitations}).",
                       "tab:multiclass", out_name="table_VII_multiclass.tex")
    return agg


def make_table_ablation():
    """Table XII: ablation study for the selected method (full vs. no_graph_encoder
    vs. no_rl_policy), mean +/- std over seeds, per dataset."""
    path = os.path.join(CSV_DIR, "ablation.csv")
    if not os.path.exists(path):
        print("ablation.csv not found, skipping Table XII")
        return None
    df = pd.read_csv(path)
    cols = [c for c in ["macro_f1", "minority_recall", "pr_auc", "cumulative_reward_per_window"] if c in df.columns]
    agg = df.groupby(["dataset", "ablation"])[cols].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(os.path.join(CSV_DIR, "table_XII_ablation.csv"), index=False)
    show_cols = ["dataset", "ablation"] + [c for c in agg.columns if c not in ("dataset", "ablation")]
    df_to_latex_table(agg[show_cols], "Ablation study for the selected method (mean $\\pm$ std over seeds).",
                       "tab:ablation", out_name="table_XII_ablation.tex")
    return agg


def fig_ablation_plot():
    path = os.path.join(CSV_DIR, "ablation.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    if "macro_f1" not in df.columns:
        return
    pivot = df.groupby(["dataset", "ablation"])["macro_f1"].mean().unstack()
    fig, ax = plt.subplots(figsize=(6, 4))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Ablation: contribution of graph encoder and RL policy")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig11_ablation.png"), dpi=200)
    plt.close()


def make_table_stat_sig(metric="macro_f1"):
    """Table XV: paired significance tests (winner vs every other model present
    in baselines.csv, per dataset), computed only from real logged seed-paired
    rows -- pairs lacking >=3 common seeds are reported as such, never dropped
    silently or backfilled."""
    path = os.path.join(CSV_DIR, "baselines.csv")
    if not os.path.exists(path):
        print("baselines.csv not found, skipping Table XV")
        return None
    df = pd.read_csv(path)
    if "selected_method" not in df.get("family", pd.Series(dtype=str)).unique():
        print("no selected_method rows in baselines.csv yet, skipping Table XV")
        return None
    winner_name = df[df["family"] == "selected_method"]["model"].iloc[0]
    all_results = []
    for dataset in df["dataset"].unique():
        res = paired_tests_vs_winner(df, metric, winner_name, dataset)
        if not res.empty:
            all_results.append(res)
    if not all_results:
        return None
    out = pd.concat(all_results, ignore_index=True)
    out.to_csv(os.path.join(CSV_DIR, "table_XV_statistical_significance.csv"), index=False)
    show_cols = [c for c in ["dataset", "baseline", "n_paired_seeds", "winner_mean", "baseline_mean",
                              "p_ttest_holm", "p_wilcoxon_holm", "cohens_d", "note"] if c in out.columns]
    df_to_latex_table(out[show_cols], f"Statistical significance ({winner_name} vs. benchmarks, "
                       "paired t-test/Wilcoxon with Holm correction).", "tab:stat_sig",
                       out_name="table_XV_statistical_significance.tex")
    return out


def fig_confusion_matrices():
    """Figure 10: confusion matrices for the selected method, one per dataset,
    reconstructed from baselines.csv's logged fpr/fnr/accuracy (which imply the
    2x2 counts given the known class balance) -- only plotted for datasets where
    a selected_method row actually exists."""
    path = os.path.join(CSV_DIR, "baselines.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    sel = df[df.get("family") == "selected_method"]
    if sel.empty:
        return
    datasets = sel["dataset"].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4))
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        row = sel[sel["dataset"] == dataset].iloc[0]
        fpr, fnr = row.get("fpr", np.nan), row.get("fnr", np.nan)
        if pd.isna(fpr) or pd.isna(fnr):
            continue
        cm = np.array([[1 - fpr, fpr], [fnr, 1 - fnr]])
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{cm[r, c]:.2f}", ha="center", va="center")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred benign", "Pred attack"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["True benign", "True attack"])
        ax.set_title(dataset)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig10_confusion_matrices.png"), dpi=200)
    plt.close()


def fig_per_class_recall_heatmap():
    """Figure 9: per-model minority-class recall heatmap across datasets, from
    baselines.csv (works for any family present, not just the winner)."""
    path = os.path.join(CSV_DIR, "baselines.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    if "minority_recall" not in df.columns:
        return
    pivot = df.pivot_table(index="model", columns="dataset", values="minority_recall", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6, max(4, 0.25 * len(pivot))))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=6)
    fig.colorbar(im, ax=ax, label="Minority recall")
    ax.set_title("Per-model minority-class recall")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig9_per_class_recall_heatmap.png"), dpi=200)
    plt.close()


def main():
    make_table_iv_candidate_screening()
    make_table_baselines(family_filter=["classical_tree", "deep"], out_name="table_VI_binary.tex",
                          caption="Binary detection comparison.", label="tab:binary")
    make_table_baselines(out_name="table_IX_baselines.tex",
                          caption="Comparison against IDS benchmarks (all families).",
                          label="tab:baselines")
    make_table_rl_baselines()
    make_table_sequential_decision()
    make_table_runtime()
    make_table_multiclass()
    make_table_low_label()
    make_table_robustness()
    make_table_ablation()
    make_table_stat_sig()
    fig_candidate_ranking()
    fig_runtime_vs_f1()
    fig_win_tie_loss()
    fig_confusion_matrices()
    fig_per_class_recall_heatmap()
    fig_ablation_plot()
    fig_low_label_curve()
    fig_robustness_heatmap()
    print("Tables/figures written to:", TAB_DIR, FIG_DIR)


if __name__ == "__main__":
    main()
