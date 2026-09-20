"""
Generates AegisGraph-X-specific IEEE LaTeX tables and figures from the CSVs
produced by run_aegis.py, run_aegis_select.py, append_aegis_to_baselines.py,
run_aegis_statistics.py, run_aegis_reward_modes.py, run_aegis_robustness.py,
run_aegis_low_label.py, and run_aegis_calibration.py. Reuses the existing
legacy tables/figures (dataset screening, split protocol, legacy POMDP,
multiclass reference) unchanged; only the AegisGraph-X-era tables/figures
are (re)written here.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE, "results", "csv")
TABLES_DIR = os.path.join(BASE, "tables")
FIGURES_DIR = os.path.join(BASE, "figures")
MANIFESTS_DIR = os.path.join(BASE, "manifests")
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DATASETS = ["UNSW-NB15", "ToN-IoT-Network", "CICIoT2023"]
STRONGEST_TREE_PER_DATASET = {
    "UNSW-NB15": "XGBoost", "ToN-IoT-Network": "LightGBM", "CICIoT2023": "GradientBoosting",
}


def esc(s):
    return str(s).replace("_", "\\_")


def write(path, content):
    with open(path, "w") as f:
        f.write(content)


def table_selection():
    path = os.path.join(MANIFESTS_DIR, "aegis_candidate_selection_report.json")
    with open(path) as f:
        report = json.load(f)
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             r"\caption{Validation-only selection among Candidates A/B/C (Section~\ref{sec:results_selection}).}",
             r"\label{tab:aegis_selection}", r"\begin{tabular}{lcccc}", r"\hline",
             r"Candidate & Mean val. macro-F1 & Min. minority recall & Gate & Gated score \\", r"\hline"]
    for name, info in report["candidates"].items():
        pretty = {"A_TreeGuidedGraphMoE": "A: backbone (no RL)", "B_BudgetedRLInvestigation": "B: + RL investigation",
                  "C_SafetyShieldedRL": "C: + safety shield"}[name]
        marker = " (selected)" if name == report["winner"] else ""
        lines.append(f"{pretty}{marker} & {info['mean_macro_f1']:.4f} & {info['min_minority_recall']:.4f} & "
                      f"{info['gate']:.3f} & {info['gated_score']:.4f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    write(os.path.join(TABLES_DIR, "table_aegis_selection.tex"), "\n".join(lines))


def table_v_binary_detection():
    """AegisGraph-X's (Candidate A, winner per table_aegis_selection.tex) own
    binary detection performance, from the 10-seed row-cap-pinned re-run --
    this is a self-comparison (no baseline pairing needed), so it uses the
    fuller 10-seed data even though Table VIII's paired baseline comparison
    stays at 3 seeds (see make_table_v_10seed.py / table_viii_strongest_baseline
    docstrings for why the two adjacent tables intentionally differ)."""
    src = os.path.join(RESULTS_DIR, "aegis_candidates_v2_10seed.csv")
    if not os.path.exists(src):
        src = os.path.join(RESULTS_DIR, "aegis_candidates.csv")
    df = pd.read_csv(src)
    df = df[(df["split"] == "test") & (df["candidate"] == "A_TreeGuidedGraphMoE")]
    n_seeds = df["seed"].nunique()
    metrics = ["macro_f1", "balanced_accuracy", "minority_recall", "fpr", "fnr", "roc_auc", "pr_auc"]
    pretty = {"macro_f1": "Macro-F1", "balanced_accuracy": "Balanced acc.", "minority_recall": "Minority recall",
              "fpr": "FPR", "fnr": "FNR", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             r"\caption{AegisGraph-X (Candidate A) binary detection performance "
             r"(TEST split, mean$\pm$std over %d seeds, row-cap-pinned).}" % n_seeds,
             r"\label{tab:aegis_binary}", r"\begin{tabular}{lccc}", r"\hline",
             r"Metric & UNSW-NB15 & ToN-IoT-Network & CICIoT2023 \\", r"\hline"]
    for metric in metrics:
        vals = []
        for ds in DATASETS:
            sub = df[df["dataset"] == ds][metric]
            vals.append(f"{sub.mean():.4f}$\\pm${sub.std():.4f}")
        lines.append(f"{pretty[metric]} & " + " & ".join(vals) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    write(os.path.join(TABLES_DIR, "table_V_aegis_binary.tex"), "\n".join(lines))


def table_viii_strongest_baseline():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "baselines.csv"))
    df = df[df["eval_split"] == "test"]
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             r"\caption{AegisGraph-X vs.\ the strongest tree/boosting baseline per dataset (TEST split, macro-F1, "
             r"mean$\pm$std, 3 seeds -- paired against the 37-baseline suite, which was not re-run at 10 seeds; "
             r"see Table~\ref{tab:aegis_binary} for AegisGraph-X's own 10-seed figures).}",
             r"\label{tab:aegis_vs_strongest}", r"\begin{tabular}{lccc}", r"\hline",
             r"Dataset & Strongest tree baseline & AegisGraph-X & Statistically \\",
             r" & (model, macro-F1) & (macro-F1) & tied / superior / inferior \\", r"\hline"]
    stat_df = pd.read_csv(os.path.join(RESULTS_DIR, "aegis_statistical_tests.csv"))
    for ds in DATASETS:
        strongest = STRONGEST_TREE_PER_DATASET[ds]
        sub_b = df[(df["dataset"] == ds) & (df["model"] == strongest)]
        sub_a = df[(df["dataset"] == ds) & (df["model"] == "43_AegisGraphX")]
        b_mean = sub_b["macro_f1"].mean()
        a_mean, a_std = sub_a["macro_f1"].mean(), sub_a["macro_f1"].std()
        row = stat_df[(stat_df["dataset"] == ds) & (stat_df["baseline"] == strongest)]
        label = row["label"].iloc[0] if len(row) else "n/a"
        lines.append(f"{esc(ds)} & {esc(strongest)}, {b_mean:.4f} & {a_mean:.4f}$\\pm${a_std:.4f} & {esc(label)} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    write(os.path.join(TABLES_DIR, "table_VIII_aegis_vs_strongest.tex"), "\n".join(lines))


def table_ix_x_reward_modes():
    path = os.path.join(RESULTS_DIR, "aegis_reward_modes.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    df = df[df["split"] == "test"]
    for mode, tabname, label, cap in [
        ("Mode1_CommonDefaultReward", "table_IX_aegis_common_reward.tex", "tab:aegis_common_reward",
         "Sequential decision utility, Mode 1: common default reward weights (Section~\\ref{sec:aegis_rl})."),
        ("Mode2_ValidationTunedReward", "table_X_aegis_tuned_reward.tex", "tab:aegis_tuned_reward",
         "Sequential decision utility, Mode 2: validation-tuned reward weights per dataset (Section~\\ref{sec:aegis_rl})."),
    ]:
        sub = df[df["mode"] == mode]
        lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize", f"\\caption{{{cap}}}", f"\\label{{{label}}}",
                 r"\begin{tabular}{lccccc}", r"\hline",
                 r"Dataset & Macro-F1 & Minority recall & Telemetry cost & Escalation rate & Isolation rate \\",
                 r"\hline"]
        for ds in DATASETS:
            s = sub[sub["dataset"] == ds]
            if len(s) == 0:
                continue
            lines.append(f"{esc(ds)} & {s['macro_f1'].mean():.4f}$\\pm${s['macro_f1'].std():.4f} & "
                          f"{s['minority_recall'].mean():.4f} & {s['telemetry_cost'].mean():.4f} & "
                          f"{s['escalation_rate'].mean():.4f} & {s['isolation_rate'].mean():.4f} \\\\")
        lines += [r"\hline",
                  r"\multicolumn{6}{p{0.9\linewidth}}{\footnotesize Raw cumulative reward is not compared "
                  r"across Mode 1 / Mode 2 or across datasets, since reward weights differ between them "
                  r"(Section~\ref{sec:aegis_rl}); only these reward-scale-independent columns support "
                  r"cross-mode/cross-dataset comparison.}\\",
                  r"\end{tabular}", r"\end{table*}"]
        write(os.path.join(TABLES_DIR, tabname), "\n".join(lines))


def table_xi_ablation():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "aegis_ablation.csv"))
    df = df[df["split"] == "test"]
    order = ["Full_AegisGraphX_backbone", "No_ExpertEnsemble_BestSingleExpert", "No_MoERouter_EqualWeight",
             "No_GraphBranch", "No_TemporalBranch", "No_Calibration_Raw0.5", "Full_DistilledStudent",
             "No_FocalLoss", "No_ContrastiveLearning", "No_HardClassMining", "PlainKD_NoHardClassExtras"]
    pretty = {"Full_AegisGraphX_backbone": "Full AegisGraph-X backbone (Candidate A)",
              "No_ExpertEnsemble_BestSingleExpert": "No expert ensemble (best single expert)",
              "No_MoERouter_EqualWeight": "No MoE router (equal-weight average)",
              "No_GraphBranch": "No graph branch", "No_TemporalBranch": "No temporal branch",
              "No_Calibration_Raw0.5": "No calibration (raw @ 0.5)",
              "Full_DistilledStudent": "Full distilled student",
              "No_FocalLoss": "\\quad No focal loss", "No_ContrastiveLearning": "\\quad No contrastive learning",
              "No_HardClassMining": "\\quad No hard-class mining",
              "PlainKD_NoHardClassExtras": "\\quad Plain KD (no hard-class extras)"}
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             r"\caption{Component ablation (TEST split, macro-F1 and minority recall averaged over 3 datasets $\times$ 3 seeds).}",
             r"\label{tab:aegis_ablation}", r"\begin{tabular}{lcc}", r"\hline",
             r"Component & Macro-F1 & Minority recall \\", r"\hline"]
    for comp in order:
        sub = df[df["component"] == comp]
        if len(sub) == 0:
            continue
        lines.append(f"{pretty[comp]} & {sub['macro_f1'].mean():.4f}$\\pm${sub['macro_f1'].std():.4f} & "
                      f"{sub['minority_recall'].mean():.4f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    write(os.path.join(TABLES_DIR, "table_XI_aegis_ablation.tex"), "\n".join(lines))

    fig, ax = plt.subplots(figsize=(7, 4))
    names = [pretty[c].replace("\\quad ", "  ") for c in order if c in df["component"].unique()]
    means = [df[df["component"] == c]["macro_f1"].mean() for c in order if c in df["component"].unique()]
    ax.barh(range(len(names)), means, color="#4C72B0")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Macro-F1 (test, mean over 3 datasets x 3 seeds)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_aegis_ablation.png"), dpi=150)
    plt.close(fig)


def table_xii_low_label():
    path = os.path.join(RESULTS_DIR, "aegis_low_label.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             r"\caption{AegisGraph-X low-label regime (TEST macro-F1, seed 0).}",
             r"\label{tab:aegis_low_label}", r"\begin{tabular}{lcccc}", r"\hline",
             r"Dataset & 1\% & 10\% & 50\% & 100\% \\", r"\hline"]
    for ds in DATASETS:
        sub = df[df["dataset"] == ds].sort_values("label_fraction")
        vals = []
        for frac in [0.01, 0.10, 0.50, 1.00]:
            row = sub[np.isclose(sub["label_fraction"], frac)]
            vals.append(f"{row['macro_f1'].iloc[0]:.4f}" if len(row) else "--")
        lines.append(f"{esc(ds)} & " + " & ".join(vals) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    write(os.path.join(TABLES_DIR, "table_XII_aegis_low_label.tex"), "\n".join(lines))

    fig, ax = plt.subplots(figsize=(6, 4))
    for ds in DATASETS:
        sub = df[df["dataset"] == ds].sort_values("label_fraction")
        if len(sub):
            ax.plot(sub["label_fraction"] * 100, sub["macro_f1"], marker="o", label=ds)
    ax.set_xlabel("% training labels used"); ax.set_ylabel("Test macro-F1"); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_aegis_low_label.png"), dpi=150)
    plt.close(fig)


def table_xiii_robustness():
    path = os.path.join(RESULTS_DIR, "aegis_robustness.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    perts = df["perturbation"].unique().tolist()
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             r"\caption{AegisGraph-X robustness under distribution-shift perturbations (TEST macro-F1, mean over 2 seeds).}",
             r"\label{tab:aegis_robustness}", r"\begin{tabular}{lccc}", r"\hline",
             r"Perturbation & UNSW-NB15 & ToN-IoT-Network & CICIoT2023 \\", r"\hline"]
    for p in perts:
        vals = []
        for ds in DATASETS:
            sub = df[(df["dataset"] == ds) & (df["perturbation"] == p)]
            vals.append(f"{sub['macro_f1'].mean():.4f}" if len(sub) else "--")
        lines.append(f"{esc(p)} & " + " & ".join(vals) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    write(os.path.join(TABLES_DIR, "table_XIII_aegis_robustness.tex"), "\n".join(lines))

    pivot = df.pivot_table(index="perturbation", columns="dataset", values="macro_f1", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=8)
    plt.colorbar(im, label="Macro-F1")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_aegis_robustness_heatmap.png"), dpi=150)
    plt.close(fig)


def table_xiv_calibration():
    path = os.path.join(RESULTS_DIR, "aegis_calibration.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             r"\caption{Calibration comparison (TEST split, seed 0, mean over 3 datasets).}",
             r"\label{tab:aegis_calibration}", r"\begin{tabular}{lcccc}", r"\hline",
             r"Calibration & ECE & Brier & NLL & Alerts/10K flows \\", r"\hline"]
    for cal in df["calibration"].unique():
        sub = df[df["calibration"] == cal]
        lines.append(f"{esc(cal)} & {sub['ece'].mean():.4f} & {sub['brier'].mean():.4f} & "
                      f"{sub['nll'].mean():.4f} & {sub['alerts_per_10k_flows'].mean():.1f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    write(os.path.join(TABLES_DIR, "table_XIV_aegis_calibration.tex"), "\n".join(lines))

    rel_path = os.path.join(RESULTS_DIR, "aegis_reliability.csv")
    if os.path.exists(rel_path):
        rel = pd.read_csv(rel_path)
        fig, axes = plt.subplots(1, len(DATASETS), figsize=(4 * len(DATASETS), 4), sharey=True)
        for ax, ds in zip(axes, DATASETS):
            sub = rel[rel["dataset"] == ds]
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
            ax.plot(sub["confidence"], sub["accuracy"], marker="o")
            ax.set_title(ds, fontsize=9); ax.set_xlabel("Confidence")
        axes[0].set_ylabel("Accuracy")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "fig_aegis_reliability.png"), dpi=150)
        plt.close(fig)


def table_xv_runtime():
    src = os.path.join(RESULTS_DIR, "aegis_runtime_v2_10seed.csv")
    if not os.path.exists(src):
        src = os.path.join(RESULTS_DIR, "aegis_runtime.csv")
    df = pd.read_csv(src)
    n_seeds = df["seed"].nunique() if "seed" in df.columns else 3
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             r"\caption{Runtime/deployment comparison (mean over 3 datasets $\times$ %d seeds).}" % n_seeds,
             r"\label{tab:aegis_runtime}", r"\begin{tabular}{lccc}", r"\hline",
             r"Model & Inference latency (s/sample) & Train time (s) & Params \\", r"\hline"]
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        lat = sub["inference_latency_sec_per_sample"].mean()
        train_t = sub["train_time_sec"].mean()
        params = sub["n_params"].mean()
        params_s = f"{params:.0f}" if not np.isnan(params) else "--"
        train_s = f"{train_t:.2f}" if not np.isnan(train_t) else "--"
        lines.append(f"{esc(model)} & {lat:.2e} & {train_s} & {params_s} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    write(os.path.join(TABLES_DIR, "table_XV_aegis_runtime.tex"), "\n".join(lines))

    fig, ax = plt.subplots(figsize=(6, 4))
    cand_df = pd.read_csv(os.path.join(RESULTS_DIR, "aegis_candidates.csv"))
    a_f1 = cand_df[(cand_df["candidate"] == "A_TreeGuidedGraphMoE") & (cand_df["split"] == "test")]["macro_f1"].mean()
    student_f1 = pd.read_csv(os.path.join(RESULTS_DIR, "aegis_ablation.csv"))
    student_f1 = student_f1[(student_f1["component"] == "Full_DistilledStudent") & (student_f1["split"] == "test")]["macro_f1"].mean()
    for model, f1 in [("Full9ExpertEnsemble+MoERouter", a_f1), ("DistilledStudent", student_f1)]:
        sub = df[df["model"] == model]
        if len(sub):
            ax.scatter(sub["inference_latency_sec_per_sample"].mean(), f1, label=model, s=80)
    ax.set_xscale("log"); ax.set_xlabel("Inference latency (s/sample, log scale)"); ax.set_ylabel("Test macro-F1")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_aegis_runtime_tradeoff.png"), dpi=150)
    plt.close(fig)


def table_xvi_stat_sig():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "aegis_statistical_tests.csv"))
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             r"\caption{Statistical-significance labels vs.\ every other model (TEST split, macro-F1, Holm-corrected paired $t$-test, 3 seeds).}",
             r"\label{tab:aegis_stat_sig}", r"\begin{tabular}{lcccc}", r"\hline",
             r"Dataset & Superior & Tied & Mean-superior only & Inferior \\", r"\hline"]
    for ds in DATASETS:
        sub = df[df["dataset"] == ds]
        n_sup = (sub["label"] == "statistically superior").sum()
        n_tied = (sub["label"] == "statistically tied").sum()
        n_msup = (sub["label"] == "mean-superior only (statistically tied)").sum()
        n_inf = (sub["label"] == "inferior").sum()
        lines.append(f"{esc(ds)} & {n_sup} & {n_tied} & {n_msup} & {n_inf} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    write(os.path.join(TABLES_DIR, "table_XVI_aegis_stat_sig.tex"), "\n".join(lines))


def table_xvii_win_tie_loss():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "aegis_win_tie_loss_summary.csv"))
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             r"\caption{Win/tie/loss summary vs.\ 41 other models (37 baselines + legacy RL method + 3 proposed-variant ablations, TEST split).}",
             r"\label{tab:aegis_win_tie_loss}", r"\begin{tabular}{lccccc}", r"\hline",
             r"Dataset & Compared & Nominal wins & Stat. superior & Stat. tied & Inferior \\", r"\hline"]
    for _, r in df.iterrows():
        lines.append(f"{esc(r['dataset'])} & {r['n_baselines_compared']} & {r['n_nominal_wins']} & "
                      f"{r['n_statistically_superior']} & {r['n_statistically_tied']} & {r['n_inferior']} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    write(os.path.join(TABLES_DIR, "table_XVII_aegis_win_tie_loss.tex"), "\n".join(lines))

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["n_statistically_superior"], width=0.2, label="Stat. superior")
    ax.bar(x, df["n_statistically_tied"], width=0.2, label="Stat. tied")
    ax.bar(x + 0.2, df["n_inferior"], width=0.2, label="Inferior")
    ax.set_xticks(x); ax.set_xticklabels(df["dataset"], fontsize=8)
    ax.set_ylabel("# of 41 other models"); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_aegis_win_tie_loss.png"), dpi=150)
    plt.close(fig)


def fig_binary_macro_f1_comparison():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "baselines.csv"))
    df = df[df["eval_split"] == "test"]
    families = ["classical_tree", "deep", "graph", "rl", "legacy_rl_method", "proposed_variant", "proposed_method"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, ds in enumerate(DATASETS):
        means = []
        for fam in families:
            sub = df[(df["dataset"] == ds) & (df["family"] == fam)]
            means.append(sub["macro_f1"].mean() if len(sub) else np.nan)
        ax.plot(families, means, marker="o", label=ds)
    ax.set_xticklabels(families, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean test macro-F1"); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_aegis_family_comparison.png"), dpi=150)
    plt.close(fig)


def main():
    table_selection()
    table_v_binary_detection()
    table_viii_strongest_baseline()
    table_ix_x_reward_modes()
    table_xi_ablation()
    table_xii_low_label()
    table_xiii_robustness()
    table_xiv_calibration()
    table_xv_runtime()
    table_xvi_stat_sig()
    table_xvii_win_tie_loss()
    fig_binary_macro_f1_comparison()
    print("Saved AegisGraph-X tables/figures.")


if __name__ == "__main__":
    main()
