"""
Part 12/13: statistical significance (Table XVI) and win/tie/loss summary
(Table XVII) for AegisGraph-X (model "43_AegisGraphX") against all 42 other
rows in results/csv/baselines.csv (37 original baselines + legacy Offline
Safe Graph RL [#39] + the 3 proposed-variant ablation points [#40-42]).

Labels follow the task's required wording exactly: "statistically superior"
(AegisGraph-X's mean is higher AND p_holm < 0.05), "statistically tied"
(p_holm >= 0.05, i.e. no significant difference either way), "mean-superior
only" (AegisGraph-X's mean is higher but not significant -- collapses into
"statistically tied" under the task's binary labeling, reported as a
sub-note), "inferior" (baseline's mean is higher AND p_holm < 0.05).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import pandas as pd

from experiments.statistical_tests import paired_tests_vs_winner, holm_correction

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "results", "csv")
WINNER = "43_AegisGraphX"
DATASETS = ["UNSW-NB15", "ToN-IoT-Network", "CICIoT2023"]


def label_row(row):
    if pd.isna(row.get("p_ttest_holm")):
        return "insufficient paired seeds"
    winner_higher = row["winner_mean"] > row["baseline_mean"]
    significant = row["p_ttest_holm"] < 0.05
    if winner_higher and significant:
        return "statistically superior"
    if (not winner_higher) and significant:
        return "inferior"
    if winner_higher and not significant:
        return "mean-superior only (statistically tied)"
    return "statistically tied"


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "baselines.csv"))
    df = df[df["eval_split"] == "test"]

    all_results = []
    for dataset in DATASETS:
        res = paired_tests_vs_winner(df, metric="macro_f1", winner_name=WINNER, dataset=dataset,
                                      model_col="model", seed_col="seed")
        res["label"] = res.apply(label_row, axis=1)
        all_results.append(res)
    full = pd.concat(all_results, ignore_index=True)
    full.to_csv(os.path.join(RESULTS_DIR, "aegis_statistical_tests.csv"), index=False)

    summary_rows = []
    for dataset in DATASETS:
        sub = full[full["dataset"] == dataset]
        counts = sub["label"].value_counts().to_dict()
        n_superior = counts.get("statistically superior", 0)
        n_inferior = counts.get("inferior", 0)
        n_tied = counts.get("statistically tied", 0) + counts.get("mean-superior only (statistically tied)", 0)
        n_mean_superior_only = counts.get("mean-superior only (statistically tied)", 0)
        n_nominal_wins = int((sub["winner_mean"] > sub["baseline_mean"]).sum())
        n_total = len(sub)
        summary_rows.append(dict(dataset=dataset, n_baselines_compared=n_total,
                                  n_nominal_wins=n_nominal_wins,
                                  n_statistically_superior=n_superior,
                                  n_statistically_tied=n_tied,
                                  n_mean_superior_only=n_mean_superior_only,
                                  n_inferior=n_inferior))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(RESULTS_DIR, "aegis_win_tie_loss_summary.csv"), index=False)
    print(summary.to_string())

    print("\nInferior comparisons (AegisGraph-X loses significantly):")
    print(full[full["label"] == "inferior"][["dataset", "baseline", "winner_mean", "baseline_mean",
                                              "p_ttest_holm"]].to_string())
    print("\nStatistically superior comparisons:")
    print(full[full["label"] == "statistically superior"][["dataset", "baseline", "winner_mean",
                                                              "baseline_mean", "p_ttest_holm"]].to_string())


if __name__ == "__main__":
    main()
