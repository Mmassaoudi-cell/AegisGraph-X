"""
Appends AegisGraph-X's TEST-split rows into results/csv/baselines.csv using
the same long-format schema as the 37 existing baselines, so every downstream
table/statistics script (statistical_tests.py, make_tables_figures.py) can
treat it identically. Never modifies or removes the existing 37 baseline rows
or the legacy Candidate-3 rows (relabeled here from family="selected_method"
to family="legacy_rl_method", baseline #39, per the task's explicit
instruction to keep it as a transparent comparison point, not delete it).

Appended rows (Part 5, #39-43):
  39. Offline Safe Graph RL (legacy)          <- relabeled existing rows
  40. Tree ensemble teacher (best single tree expert, no MoE/no calibration)
  41. Graph mixture-of-experts (MoE-routed ensemble, uncalibrated)
  42. Distilled student (compact neural student, full hard-class training)
  43. AegisGraph-X (full method) -- whichever of Candidate A/B/C the
      validation-only selection (run_aegis_select.py) chose

Run AFTER run_aegis.py (aegis_candidates.csv, aegis_ablation.csv) and
run_aegis_select.py (manifests/aegis_candidate_selection_report.json).
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "results", "csv")
MANIFESTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                              "manifests")

BASELINE_SCHEMA_COLS = ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'attack_recall',
                         'minority_recall', 'precision_attack', 'fpr', 'fnr', 'roc_auc', 'pr_auc',
                         'model', 'family', 'train_time_sec', 'inference_latency_sec_per_sample',
                         'peak_memory_MB', 'seed', 'eval_split', 'dataset', 'n_params',
                         'cumulative_reward', 'cumulative_reward_per_window', 'avg_detection_delay',
                         'telemetry_cost', 'escalation_rate', 'isolation_rate', 'abstention_rate',
                         'budget_violation_rate']


def build_row(**kwargs):
    row = {c: np.nan for c in BASELINE_SCHEMA_COLS}
    row.update(kwargs)
    return row


def main():
    baselines_path = os.path.join(RESULTS_DIR, "baselines.csv")
    baselines = pd.read_csv(baselines_path)

    # 1. Relabel the legacy method as baseline #39 (kept, not deleted).
    mask = baselines["family"] == "selected_method"
    baselines.loc[mask, "family"] = "legacy_rl_method"
    baselines.loc[mask, "model"] = "39_OfflineSafeGraphRL_legacy"

    cand = pd.read_csv(os.path.join(RESULTS_DIR, "aegis_candidates.csv"))
    cand_test = cand[cand["split"] == "test"].copy()
    ablation = pd.read_csv(os.path.join(RESULTS_DIR, "aegis_ablation.csv"))
    ablation_test = ablation[ablation["split"] == "test"].copy()

    with open(os.path.join(MANIFESTS_DIR, "aegis_candidate_selection_report.json")) as f:
        selection = json.load(f)
    winner = selection["winner"]

    new_rows = []

    # 40. Tree ensemble teacher = "No_ExpertEnsemble_BestSingleExpert" ablation row
    for _, r in ablation_test[ablation_test["component"] == "No_ExpertEnsemble_BestSingleExpert"].iterrows():
        new_rows.append(build_row(
            dataset=r["dataset"], seed=r["seed"], eval_split="test",
            model="40_TreeEnsembleTeacher", family="proposed_variant",
            accuracy=r["accuracy"], balanced_accuracy=r["balanced_accuracy"], macro_f1=r["macro_f1"],
            weighted_f1=r["weighted_f1"], attack_recall=r["recall_attack"], minority_recall=r["minority_recall"],
            precision_attack=r["precision_attack"], fpr=r["fpr"], fnr=r["fnr"],
            roc_auc=r["roc_auc"], pr_auc=r["pr_auc"],
        ))

    # 41. Graph mixture-of-experts (MoE-routed, uncalibrated) = "No_Calibration_Raw0.5" ablation row
    for _, r in ablation_test[ablation_test["component"] == "No_Calibration_Raw0.5"].iterrows():
        new_rows.append(build_row(
            dataset=r["dataset"], seed=r["seed"], eval_split="test",
            model="41_GraphMixtureOfExperts", family="proposed_variant",
            accuracy=r["accuracy"], balanced_accuracy=r["balanced_accuracy"], macro_f1=r["macro_f1"],
            weighted_f1=r["weighted_f1"], attack_recall=r["recall_attack"], minority_recall=r["minority_recall"],
            precision_attack=r["precision_attack"], fpr=r["fpr"], fnr=r["fnr"],
            roc_auc=r["roc_auc"], pr_auc=r["pr_auc"],
        ))

    # 42. Distilled student = "Full_DistilledStudent" ablation row
    runtime = pd.read_csv(os.path.join(RESULTS_DIR, "aegis_runtime.csv"))
    student_runtime = runtime[runtime["model"] == "DistilledStudent"].set_index(["dataset", "seed"])
    for _, r in ablation_test[ablation_test["component"] == "Full_DistilledStudent"].iterrows():
        key = (r["dataset"], r["seed"])
        n_params = student_runtime.loc[key, "n_params"] if key in student_runtime.index else np.nan
        lat = student_runtime.loc[key, "inference_latency_sec_per_sample"] if key in student_runtime.index else np.nan
        new_rows.append(build_row(
            dataset=r["dataset"], seed=r["seed"], eval_split="test",
            model="42_DistilledStudent", family="proposed_variant",
            accuracy=r["accuracy"], balanced_accuracy=r["balanced_accuracy"], macro_f1=r["macro_f1"],
            weighted_f1=r["weighted_f1"], attack_recall=r["recall_attack"], minority_recall=r["minority_recall"],
            precision_attack=r["precision_attack"], fpr=r["fpr"], fnr=r["fnr"],
            roc_auc=r["roc_auc"], pr_auc=r["pr_auc"], n_params=n_params,
            inference_latency_sec_per_sample=lat,
        ))

    # 43. AegisGraph-X full method = whichever candidate the validation-only selection chose
    for _, r in cand_test[cand_test["candidate"] == winner].iterrows():
        new_rows.append(build_row(
            dataset=r["dataset"], seed=r["seed"], eval_split="test",
            model="43_AegisGraphX", family="proposed_method",
            accuracy=r.get("accuracy"), balanced_accuracy=r.get("balanced_accuracy"), macro_f1=r["macro_f1"],
            weighted_f1=r.get("weighted_f1"), attack_recall=r.get("attack_recall"), minority_recall=r["minority_recall"],
            precision_attack=r.get("precision_attack"), fpr=r["fpr"], fnr=r["fnr"],
            roc_auc=r.get("roc_auc"), pr_auc=r.get("pr_auc"),
            n_params=r.get("n_rl_params"), train_time_sec=r.get("latency_sec"),
            peak_memory_MB=r.get("peak_memory_MB"),
            cumulative_reward=r.get("cumulative_reward"),
            cumulative_reward_per_window=r.get("cumulative_reward_per_window"),
            avg_detection_delay=r.get("avg_detection_delay"), telemetry_cost=r.get("telemetry_cost"),
            escalation_rate=r.get("escalation_rate"), isolation_rate=r.get("isolation_rate"),
            abstention_rate=r.get("abstention_rate"), budget_violation_rate=r.get("budget_violation_rate"),
        ))

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([baselines, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["dataset", "model", "seed", "eval_split"], keep="last")
    combined.to_csv(baselines_path, index=False)
    print(f"Appended {len(new_df)} AegisGraph-X rows (models 40-43) to {baselines_path}")
    print(f"Winner selected as model 43: {winner}")
    print(combined[combined["model"].isin(
        ["40_TreeEnsembleTeacher", "41_GraphMixtureOfExperts", "42_DistilledStudent", "43_AegisGraphX"])]
          .groupby(["model", "dataset"])["macro_f1"].mean())


if __name__ == "__main__":
    main()
