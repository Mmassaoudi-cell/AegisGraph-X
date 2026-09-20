"""
Part 7: dual-mode RL evaluation (fixes the reward-scale-fairness issue
disclosed as a limitation in the prior manuscript).

Mode 1 (common reward): Candidates B/C are evaluated under the SAME fixed
default reward weights used throughout run_aegis.py -- this is exactly what
aegis_candidates.csv already contains, and on CICIoT2023 it reproduces a
genuine, disclosed RL collapse (the investigation layer learns a degenerate
always-ISOLATE policy under that dataset's extreme attack-majority class
balance, correctly gated out by run_aegis_select.py's minority-recall floor).

Mode 2 (validation-tuned reward): a small, validation-only grid search over
the reward terms most directly implicated in the collapse mechanism
(isolation cost/credit and the false-positive penalty) is run on EACH
dataset's VALIDATION split only (never test), reusing the continuous
minority-recall-gate scoring philosophy from the legacy Candidate-3 tuning
fix (manifests/technical_audit_report.md Section 7) so a config cannot look
good on validation by collapsing recall. The best config per dataset is then
evaluated on TEST and reported as a separate, explicitly-labeled Mode-2 row --
never mixed with or silently substituted for the Mode-1 (default-weight) row,
and raw cumulative reward is never compared across the two modes or across
datasets, since the weights themselves differ (Part 7 of the task spec).
"""
import sys, os, time, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import pandas as pd

from aegis.pipeline import fit_aegis_backbone, build_aegis_envs
from aegis.rl_policy import AegisPPOTrainer, aegis_rollout_classification
from methods.core import TinyGraphEncoder

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "results", "csv")
DATASETS = ["UNSW-NB15", "ToN-IoT-Network", "CICIoT2023"]
SEEDS = [0, 1, 2]
ROW_CAP = 50000
WINDOW_SIZE = 64
TELEMETRY_BUDGET = 8
N_UPDATES = 15
EPISODES_PER_UPDATE = 8
MINORITY_RECALL_FLOOR = 0.30

GRID = [
    dict(c_isolate_benign=1.2, r_isolate_malicious=0.6, w_fp=1.5),   # default (Mode 1 weights, included for reference)
    dict(c_isolate_benign=3.0, r_isolate_malicious=0.3, w_fp=1.5),
    dict(c_isolate_benign=6.0, r_isolate_malicious=0.1, w_fp=1.5),
    dict(c_isolate_benign=6.0, r_isolate_malicious=0.1, w_fp=3.0),
]


def gated_score(ids_metrics, floor=MINORITY_RECALL_FLOOR):
    gate = float(np.clip(ids_metrics.get("minority_recall", 0.0) / floor, 0.0, 1.0))
    return gate * ids_metrics.get("macro_f1", 0.0)


def tune_one(dataset, seed, backbone):
    envs_by_config = {}
    best_cfg, best_score, best_val_ids = None, -1.0, None
    for cfg in GRID:
        envs = build_aegis_envs(backbone, window_size=WINDOW_SIZE, telemetry_budget=TELEMETRY_BUDGET,
                                 reward_weights=cfg, seed=seed)
        enc = TinyGraphEncoder(in_dim=envs["train"].graphs[0].x.shape[1], hidden=16, out_dim=8)
        trainer = AegisPPOTrainer(envs["train"], enc, seed=seed)
        for _ in range(N_UPDATES):
            trainer.update(n_episodes=EPISODES_PER_UPDATE, epochs=4)
        trainer.env = envs["val"]
        ids, decision = aegis_rollout_classification(trainer)
        score = gated_score(ids)
        print(f"    grid cfg={cfg} -> val macro_f1={ids.get('macro_f1', float('nan')):.4f} "
              f"minority_recall={ids.get('minority_recall', float('nan')):.4f} gated_score={score:.4f}", flush=True)
        if score > best_score:
            best_score, best_cfg = score, cfg
            best_val_ids = ids
    return best_cfg, best_score, best_val_ids


def evaluate_mode(dataset, seed, backbone, reward_weights, mode_name):
    envs = build_aegis_envs(backbone, window_size=WINDOW_SIZE, telemetry_budget=TELEMETRY_BUDGET,
                             reward_weights=reward_weights, seed=seed)
    enc = TinyGraphEncoder(in_dim=envs["train"].graphs[0].x.shape[1], hidden=16, out_dim=8)
    trainer = AegisPPOTrainer(envs["train"], enc, seed=seed)
    for _ in range(N_UPDATES):
        trainer.update(n_episodes=EPISODES_PER_UPDATE, epochs=4)

    rows = []
    for split in ["val", "test"]:
        trainer.env = envs[split]
        ids, decision = aegis_rollout_classification(trainer)
        row = dict(dataset=dataset, seed=seed, mode=mode_name, split=split, reward_weights=str(reward_weights))
        row.update(ids)
        row.update(decision)
        rows.append(row)
    return rows


def main():
    rows = []
    for dataset in DATASETS:
        # Reward-weight grid search is run once per dataset (seed=0's backbone), not per seed,
        # for tractability -- consistent with the legacy tuning fix's "tune on one split, apply
        # uniformly" pattern; Mode-2 final evaluation below still runs across all seeds.
        print(f"=== reward-mode tuning {dataset} ===", flush=True)
        backbone0 = fit_aegis_backbone(dataset, seed=0, row_cap=ROW_CAP, window_size=WINDOW_SIZE,
                                        telemetry_budget=TELEMETRY_BUDGET, epochs_neural=4, epochs_graph=4, top_k=3)
        best_cfg, best_score, best_val_ids = tune_one(dataset, 0, backbone0)
        print(f"  best validation-tuned reward config for {dataset}: {best_cfg} (gated_score={best_score:.4f})",
              flush=True)

        for seed in SEEDS:
            backbone = backbone0 if seed == 0 else fit_aegis_backbone(
                dataset, seed=seed, row_cap=ROW_CAP, window_size=WINDOW_SIZE, telemetry_budget=TELEMETRY_BUDGET,
                epochs_neural=4, epochs_graph=4, top_k=3)
            rows.extend(evaluate_mode(dataset, seed, backbone, None, "Mode1_CommonDefaultReward"))
            rows.extend(evaluate_mode(dataset, seed, backbone, best_cfg, "Mode2_ValidationTunedReward"))
            pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "aegis_reward_modes.csv"), index=False)
            print(f"  {dataset} seed={seed} done", flush=True)
    print("Saved results/csv/aegis_reward_modes.csv")


if __name__ == "__main__":
    main()
