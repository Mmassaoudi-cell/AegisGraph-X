"""
Post-mortem re-run after fixing the ISOLATE double-credit bug in
env/pomdp_env.py and env/pomdp_env_aegis.py (see conversation/audit addendum).

Baseline #39 (legacy Candidate 3 / Offline Safe Graph RL) needs NO re-run:
results/csv/baselines.csv already shows isolation_rate == 0.0 for all 15
(dataset, seed) rows under family=="legacy_rl_method" -- its evaluated policy
never selects ISOLATE even once (consistent with the logging policy only
proposing it via rare uniform-random exploration, plus CQL's conservative
penalty and the explicit saw_isolate safety mask), so the bug had zero effect
on its already-reported numbers. This script only re-runs what the fix can
actually change: AegisGraph-X's own Candidate A/B/C (A unaffected structurally
since it never touches the RL env, but re-run for a clean paired comparison)
and the dual-mode reward evaluation.

Candidate A's backbone (experts + router + calibration) is refit identically
to before (the fix touches only the ISOLATE branch of the env, not the
backbone), so its numbers are expected to match the pre-fix run up to normal
seed variance -- refit here only because backbones are not persisted to disk,
not because the fix could plausibly change them.
"""
import sys, os, time, tracemalloc
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import pandas as pd

from aegis.pipeline import fit_aegis_backbone, build_aegis_envs, backbone_metrics
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

GRID = [
    dict(c_isolate_benign=1.2, r_isolate_malicious=0.6, w_fp=1.5),
    dict(c_isolate_benign=3.0, r_isolate_malicious=0.3, w_fp=1.5),
    dict(c_isolate_benign=6.0, r_isolate_malicious=0.1, w_fp=1.5),
    dict(c_isolate_benign=6.0, r_isolate_malicious=0.1, w_fp=3.0),
]
MINORITY_RECALL_FLOOR = 0.30


def gated_score(ids_metrics, floor=MINORITY_RECALL_FLOOR):
    gate = float(np.clip(ids_metrics.get("minority_recall", 0.0) / floor, 0.0, 1.0))
    return gate * ids_metrics.get("macro_f1", 0.0)


def run_candidates(dataset, seed, backbone):
    rows = []
    for split in ["val", "test"]:
        y = backbone["parts"][split]["y"]
        ens_p = backbone[f"{split}_ensemble_p"]
        m = backbone_metrics(ens_p, y, backbone["calib"], threshold_key="temperature")
        m.update(dataset=dataset, seed=seed, candidate="A_TreeGuidedGraphMoE", split=split)
        rows.append(m)

    envs = build_aegis_envs(backbone, window_size=WINDOW_SIZE, telemetry_budget=TELEMETRY_BUDGET, seed=seed)
    enc = TinyGraphEncoder(in_dim=envs["train"].graphs[0].x.shape[1], hidden=16, out_dim=8)
    trainer = AegisPPOTrainer(envs["train"], enc, seed=seed)
    for _ in range(N_UPDATES):
        trainer.update(n_episodes=EPISODES_PER_UPDATE, epochs=4)

    for shield, cand_name in [(False, "B_BudgetedRLInvestigation"), (True, "C_SafetyShieldedRL")]:
        trainer.shield = shield
        for split in ["val", "test"]:
            trainer.env = envs[split]
            ids, decision = aegis_rollout_classification(trainer)
            row = dict(dataset=dataset, seed=seed, candidate=cand_name, split=split)
            row.update(ids)
            row.update(decision)
            rows.append(row)
    return rows


def tune_reward_grid(dataset, backbone):
    best_cfg, best_score = None, -1.0
    for cfg in GRID:
        envs = build_aegis_envs(backbone, window_size=WINDOW_SIZE, telemetry_budget=TELEMETRY_BUDGET,
                                 reward_weights=cfg, seed=0)
        enc = TinyGraphEncoder(in_dim=envs["train"].graphs[0].x.shape[1], hidden=16, out_dim=8)
        trainer = AegisPPOTrainer(envs["train"], enc, seed=0)
        for _ in range(N_UPDATES):
            trainer.update(n_episodes=EPISODES_PER_UPDATE, epochs=4)
        trainer.env = envs["val"]
        ids, decision = aegis_rollout_classification(trainer)
        score = gated_score(ids)
        print(f"    [{dataset}] grid cfg={cfg} -> val macro_f1={ids.get('macro_f1', float('nan')):.4f} "
              f"minority_recall={ids.get('minority_recall', float('nan')):.4f} gated_score={score:.4f}", flush=True)
        if score > best_score:
            best_score, best_cfg = score, cfg
    return best_cfg


def run_reward_modes(dataset, seed, backbone, best_cfg):
    rows = []
    for mode_name, weights in [("Mode1_CommonDefaultReward", None), ("Mode2_ValidationTunedReward", best_cfg)]:
        envs = build_aegis_envs(backbone, window_size=WINDOW_SIZE, telemetry_budget=TELEMETRY_BUDGET,
                                 reward_weights=weights, seed=seed)
        enc = TinyGraphEncoder(in_dim=envs["train"].graphs[0].x.shape[1], hidden=16, out_dim=8)
        trainer = AegisPPOTrainer(envs["train"], enc, seed=seed)
        for _ in range(N_UPDATES):
            trainer.update(n_episodes=EPISODES_PER_UPDATE, epochs=4)
        for split in ["val", "test"]:
            trainer.env = envs[split]
            ids, decision = aegis_rollout_classification(trainer)
            row = dict(dataset=dataset, seed=seed, mode=mode_name, split=split, reward_weights=str(weights))
            row.update(ids)
            row.update(decision)
            rows.append(row)
    return rows


def main():
    cand_rows, reward_rows = [], []
    for dataset in DATASETS:
        print(f"=== {dataset}: grid search for reward modes (post-fix) ===", flush=True)
        backbone0 = fit_aegis_backbone(dataset, seed=0, row_cap=ROW_CAP, window_size=WINDOW_SIZE,
                                        telemetry_budget=TELEMETRY_BUDGET, epochs_neural=4, epochs_graph=4, top_k=3)
        best_cfg = tune_reward_grid(dataset, backbone0)
        print(f"  best post-fix reward config for {dataset}: {best_cfg}", flush=True)

        for seed in SEEDS:
            print(f"=== {dataset} seed={seed} (post-fix candidates) ===", flush=True)
            backbone = backbone0 if seed == 0 else fit_aegis_backbone(
                dataset, seed=seed, row_cap=ROW_CAP, window_size=WINDOW_SIZE, telemetry_budget=TELEMETRY_BUDGET,
                epochs_neural=4, epochs_graph=4, top_k=3)
            try:
                cand_rows.extend(run_candidates(dataset, seed, backbone))
            except Exception as e:
                import traceback; traceback.print_exc()
            try:
                reward_rows.extend(run_reward_modes(dataset, seed, backbone, best_cfg))
            except Exception as e:
                import traceback; traceback.print_exc()
            pd.DataFrame(cand_rows).to_csv(os.path.join(RESULTS_DIR, "aegis_candidates_postfix.csv"), index=False)
            pd.DataFrame(reward_rows).to_csv(os.path.join(RESULTS_DIR, "aegis_reward_modes_postfix.csv"), index=False)
            print(f"  done, {len(cand_rows)} candidate rows, {len(reward_rows)} reward-mode rows so far", flush=True)
    print("Saved aegis_candidates_postfix.csv, aegis_reward_modes_postfix.csv")


if __name__ == "__main__":
    main()
