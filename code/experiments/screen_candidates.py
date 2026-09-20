"""
Part 6: Candidate-screening protocol.

Runs all 10 candidate methods (smallest credible prototypes) on VALIDATION data
only (each candidate trains on the TRAIN split's env and is scored on the VAL
split's env; TEST is never touched here) across the 3 primary datasets, for a
small number of seeds, and writes results/csv/candidate_screening.csv with the
12 screening criteria from Part 6 (macro-F1, minority recall, PR-AUC, FPR, FNR,
cumulative reward, detection delay, action cost [telemetry_cost], telemetry
budget usage, latency, memory, seed-to-seed stability -- stability is computed
downstream as std-across-seeds once all seeds are in the CSV).
"""
import sys, os, time, json, tracemalloc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd

from experiments.build_envs import build_all
from methods import (candidate_01_conformal as c1, candidate_02_causal as c2,
                      candidate_03_offline_safe as c3, candidate_04_budgeted_acquisition as c4,
                      candidate_05_hierarchical_marl as c5, candidate_06_continual_meta as c6,
                      candidate_07_adversarial as c7, candidate_08_neurosymbolic as c8,
                      candidate_09_world_model as c9, candidate_10_federated_robust as c10)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASETS = ["UNSW-NB15", "ToN-IoT-Network", "CICIoT2023"]
SEEDS = [0, 1]
N_UPDATES = 20
EPISODES_PER_UPDATE = 8
# Screening is a validation-only *model-selection* pass (Part 6), not the final
# test-set comparison (Part 12) -- so every dataset is capped to the same modest
# row budget here to keep 10 candidates x 3 datasets x 2 seeds tractable. The
# final comparison against baselines (screen_candidates is never run on TEST)
# uses the full/larger row budgets in build_envs.DATASET_ROW_CAPS instead.
SCREENING_ROW_CAP = 60000


def run_candidate(name, fn_kwargs_builder, bundle, seed):
    t0 = time.time()
    tracemalloc.start()
    result = fn_kwargs_builder(bundle, seed)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    latency = time.time() - t0
    result["latency_sec"] = latency
    result["peak_memory_MB"] = peak / 1e6
    return result


def candidate_runners():
    def r1(b, seed):
        env_val = b["envs"]["val"]
        y_calib = b["parts"]["val"]["y"][: max(len(b["parts"]["val"]["y"]) // 3, 50)]
        X_calib = b["parts"]["val"]["X"][: len(y_calib)]
        return c1.run(b["envs"]["train"], env_val, X_calib, y_calib, b["probe"],
                       n_updates=N_UPDATES, episodes_per_update=EPISODES_PER_UPDATE, seed=seed)

    def r2(b, seed):
        return c2.run(b["envs"]["train"], b["envs"]["val"], b["probe"],
                       n_updates=N_UPDATES, episodes_per_update=EPISODES_PER_UPDATE, seed=seed)

    def r3(b, seed):
        return c3.run(b["envs"]["train"], b["envs"]["val"], n_episodes_offline=150, epochs=10, seed=seed)

    def r4(b, seed):
        return c4.run(b["envs"]["train"], b["envs"]["val"],
                       n_updates=N_UPDATES, episodes_per_update=EPISODES_PER_UPDATE, seed=seed)

    def r5(b, seed):
        return c5.run(b["envs"]["train"], b["envs"]["val"],
                       n_updates=N_UPDATES, episodes_per_update=EPISODES_PER_UPDATE, seed=seed)

    def r6(b, seed):
        y_multi_col = None
        train_df = b["parts"]["train"]["df"]
        for cand_col in ["type", "attack_cat", "Attack_type", "label"]:
            if cand_col in train_df.columns and train_df[cand_col].nunique() > 2:
                y_multi_col = cand_col
                break
        return c6.run(b["envs"]["train"], b["envs"]["val"], train_df=train_df, y_multi_col=y_multi_col,
                       window_size=64, n_updates=N_UPDATES, episodes_per_update=EPISODES_PER_UPDATE, seed=seed)

    def r7(b, seed):
        return c7.run(b["envs"]["train"], b["envs"]["val"],
                       n_updates=N_UPDATES, episodes_per_update=EPISODES_PER_UPDATE, seed=seed)

    def r8(b, seed):
        y_multi_train = b["parts"]["train"]["y_multi"]
        return c8.run(b["envs"]["train"], b["envs"]["val"], X_train=b["parts"]["train"]["X"],
                       y_multi_train=y_multi_train, n_updates=N_UPDATES,
                       episodes_per_update=EPISODES_PER_UPDATE, seed=seed)

    def r9(b, seed):
        return c9.run(b["envs"]["train"], b["envs"]["val"], b["parts"]["train"]["y"],
                       n_pretrain_steps=200, n_updates=N_UPDATES,
                       episodes_per_update=EPISODES_PER_UPDATE, seed=seed)

    def r10(b, seed):
        train_env = b["envs"]["train"]
        n_flows = train_env.n_flows
        k_sites = 5
        shard_bounds = np.linspace(0, n_flows, k_sites + 1).astype(int)

        def build_shard(site):
            from env.pomdp_env import IDSPOMDPEnv, EnvConfig
            lo, hi = shard_bounds[site], shard_bounds[site + 1]
            p = b["parts"]["train"]
            cfg = EnvConfig(window_size=64, telemetry_budget=8, seed=seed + site)
            # flow_index[:,0] still indexes into the full (unsliced) p["graphs"] window
            # list, which is correct and left untouched -- only X/y/host_keys/flow_index
            # rows are restricted to this site's shard.
            return IDSPOMDPEnv(p["X"][lo:hi], p["y"][lo:hi], p["host_keys"][lo:hi],
                                p["flow_index"][lo:hi], p["graphs"], b["probe"], config=cfg)
        return c10.run(build_shard, b["envs"]["val"], k_sites=k_sites, n_malicious=1,
                        n_rounds=5, local_updates_per_round=2, episodes_per_update=6, seed=seed)

    return {
        "C1_ConformalRiskGraphRL": r1,
        "C2_CausalCounterfactualRL": r2,
        "C3_OfflineSafeGraphRL": r3,
        "C4_BudgetedActiveAcquisitionRL": r4,
        "C5_HierarchicalMultiAgentRL": r5,
        "C6_ContinualMetaGraphRL": r6,
        "C7_LogicPreservingAdversarialRL": r7,
        "C8_NeuroSymbolicATTCKRL": r8,
        "C9_SelfSupervisedWorldModelRL": r9,
        "C10_ByzantineRobustFederatedRL": r10,
    }


def main():
    runners = candidate_runners()
    rows = []
    for dataset in DATASETS:
        print(f"\n=== building envs for {dataset} ===", flush=True)
        bundle = build_all(dataset, row_cap=SCREENING_ROW_CAP)
        for cand_name, fn in runners.items():
            for seed in SEEDS:
                print(f"  [{dataset}] {cand_name} seed={seed} ...", flush=True)
                try:
                    res = run_candidate(cand_name, fn, bundle, seed)
                    res.update({"dataset": dataset, "candidate": cand_name, "seed": seed, "status": "ok"})
                except Exception as e:
                    res = {"dataset": dataset, "candidate": cand_name, "seed": seed,
                           "status": f"ERROR: {type(e).__name__}: {e}"}
                rows.append(res)
                pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "candidate_screening.csv"), index=False)
                print(f"    -> {res.get('status')} macro_f1={res.get('macro_f1')} "
                      f"reward={res.get('cumulative_reward_per_window')}", flush=True)
    print("\nSaved results/csv/candidate_screening.csv")


if __name__ == "__main__":
    main()
