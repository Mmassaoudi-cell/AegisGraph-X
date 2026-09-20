"""
Single consolidated AegisGraph-X experiment driver. Fits the expert-pool +
MoE-router backbone ONCE per (dataset, seed) and reuses it for everything
downstream (avoids re-fitting 9 experts multiple times across separate
scripts):

  - Candidate A (backbone only, calibrated) classification metrics
  - Ablation variants: no-expert-ensemble (best single expert), no-MoE-router
    (equal-weight average), no-graph-branch, no-temporal-branch, no-calibration
  - Distilled compact student: full (focal+contrastive+hard-mining all on),
    and one-at-a-time-off ablations (no-focal / no-contrastive / no-hard-mining),
    plus a plain-KD-only student (all three off) for the "no tree distillation
    hard-class extras" reference point
  - Candidate B (+ budgeted RL investigation layer) and Candidate C
    (+ conformal safety shield), trained on the TRAIN-split AegisPOMDPEnv and
    evaluated greedily on VAL and TEST
  - Runtime/latency/memory/parameter-count for each deployed variant (best
    tree alone, full 9-expert ensemble, distilled student, full AegisGraph-X)

All thresholds/calibration are fit on VAL only; TEST is scored, never fit.
Output: results/csv/aegis_candidates.csv (Candidate A/B/C rows, one per
dataset x seed x candidate x split) and results/csv/aegis_ablation.csv
(component-ablation rows) and results/csv/aegis_runtime.csv.
"""
import sys, os, time, tracemalloc
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import pandas as pd
import torch

from aegis.pipeline import (fit_aegis_backbone, build_aegis_envs, backbone_metrics,
                             simple_classification_metrics, ablated_router_ensemble,
                             equal_weight_ensemble, best_single_expert)
from aegis.calibrate import select_threshold_max_f1
from aegis.distill import train_distilled_student, student_predict_proba, tree_leaf_embedding, DistilledStudent
from aegis.rl_policy import AegisPPOTrainer, aegis_rollout_classification
from methods.core import TinyGraphEncoder

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "results", "csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASETS = ["UNSW-NB15", "ToN-IoT-Network", "CICIoT2023"]
SEEDS = [0, 1, 2]
ROW_CAP = 50000
WINDOW_SIZE = 64
TELEMETRY_BUDGET = 8
N_UPDATES = 15
EPISODES_PER_UPDATE = 8
STUDENT_EPOCHS = 15


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def measure_latency(predict_fn, X, n_repeat=3):
    times = []
    for _ in range(n_repeat):
        t0 = time.time()
        predict_fn(X)
        times.append(time.time() - t0)
    return float(np.min(times)) / max(len(X), 1)


def run_candidate_a_and_ablation(dataset, seed, backbone):
    rows, ablation_rows = [], []
    y_val, y_test = backbone["parts"]["val"]["y"], backbone["parts"]["test"]["y"]

    for split, y in [("val", y_val), ("test", y_test)]:
        m = backbone_metrics(backbone[f"{split}_ensemble_p"], y, backbone["calib"], threshold_key="temperature")
        m.update(dataset=dataset, seed=seed, candidate="A_TreeGuidedGraphMoE", split=split,
                  n_experts=backbone["n_experts"])
        rows.append(m)

    def add_ablation(name, p_val, p_test, extra=None):
        t = select_threshold_max_f1(p_val, y_val)
        for split, p, y in [("val", p_val, y_val), ("test", p_test, y_test)]:
            m = simple_classification_metrics(p, y, t)
            m.update(dataset=dataset, seed=seed, component=name, split=split)
            if extra:
                m.update(extra)
            ablation_rows.append(m)

    a_val = backbone["val_ensemble_p"]; a_test = backbone["test_ensemble_p"]
    add_ablation("Full_AegisGraphX_backbone", a_val, a_test)

    best_name, best_p = best_single_expert(backbone)
    add_ablation("No_ExpertEnsemble_BestSingleExpert", best_p["val"], best_p["test"], extra={"note": best_name})

    eq = equal_weight_ensemble(backbone)
    add_ablation("No_MoERouter_EqualWeight", eq["val"], eq["test"])

    no_graph = ablated_router_ensemble(backbone, drop_experts=("ProtocolRoleGraphSAGE",), seed=seed)
    add_ablation("No_GraphBranch", no_graph["val"], no_graph["test"])

    no_temporal = ablated_router_ensemble(backbone, drop_experts=("TCN",), seed=seed)
    add_ablation("No_TemporalBranch", no_temporal["val"], no_temporal["test"])

    # No-calibration: raw ensemble probability thresholded at the fixed 0.5 point
    # (rather than the val-selected calibrated threshold used by Candidate A).
    for split, p, y in [("val", a_val, y_val), ("test", a_test, y_test)]:
        m = simple_classification_metrics(p, y, 0.5)
        m.update(dataset=dataset, seed=seed, component="No_Calibration_Raw0.5", split=split)
        ablation_rows.append(m)

    return rows, ablation_rows


def run_distillation(dataset, seed, backbone):
    ablation_rows, runtime_rows = [], []
    X_train = backbone["parts"]["train"]["X"]
    y_train = backbone["parts"]["train"]["y"]
    y_val, y_test = backbone["parts"]["val"]["y"], backbone["parts"]["test"]["y"]
    X_val, X_test = backbone["parts"]["val"]["X"], backbone["parts"]["test"]["X"]

    teacher_p_train = backbone["calib"].apply_all(backbone["train_ensemble_p"])["temperature"]

    tree_experts = backbone["tree_experts"]
    best_tree_name = max(tree_experts.keys(),
                          key=lambda n: np.mean(backbone["val_prob_matrix"][:, backbone["expert_names"].index(n)]))
    leaf_embed = tree_leaf_embedding(tree_experts[best_tree_name], X_train)

    variants = {
        "Full_DistilledStudent": dict(use_focal=True, use_contrastive=True, use_hard_mining=True),
        "No_FocalLoss": dict(use_focal=False, use_contrastive=True, use_hard_mining=True),
        "No_ContrastiveLearning": dict(use_focal=True, use_contrastive=False, use_hard_mining=True),
        "No_HardClassMining": dict(use_focal=True, use_contrastive=True, use_hard_mining=False),
        "PlainKD_NoHardClassExtras": dict(use_focal=False, use_contrastive=False, use_hard_mining=False),
    }
    for name, kwargs in variants.items():
        t0 = time.time()
        student = train_distilled_student(X_train, y_train, teacher_p_train, teacher_leaf_embed=leaf_embed,
                                           epochs=STUDENT_EPOCHS, seed=seed, **kwargs)
        train_time = time.time() - t0
        p_val = student_predict_proba(student, X_val)
        p_test = student_predict_proba(student, X_test)
        t = select_threshold_max_f1(p_val, y_val)
        for split, p, y in [("val", p_val, y_val), ("test", p_test, y_test)]:
            m = simple_classification_metrics(p, y, t)
            m.update(dataset=dataset, seed=seed, component=name, split=split, train_time_sec=train_time)
            ablation_rows.append(m)

        if name == "Full_DistilledStudent":
            latency = measure_latency(lambda X: student_predict_proba(student, X), X_test)
            runtime_rows.append(dict(dataset=dataset, seed=seed, model="DistilledStudent",
                                      n_params=count_params(student), inference_latency_sec_per_sample=latency,
                                      train_time_sec=train_time, model_size_kb=count_params(student) * 4 / 1024))
    return ablation_rows, runtime_rows


def run_rl(dataset, seed, backbone):
    rows = []
    envs = build_aegis_envs(backbone, window_size=WINDOW_SIZE, telemetry_budget=TELEMETRY_BUDGET, seed=seed)
    graph_in_dim = envs["train"].graphs[0].x.shape[1]
    enc = TinyGraphEncoder(in_dim=graph_in_dim, hidden=16, out_dim=8)
    trainer = AegisPPOTrainer(envs["train"], enc, seed=seed)

    t1 = time.time()
    tracemalloc.start()
    for _ in range(N_UPDATES):
        trainer.update(n_episodes=EPISODES_PER_UPDATE, epochs=4)
    _, peak_rl = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rl_train_time = time.time() - t1
    n_params = count_params(trainer.ac) + count_params(trainer.enc)

    for shield, cand_name in [(False, "B_BudgetedRLInvestigation"), (True, "C_SafetyShieldedRL")]:
        trainer.shield = shield
        for split in ["val", "test"]:
            trainer.env = envs[split]
            t2 = time.time()
            ids, decision = aegis_rollout_classification(trainer)
            eval_time = time.time() - t2
            row = dict(dataset=dataset, seed=seed, candidate=cand_name, split=split,
                       rl_train_time_sec=rl_train_time, eval_latency_sec=eval_time,
                       peak_memory_MB=peak_rl / 1e6, n_rl_params=n_params)
            row.update(ids)
            row.update(decision)
            rows.append(row)
    trainer.env = envs["train"]
    return rows, n_params, rl_train_time


def run_one(dataset, seed):
    t0 = time.time()
    tracemalloc.start()
    backbone = fit_aegis_backbone(dataset, seed=seed, row_cap=ROW_CAP, window_size=WINDOW_SIZE,
                                   telemetry_budget=TELEMETRY_BUDGET, epochs_neural=4, epochs_graph=4, top_k=3)
    _, peak_backbone = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    backbone_time = time.time() - t0

    cand_rows, ablation_rows = run_candidate_a_and_ablation(dataset, seed, backbone)
    for r in cand_rows:
        r["latency_sec"] = backbone_time
        r["peak_memory_MB"] = peak_backbone / 1e6

    student_ablation_rows, student_runtime_rows = run_distillation(dataset, seed, backbone)
    ablation_rows.extend(student_ablation_rows)

    rl_rows, n_rl_params, rl_train_time = run_rl(dataset, seed, backbone)
    for r in rl_rows:
        r["latency_sec"] = backbone_time + rl_train_time
        r["n_experts"] = backbone["n_experts"]
    cand_rows.extend(rl_rows)

    X_test = backbone["parts"]["test"]["X"]
    names = backbone["expert_names"]
    best_tree_name = max(backbone["tree_experts"].keys(),
                          key=lambda n: np.mean(backbone["val_prob_matrix"][:, names.index(n)]))
    tree_latency = measure_latency(lambda X: backbone["tree_experts"][best_tree_name].predict_proba(X), X_test)

    def full_ensemble_predict(X):
        from aegis.pipeline import _score_all_experts, compute_graph_context
        from aegis.router import build_context_matrix, route_predict
        pm = _score_all_experts(backbone["tree_experts"], backbone["neural_bundle"], backbone["graph_bundle"],
                                 X, backbone["parts"]["test"]["df"].iloc[:len(X)], backbone["spec"],
                                 backbone["parts"]["test"]["y"][:len(X)], names)
        return pm
    ensemble_latency = measure_latency(full_ensemble_predict, X_test, n_repeat=1)

    runtime_rows = [
        dict(dataset=dataset, seed=seed, model="BestSingleTreeExpert", n_params=np.nan,
             inference_latency_sec_per_sample=tree_latency, train_time_sec=np.nan, model_size_kb=np.nan),
        dict(dataset=dataset, seed=seed, model="Full9ExpertEnsemble+MoERouter", n_params=np.nan,
             inference_latency_sec_per_sample=ensemble_latency, train_time_sec=backbone_time,
             peak_memory_MB=peak_backbone / 1e6, model_size_kb=np.nan),
        dict(dataset=dataset, seed=seed, model="FullAegisGraphX_backbone+RL", n_params=n_rl_params,
             inference_latency_sec_per_sample=ensemble_latency, train_time_sec=backbone_time + rl_train_time,
             model_size_kb=np.nan),
    ]
    runtime_rows.extend(student_runtime_rows)

    return cand_rows, ablation_rows, runtime_rows


def main():
    all_cand, all_ablation, all_runtime = [], [], []
    for dataset in DATASETS:
        for seed in SEEDS:
            print(f"=== {dataset} seed={seed} ===", flush=True)
            try:
                cand_rows, ablation_rows, runtime_rows = run_one(dataset, seed)
                all_cand.extend(cand_rows)
                all_ablation.extend(ablation_rows)
                all_runtime.extend(runtime_rows)
            except Exception as e:
                import traceback; traceback.print_exc()
                all_cand.append(dict(dataset=dataset, seed=seed, candidate="ERROR", split="na",
                                      error=f"{type(e).__name__}: {e}"))
            pd.DataFrame(all_cand).to_csv(os.path.join(RESULTS_DIR, "aegis_candidates.csv"), index=False)
            pd.DataFrame(all_ablation).to_csv(os.path.join(RESULTS_DIR, "aegis_ablation.csv"), index=False)
            pd.DataFrame(all_runtime).to_csv(os.path.join(RESULTS_DIR, "aegis_runtime.csv"), index=False)
            print(f"  done, {len(all_cand)} candidate rows, {len(all_ablation)} ablation rows so far", flush=True)
    print("Saved aegis_candidates.csv, aegis_ablation.csv, aegis_runtime.csv")


if __name__ == "__main__":
    main()
