"""
Part 9: robustness / distribution-shift stress test for AegisGraph-X's
classification backbone, reusing experiments/perturbations.py's PERTURBATIONS
grid (feature noise, missing-feature injection, label noise, class-imbalance
stress, clean baseline) applied to the TEST split ONLY at evaluation time
(training/calibration are never touched).

Perturbing X_test (used by the tree/neural experts and as the graph expert's
node/edge-attribute features) while keeping the TEST df's raw
IP/port/protocol identity columns intact preserves graph topology under
perturbation -- exactly what a real feature-corruption/telemetry-loss event
would look like (structure survives, feature values degrade) -- so no
backbone refit is needed per perturbation: one backbone fit per
(dataset, seed) is reused for every perturbation in the grid.

Scope note (disclosed): run at 2 seeds (0, 1) rather than the full study's
5, for tractability -- the same kind of bounded-but-disclosed reduction used
throughout this project (see manifests/technical_audit_report.md).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import pandas as pd

from aegis.pipeline import fit_aegis_backbone, simple_classification_metrics, _score_all_experts
from aegis.router import build_context_matrix, route_predict
from aegis.calibrate import select_threshold_max_f1
from experiments.perturbations import PERTURBATIONS

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "results", "csv")
DATASETS = ["UNSW-NB15", "ToN-IoT-Network", "CICIoT2023"]
SEEDS = [0, 1]
ROW_CAP = 50000


def score_perturbed(backbone, X_pert, y_pert):
    from aegis.pipeline import compute_graph_context
    test = backbone["parts"]["test"]
    n = len(X_pert)
    df_use = test["df"].iloc[:n].reset_index(drop=True)
    prob_matrix = _score_all_experts(backbone["tree_experts"], backbone["neural_bundle"], backbone["graph_bundle"],
                                      X_pert, df_use, backbone["spec"], y_pert, backbone["expert_names"])
    ctx = compute_graph_context(test["flow_index"][:n], test["graphs"], np.arange(n))
    raw_summary = X_pert[:, :min(5, X_pert.shape[1])]
    context = build_context_matrix(prob_matrix, ctx, raw_summary)
    p, _ = route_predict(backbone["router"], context, prob_matrix)
    return p


def main():
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            print(f"=== robustness {dataset} seed={seed} ===", flush=True)
            try:
                b = fit_aegis_backbone(dataset, seed=seed, row_cap=ROW_CAP, window_size=64,
                                        telemetry_budget=8, epochs_neural=4, epochs_graph=4, top_k=3)
            except Exception as e:
                print("  ERROR:", e); continue
            y_val = b["parts"]["val"]["y"]
            t = select_threshold_max_f1(b["val_ensemble_p"], y_val)
            X_test, y_test = b["parts"]["test"]["X"], b["parts"]["test"]["y"]
            rng = np.random.default_rng(seed)
            for pname, pfn in PERTURBATIONS.items():
                X_p, y_p = pfn(X_test, y_test, rng)
                p = score_perturbed(b, X_p, y_p)
                m = simple_classification_metrics(p, y_p, t)
                m.update(dataset=dataset, seed=seed, model="AegisGraphX", perturbation=pname)
                rows.append(m)
                print(f"  {pname}: macro_f1={m['macro_f1']:.4f} minority_recall={m['minority_recall']:.4f}", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "aegis_robustness.csv"), index=False)
    print("Saved results/csv/aegis_robustness.csv")


if __name__ == "__main__":
    main()
