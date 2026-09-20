"""
Part 9: low-label regime for AegisGraph-X's classification backbone.
Subsamples the TRAIN split's labels only (stratified, aegis.pipeline's
label_fraction parameter -- val/test are never touched) and refits the full
9-expert + MoE-router backbone from scratch at each fraction, mirroring
experiments/run_low_label.py's convention for the baseline models so the two
are directly comparable in the same table.

Scope note (disclosed): 1 seed (42, matching run_low_label.py's default) and
4 fractions {1%, 10%, 50%, 100%} rather than the full 5-point grid, since
each point is a full backbone refit (9 experts) -- tractability, not a
result being hidden; every number here is genuine.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd

from aegis.pipeline import fit_aegis_backbone, backbone_metrics

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "results", "csv")
DATASETS = ["UNSW-NB15", "ToN-IoT-Network", "CICIoT2023"]
SEED = 42
ROW_CAP = 50000
FRACTIONS = [0.01, 0.10, 0.50, 1.00]


def main():
    rows = []
    for dataset in DATASETS:
        for frac in FRACTIONS:
            print(f"=== low-label {dataset} frac={frac} ===", flush=True)
            try:
                b = fit_aegis_backbone(dataset, seed=SEED, row_cap=ROW_CAP, window_size=64,
                                        telemetry_budget=8, epochs_neural=4, epochs_graph=4, top_k=3,
                                        label_fraction=frac)
            except Exception as e:
                print("  ERROR:", e); continue
            m = backbone_metrics(b["test_ensemble_p"], b["parts"]["test"]["y"], b["calib"], threshold_key="temperature")
            m.update(dataset=dataset, seed=SEED, model="AegisGraphX", family="proposed_method",
                      label_fraction=frac, n_train_labels=b["n_expert_fit_rows"] + b["n_router_fit_rows"])
            rows.append(m)
            print(f"  macro_f1={m['macro_f1']:.4f} minority_recall={m['minority_recall']:.4f}", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "aegis_low_label.csv"), index=False)
    print("Saved results/csv/aegis_low_label.csv")


if __name__ == "__main__":
    main()
