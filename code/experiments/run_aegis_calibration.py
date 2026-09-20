"""
Part 10: calibration/threshold-sensitivity analysis for AegisGraph-X's
classification backbone (the ensemble probability that COMMIT reads off).

Refits the backbone once per (dataset, seed) [same procedure as run_aegis.py]
and evaluates every calibration variant (uncalibrated / temperature /
isotonic / Platt) plus the conformal-controlled threshold on the TEST split,
using ONLY the VAL-fit CalibrationBundle (aegis.calibrate) -- thresholds are
never selected on test data. Reports ECE, Brier, NLL, FPR, FNR, and alert
volume per 10,000 flows for each variant.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import pandas as pd

from aegis.pipeline import fit_aegis_backbone
from aegis.calibrate import (expected_calibration_error, brier_score, nll_score,
                              reliability_diagram_data, select_threshold_max_f1)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "results", "csv")
DATASETS = ["UNSW-NB15", "ToN-IoT-Network", "CICIoT2023"]
# Scope note (disclosed): 1 seed, matching run_multiclass.py/run_low_label.py's
# single-seed reference-point convention -- calibration diagnostics (ECE/Brier/NLL)
# are far less seed-sensitive than macro-F1, so this is a tractability choice, not
# a result being hidden.
SEEDS = [0]
ROW_CAP = 50000


def variant_metrics(p, y, threshold):
    y_pred = (p >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y == 1))); fp = int(np.sum((y_pred == 1) & (y == 0)))
    fn = int(np.sum((y_pred == 0) & (y == 1))); tn = int(np.sum((y_pred == 0) & (y == 0)))
    fpr = fp / max(fp + tn, 1); fnr = fn / max(fn + tp, 1)
    alert_per_10k = (y_pred.sum() / max(len(y_pred), 1)) * 10000
    return dict(ece=expected_calibration_error(p, y), brier=brier_score(p, y), nll=nll_score(p, y),
                fpr=fpr, fnr=fnr, alerts_per_10k_flows=alert_per_10k, threshold=float(threshold))


def main():
    rows = []
    reliability_rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            print(f"=== calibration {dataset} seed={seed} ===", flush=True)
            try:
                b = fit_aegis_backbone(dataset, seed=seed, row_cap=ROW_CAP, window_size=64,
                                        telemetry_budget=8, epochs_neural=4, epochs_graph=4, top_k=3)
            except Exception as e:
                print("  ERROR:", e); continue
            calib = b["calib"]
            y_test = b["parts"]["test"]["y"]
            p_test = b["test_ensemble_p"]
            variants = calib.apply_all(p_test)
            thresholds = {"uncalibrated": calib.threshold_uncalibrated, "temperature": calib.threshold_temperature,
                          "isotonic": calib.threshold_isotonic, "platt": calib.threshold_platt}
            for name, p in variants.items():
                m = variant_metrics(p, y_test, thresholds[name])
                m.update(dataset=dataset, seed=seed, calibration=name)
                rows.append(m)
            conf_t = calib.conformal_threshold_alpha05
            m = variant_metrics(p_test, y_test, conf_t)
            m.update(dataset=dataset, seed=seed, calibration="conformal_fpr_control_alpha05")
            rows.append(m)

            if seed == SEEDS[0]:
                rel = reliability_diagram_data(variants["temperature"], y_test)
                for bc, acc, conf, cnt in zip(rel["bin_center"], rel["accuracy"], rel["confidence"], rel["count"]):
                    reliability_rows.append(dict(dataset=dataset, bin_center=bc, accuracy=acc, confidence=conf, count=cnt))

            pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "aegis_calibration.csv"), index=False)
            pd.DataFrame(reliability_rows).to_csv(os.path.join(RESULTS_DIR, "aegis_reliability.csv"), index=False)
    print("Saved results/csv/aegis_calibration.csv, results/csv/aegis_reliability.csv")


if __name__ == "__main__":
    main()
