"""
Part 12, Experiment 8: low-label setting (Table XI).

Subsamples the TRAINING split's labels only (never touching val/test) to
1/5/10/25/50/100%, using a stratified subsample so both classes remain
represented at every fraction, retrains a representative tree model and a
small deep model, and evaluates on the untouched eval split. This directly
supports the paper's low-label-regime claim: does the proposed method's
information-acquisition/graph-structure advantage survive when only a small
fraction of training labels are available (a realistic SOC constraint)?
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from experiments.build_envs import build_all
from methods.metrics_utils import ids_metrics_from_scores
from baselines.deep_baselines import MLP, FlowDataset, train_and_eval

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]


def stratified_subsample(X, y, frac, seed):
    if frac >= 1.0:
        return X, y
    idx_all = np.arange(len(y))
    keep_idx, _ = train_test_split(idx_all, train_size=frac, random_state=seed, stratify=y)
    return X[keep_idx], y[keep_idx]


def run_low_label(dataset_name, seed=42, row_cap="default", eval_split="test"):
    bundle = build_all(dataset_name, row_cap=row_cap, seed=seed)
    parts = bundle["parts"]
    X_train_full, y_train_full = parts["train"]["X"], parts["train"]["y"]
    X_eval, y_eval = parts[eval_split]["X"], parts[eval_split]["y"]

    rows = []
    for frac in FRACTIONS:
        X_tr, y_tr = stratified_subsample(X_train_full, y_train_full, frac, seed)
        if len(np.unique(y_tr)) < 2:
            continue

        tree = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(X_tr, y_tr)
        scores = tree.predict_proba(X_eval)[:, 1]
        m_tree = ids_metrics_from_scores(y_eval, scores)
        m_tree.update({"model": "HistGradientBoosting", "family": "classical_tree",
                       "label_fraction": frac, "n_train_labels": len(y_tr),
                       "dataset": dataset_name, "seed": seed})
        rows.append(m_tree)

        class_weight = torch.tensor([1.0, max((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), 0.2)],
                                     dtype=torch.float32)
        mlp = MLP(X_tr.shape[1])
        trues, scores2, _, _, _ = train_and_eval(mlp, FlowDataset(X_tr, y_tr), FlowDataset(X_eval, y_eval),
                                                  epochs=8, class_weight=class_weight, seed=seed)
        m_mlp = ids_metrics_from_scores(trues, scores2)
        m_mlp.update({"model": "MLP", "family": "deep", "label_fraction": frac,
                     "n_train_labels": len(y_tr), "dataset": dataset_name, "seed": seed})
        rows.append(m_mlp)

        print(f"[{dataset_name}] frac={frac}: tree_macro_f1={m_tree['macro_f1']:.4f} "
              f"mlp_macro_f1={m_mlp['macro_f1']:.4f} n={len(y_tr)}", flush=True)
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["UNSW-NB15", "ToN-IoT-Network", "CICIoT2023"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--row_cap", default="default")
    ap.add_argument("--eval_split", default="test")
    args = ap.parse_args()
    row_cap = args.row_cap if args.row_cap == "default" else int(args.row_cap)

    all_rows = []
    out_path = os.path.join(RESULTS_DIR, "low_label.csv")
    if os.path.exists(out_path):
        all_rows = pd.read_csv(out_path).to_dict("records")
    for dataset in args.datasets:
        rows = run_low_label(dataset, seed=args.seed, row_cap=row_cap, eval_split=args.eval_split)
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"Saved {len(all_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
