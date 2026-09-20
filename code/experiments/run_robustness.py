"""
Part 12, Experiment 10: robustness / distribution-shift table (Table XIII).

Trains a representative subset of baselines (best available tree model +
best available deep model) once on TRAIN, then re-evaluates them under each
perturbation in perturbations.PERTURBATIONS applied to the (already
train-fit-scaled) EVAL split only. The winning RL candidate is added to this
same table by run_robustness_for_env() once Part 6 screening has identified it
(see code/experiments/finalize_winner.py), reusing the identical perturbation
set so Table XIII compares the proposed method and baselines under exactly
the same stressors.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd

from experiments.build_envs import build_all
from experiments.perturbations import PERTURBATIONS
from methods.metrics_utils import ids_metrics_from_scores
from sklearn.ensemble import HistGradientBoostingClassifier
import torch

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "csv")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_robustness_tabular(dataset_name, seed=42, row_cap="default", eval_split="test"):
    bundle = build_all(dataset_name, row_cap=row_cap, seed=seed)
    parts = bundle["parts"]
    X_train, y_train = parts["train"]["X"], parts["train"]["y"]
    X_eval0, y_eval0 = parts[eval_split]["X"], parts[eval_split]["y"]

    tree_model = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(X_train, y_train)

    from baselines.deep_baselines import MLP, FlowDataset, train_and_eval
    class_weight = torch.tensor([1.0, max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 0.2)],
                                 dtype=torch.float32)
    mlp = MLP(X_train.shape[1])
    tr_ds = FlowDataset(X_train, y_train)

    rng = np.random.default_rng(seed)
    rows = []
    for pert_name, pert_fn in PERTURBATIONS.items():
        X_p, y_p = pert_fn(X_eval0.copy(), y_eval0.copy(), rng)

        scores_tree = tree_model.predict_proba(X_p)[:, 1]
        m_tree = ids_metrics_from_scores(y_p, scores_tree)
        m_tree.update({"model": "HistGradientBoosting", "family": "classical_tree",
                       "perturbation": pert_name, "dataset": dataset_name, "seed": seed})
        rows.append(m_tree)

        from baselines.deep_baselines import FlowDataset as FD
        ev_ds = FD(X_p, y_p)
        trues, scores, _, _, _ = train_and_eval(mlp, tr_ds, ev_ds, epochs=6, class_weight=class_weight, seed=seed)
        m_mlp = ids_metrics_from_scores(trues, scores)
        m_mlp.update({"model": "MLP", "family": "deep", "perturbation": pert_name,
                     "dataset": dataset_name, "seed": seed})
        rows.append(m_mlp)

        print(f"[{dataset_name}] {pert_name}: tree_macro_f1={m_tree['macro_f1']:.4f} "
              f"mlp_macro_f1={m_mlp['macro_f1']:.4f}", flush=True)
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
    out_path = os.path.join(RESULTS_DIR, "robustness.csv")
    if os.path.exists(out_path):
        all_rows = pd.read_csv(out_path).to_dict("records")
    for dataset in args.datasets:
        rows = run_robustness_tabular(dataset, seed=args.seed, row_cap=row_cap, eval_split=args.eval_split)
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"Saved {len(all_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
