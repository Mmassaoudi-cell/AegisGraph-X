"""
Part 8/Experiment 3-4: run all ~37 baselines (12 classical/tree + 9 deep +
5 graph + 11 RL) on a given dataset's leakage-safe splits, at a chosen
eval_split ('val' for anything used in model/threshold selection, 'test' for
the final reported comparison), and append rows to results/csv/baselines.csv.
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
import numpy as np

from experiments.build_envs import build_all
from graphs.graph_builder import build_feature_rich_graphs
from baselines import classical_baselines, deep_baselines, graph_baselines, rl_baselines

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "csv")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_CSV = os.path.join(RESULTS_DIR, "baselines.csv")


def run_for_dataset(dataset_name, seed=42, eval_split="test", row_cap="default",
                     include=("classical", "deep", "graph", "rl")):
    bundle = build_all(dataset_name, row_cap=row_cap, seed=seed)
    parts = bundle["parts"]
    rows = []

    if "classical" in include:
        print(f"[{dataset_name}] classical/tree baselines...", flush=True)
        res = classical_baselines.run_all(parts["train"]["X"], parts["train"]["y"],
                                           parts["val"]["X"], parts["val"]["y"],
                                           parts["test"]["X"], parts["test"]["y"],
                                           seed=seed, eval_split=eval_split)
        for r in res:
            r.update({"dataset": dataset_name, "seed": seed})
        rows.extend(res)

    if "deep" in include:
        print(f"[{dataset_name}] deep baselines...", flush=True)
        eval_part = parts[eval_split]
        res = deep_baselines.run_all(parts["train"]["X"], parts["train"]["y"],
                                      eval_part["X"], eval_part["y"], seed=seed, eval_split=eval_split)
        for r in res:
            r.update({"dataset": dataset_name, "seed": seed})
        rows.extend(res)

    if "graph" in include:
        print(f"[{dataset_name}] graph baselines...", flush=True)
        spec = bundle["spec"]
        train_graphs, train_entities = build_feature_rich_graphs(
            parts["train"]["df"], spec, parts["train"]["X"], parts["train"]["y"], window_size=64)
        eval_part = parts[eval_split]
        eval_graphs, eval_entities = build_feature_rich_graphs(
            eval_part["df"], spec, eval_part["X"], eval_part["y"], window_size=64)
        in_dim = parts["train"]["X"].shape[1]
        res = graph_baselines.run_all(train_graphs, eval_graphs, in_dim, in_dim,
                                       train_entities, eval_entities, seed=seed, eval_split=eval_split)
        for r in res:
            r.update({"dataset": dataset_name, "seed": seed})
        rows.extend(res)

    if "rl" in include:
        print(f"[{dataset_name}] RL baselines...", flush=True)
        envs = bundle["envs"]
        eval_env = envs[eval_split]
        res_dict = rl_baselines.run_all(envs["train"], eval_env, seed=seed)
        for name, r in res_dict.items():
            r.update({"model": name, "family": "rl", "dataset": dataset_name, "seed": seed,
                      "eval_split": eval_split})
            rows.append(r)

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["UNSW-NB15", "ToN-IoT-Network", "CICIoT2023"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--eval_split", default="test")
    ap.add_argument("--row_cap", default="default")
    ap.add_argument("--include", nargs="+", default=["classical", "deep", "graph", "rl"])
    args = ap.parse_args()

    all_rows = []
    if os.path.exists(OUT_CSV):
        all_rows = pd.read_csv(OUT_CSV).to_dict("records")

    row_cap = args.row_cap
    if row_cap != "default":
        row_cap = int(row_cap)

    for dataset in args.datasets:
        for seed in args.seeds:
            rows = run_for_dataset(dataset, seed=seed, eval_split=args.eval_split,
                                    row_cap=row_cap, include=tuple(args.include))
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
    print(f"Saved {len(all_rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
