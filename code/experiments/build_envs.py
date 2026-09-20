"""Shared helper: dataset name -> (env_train, env_val, env_test, X, y, train_df, val_df,
test_df, spec) built through the leakage-safe split -> train-only fit -> graph
construction -> POMDP env pipeline. Used by every experiment script so the exact
same preprocessing is never re-derived ad hoc per script."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from preprocessing.leakage_safe_split import leakage_safe_split, fit_transform_train_only
from graphs.graph_builder import build_graph_sequence
from env.pomdp_env import IDSPOMDPEnv, EnvConfig, fit_probe

DATASET_ROW_CAPS = {
    "CICIoT2023": 600000,     # subsampled per native split (see dataset_selection_report.md)
    "ToN-IoT-Network": None,
    "UNSW-NB15": None,
    "RT-IoT2022": None,
}


def host_key_series(df, spec):
    if spec.src_ip_col and spec.src_ip_col in df.columns:
        return df[spec.src_ip_col].astype(str).values
    if spec.src_port_col and spec.src_port_col in df.columns:
        return df[spec.src_port_col].astype(str).values
    return np.arange(len(df)).astype(str)


def build_all(dataset_name, window_size=64, telemetry_budget=8, seed=42, row_cap="default",
              reward_weights=None):
    """reward_weights: optional dict of env.pomdp_env.RewardWeights field overrides
    (e.g. {"w_fn": 2.0, "w_fp": 1.0}), used by tune_hyperparams.py's Part-11 search
    over reward weights; defaults to RewardWeights()'s built-in values when None."""
    from env.pomdp_env import RewardWeights
    cap = DATASET_ROW_CAPS.get(dataset_name) if row_cap == "default" else row_cap
    df, splits, spec, manifest = leakage_safe_split(dataset_name, max_total_rows=cap, random_state=seed)
    X, feat_names, meta = fit_transform_train_only(df, splits, spec)

    parts = {}
    for name, idx in splits.items():
        part_df = df.iloc[idx].reset_index(drop=True)
        y_bin = df["__y_bin__"].iloc[idx].values
        y_multi = df["__y_multi__"].iloc[idx].values if "__y_multi__" in df.columns else None
        graphs, flow_index, graph_type = build_graph_sequence(part_df, spec, window_size=window_size)
        host_keys = host_key_series(part_df, spec)
        parts[name] = dict(df=part_df, X=X[name], y=y_bin, y_multi=y_multi,
                            graphs=graphs, flow_index=flow_index, host_keys=host_keys)

    probe = fit_probe(X["train"], parts["train"]["y"], seed=seed)

    envs = {}
    for name in ["train", "val", "test"]:
        p = parts[name]
        rw = RewardWeights(**reward_weights) if reward_weights else RewardWeights()
        cfg = EnvConfig(window_size=window_size, telemetry_budget=telemetry_budget, seed=seed, reward=rw)
        envs[name] = IDSPOMDPEnv(p["X"], p["y"], p["host_keys"], p["flow_index"], p["graphs"],
                                  probe, y_multi=p["y_multi"], config=cfg)

    return dict(envs=envs, parts=parts, spec=spec, feat_names=feat_names, probe=probe,
                manifest=manifest)
