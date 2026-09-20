"""Smoke test: leakage-safe split -> train-only fit -> graph construction -> POMDP env,
run with a random policy for a handful of episodes, on a real (small) dataset."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from preprocessing.leakage_safe_split import leakage_safe_split, fit_transform_train_only
from graphs.graph_builder import build_graph_sequence
from env.pomdp_env import IDSPOMDPEnv, EnvConfig, N_ACTIONS, fit_probe


def host_key_series(df, spec):
    if spec.src_ip_col and spec.src_ip_col in df.columns:
        return df[spec.src_ip_col].astype(str).values
    if spec.src_port_col and spec.src_port_col in df.columns:
        return df[spec.src_port_col].astype(str).values
    return np.arange(len(df)).astype(str)


def main(dataset_name):
    print(f"=== {dataset_name} ===")
    df, splits, spec, manifest = leakage_safe_split(dataset_name, max_total_rows=60000)
    X, feat_names, meta = fit_transform_train_only(df, splits, spec)
    print("feature dim:", X["train"].shape[1], "| train/val/test sizes:",
          {k: v.shape[0] for k, v in X.items()})

    val_df = df.iloc[splits["val"]].reset_index(drop=True)
    graphs, flow_index, graph_type = build_graph_sequence(val_df, spec, window_size=64)
    print("graph_type:", graph_type, "| n windows:", len(graphs))

    y_val = df["__y_bin__"].iloc[splits["val"]].values
    y_train = df["__y_bin__"].iloc[splits["train"]].values
    probe = fit_probe(X["train"], y_train)

    host_keys = host_key_series(val_df, spec)
    env = IDSPOMDPEnv(X["val"], y_val, host_keys, flow_index, graphs, probe, config=EnvConfig(window_size=64))

    rng = np.random.default_rng(0)
    n_episodes = 5
    for ep in range(n_episodes):
        obs = env.reset(episode_idx=ep)
        done = False
        steps = 0
        while not done:
            a = rng.integers(0, N_ACTIONS)
            obs, r, done, info = env.step(a)
            steps += 1
            if steps > 2000:
                raise RuntimeError("episode did not terminate -- infinite loop bug")
        print(f"  episode {ep}: steps={steps} summary={env.episode_summary()}")
    print("OK\n")


if __name__ == "__main__":
    for name in ["UNSW-NB15", "RT-IoT2022", "ToN-IoT-Network"]:
        main(name)
