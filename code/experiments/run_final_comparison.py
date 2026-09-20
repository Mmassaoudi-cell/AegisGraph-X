"""
Part 12, Experiments 3-7: final repeated-seed comparison of the selected
candidate (named in manifests/candidate_selection_report.json, produced by
select_winner.py from VALIDATION-only screening) against baselines, on the
TEST split only (the one split untouched by screening/tuning). Appends rows
to results/csv/baselines.csv (family="selected_method") so make_tables_figures.py
and statistical_tests.py can treat it identically to every baseline row.
"""
import sys, os, json, argparse, importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd

from experiments.build_envs import build_all
from experiments.tune_hyperparams import CANDIDATE_MODULES
from methods.metrics_utils import ids_metrics_from_scores  # noqa: F401 (re-exported for downstream use)

MANIFEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "manifests")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "csv")


def load_winner():
    path = os.path.join(MANIFEST_DIR, "candidate_selection_report.json")
    with open(path) as f:
        report = json.load(f)
    return report["selected_candidate"]


def load_tuned_params(candidate_name):
    """Loads the Optuna best_params from tune_hyperparams.py's output for this
    candidate, if present (any dataset's tuning log -- see run_selected_method
    for how these are applied uniformly across datasets). Returns {} if no
    tuning has been run yet, in which case run_selected_method falls back to
    its own untuned defaults rather than failing."""
    if not os.path.isdir(MANIFEST_DIR):
        return {}
    for fn in os.listdir(MANIFEST_DIR):
        if fn.startswith(f"tuning_log_{candidate_name}_") and fn.endswith(".json"):
            with open(os.path.join(MANIFEST_DIR, fn)) as f:
                return json.load(f).get("best_params", {})
    return {}


def run_selected_method(candidate_name, dataset_name, seed, row_cap, n_updates=None, episodes_per_update=None,
                         tuned_params=None):
    mod = importlib.import_module(CANDIDATE_MODULES[candidate_name])
    tuned_params = tuned_params if tuned_params is not None else load_tuned_params(candidate_name)
    n_updates = n_updates or tuned_params.get("n_updates", 25)
    episodes_per_update = episodes_per_update or tuned_params.get("episodes_per_update", 8)
    window_size = tuned_params.get("window_size", 64)
    telemetry_budget = tuned_params.get("telemetry_budget", 8)
    reward_weights = {k: tuned_params[k] for k in ("w_tp", "w_fn", "w_fp") if k in tuned_params} or None

    bundle = build_all(dataset_name, row_cap=row_cap, seed=seed, window_size=window_size,
                        telemetry_budget=telemetry_budget, reward_weights=reward_weights)
    env_train, env_test = bundle["envs"]["train"], bundle["envs"]["test"]
    kwargs = dict(n_updates=n_updates, episodes_per_update=episodes_per_update, seed=seed)

    if candidate_name == "C1_ConformalRiskGraphRL":
        alpha = tuned_params.get("conformal_alpha", 0.1)
        y_calib = bundle["parts"]["val"]["y"][: max(len(bundle["parts"]["val"]["y"]) // 3, 50)]
        X_calib = bundle["parts"]["val"]["X"][: len(y_calib)]
        res = mod.run(env_train, env_test, X_calib, y_calib, bundle["probe"], alpha=alpha, **kwargs)
    elif candidate_name == "C4_BudgetedActiveAcquisitionRL":
        info_gain_weight = tuned_params.get("info_gain_weight", 0.4)
        res = mod.run(env_train, env_test, info_gain_weight=info_gain_weight, **kwargs)
    elif candidate_name == "C3_OfflineSafeGraphRL":
        cql_alpha = tuned_params.get("cql_alpha", 1.0)
        n_episodes_offline = tuned_params.get("n_episodes_offline", 200)
        epochs = tuned_params.get("epochs", 12)
        behavior_epsilon = tuned_params.get("behavior_epsilon", 0.3)
        res = mod.run(env_train, env_test, cql_alpha=cql_alpha, n_episodes_offline=n_episodes_offline,
                      epochs=epochs, seed=seed, behavior_epsilon=behavior_epsilon)
    elif candidate_name == "C6_ContinualMetaGraphRL":
        train_df = bundle["parts"]["train"]["df"]
        y_multi_col = None
        for cand_col in ["type", "attack_cat", "Attack_type", "label"]:
            if cand_col in train_df.columns and train_df[cand_col].nunique() > 2:
                y_multi_col = cand_col
                break
        res = mod.run(env_train, env_test, train_df=train_df, y_multi_col=y_multi_col, **kwargs)
    elif candidate_name == "C8_NeuroSymbolicATTCKRL":
        y_multi_train = bundle["parts"]["train"]["y_multi"]
        res = mod.run(env_train, env_test, X_train=bundle["parts"]["train"]["X"],
                      y_multi_train=y_multi_train, **kwargs)
    elif candidate_name == "C9_SelfSupervisedWorldModelRL":
        res = mod.run(env_train, env_test, bundle["parts"]["train"]["y"], n_pretrain_steps=300, **kwargs)
    elif candidate_name == "C10_ByzantineRobustFederatedRL":
        import numpy as np
        from env.pomdp_env import IDSPOMDPEnv, EnvConfig
        n_flows = env_train.n_flows
        k_sites = 5
        bounds = np.linspace(0, n_flows, k_sites + 1).astype(int)
        p = bundle["parts"]["train"]

        def build_shard(site):
            lo, hi = bounds[site], bounds[site + 1]
            cfg = EnvConfig(window_size=64, telemetry_budget=8, seed=seed + site)
            return IDSPOMDPEnv(p["X"][lo:hi], p["y"][lo:hi], p["host_keys"][lo:hi],
                                p["flow_index"][lo:hi], p["graphs"], bundle["probe"], config=cfg)
        res = mod.run(build_shard, env_test, k_sites=k_sites, n_malicious=1, n_rounds=8,
                      local_updates_per_round=3, episodes_per_update=6, seed=seed)
    else:
        res = mod.run(env_train, env_test, **kwargs)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["UNSW-NB15", "ToN-IoT-Network", "CICIoT2023"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--row_cap", type=int, default=100000)
    args = ap.parse_args()

    winner = load_winner()
    print("Selected method:", winner)

    out_path = os.path.join(RESULTS_DIR, "baselines.csv")
    rows = []
    if os.path.exists(out_path):
        rows = pd.read_csv(out_path).to_dict("records")

    for dataset in args.datasets:
        for seed in args.seeds:
            print(f"[{dataset}] selected method ({winner}) seed={seed} ...", flush=True)
            try:
                res = run_selected_method(winner, dataset, seed, args.row_cap)
                res.update({"model": winner, "family": "selected_method", "dataset": dataset,
                           "seed": seed, "eval_split": "test"})
            except Exception as e:
                res = {"model": winner, "family": "selected_method", "dataset": dataset,
                      "seed": seed, "eval_split": "test", "error": str(e)}
            rows.append(res)
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"  -> macro_f1={res.get('macro_f1')} reward={res.get('cumulative_reward_per_window')}",
                  flush=True)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
