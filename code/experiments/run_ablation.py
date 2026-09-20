"""
Part 12, Experiment 9 / Table XII: ablation study for the selected method.

Two ablations are universal (meaningful for any of the 10 candidates) and
implemented generically here:
  - "no_graph_encoder": zero out the graph-embedding contribution to every
    observation (TinyGraphEncoder.window_summary replaced by a zero vector),
    isolating how much the graph structure actually contributes.
  - "no_rl_policy": replace the learned sequential policy with the
    non-sequential static-threshold rule policy (classify immediately at 0.5
    on the shared probe, never using INFO/ESCALATE/ISOLATE/DEFER), isolating
    how much the sequential/RL formulation itself contributes over a plain
    classifier using the same underlying signal.

Candidate-specific ablations (no uncertainty calibration / no budget
constraint / no symbolic constraints / no world model / no offline safety
constraint / no active acquisition, "if used" per the task spec) are added
via `CANDIDATE_SPECIFIC_ABLATIONS` once select_winner.py has named the winner,
since the task spec itself only requires ablating modules the winning method
actually has.
"""
import sys, os, json, importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
import torch

from experiments.build_envs import build_all
from experiments.tune_hyperparams import CANDIDATE_MODULES
from experiments.run_final_comparison import run_selected_method, load_winner
from baselines.rl_baselines import run_rule_policy, static_threshold_policy
from methods.core import TinyGraphEncoder

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "csv")

# candidate_name -> {ablation_label: kwargs override passed to that candidate's run()}
# Populated once the winner is known; e.g. for C4 (Budgeted Active Acquisition):
#   {"no_active_acquisition": {"info_gain_weight": 0.0}}
# for C1 (Conformal): {"no_uncertainty_calibration": {"alpha": 0.999}}  (alpha->no effective set)
# for C8 (Neuro-symbolic): requires passing mask_fn=None, handled via a dedicated code path.
CANDIDATE_SPECIFIC_ABLATIONS = {
    "C4_BudgetedActiveAcquisitionRL": {"no_active_acquisition": {"info_gain_weight": 0.0}},
    "C1_ConformalRiskGraphRL": {"no_uncertainty_calibration": {"alpha": 0.999}},
}


def run_no_graph_encoder(candidate_name, dataset_name, seed, row_cap):
    """Monkeypatch TinyGraphEncoder.window_summary to return zeros for the
    duration of this call, then run the candidate exactly as in the final
    comparison -- isolating the graph encoder's contribution."""
    original = TinyGraphEncoder.window_summary

    def zeroed(self, graph):
        return torch.zeros(self.out.out_features)

    TinyGraphEncoder.window_summary = zeroed
    try:
        res = run_selected_method(candidate_name, dataset_name, seed, row_cap)
    finally:
        TinyGraphEncoder.window_summary = original
    return res


def run_no_rl_policy(dataset_name, seed, row_cap):
    bundle = build_all(dataset_name, row_cap=row_cap, seed=seed)
    return run_rule_policy(bundle["envs"]["test"], static_threshold_policy)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["UNSW-NB15", "ToN-IoT-Network", "CICIoT2023"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--row_cap", type=int, default=60000)
    args = ap.parse_args()

    winner = load_winner()
    print("Ablating selected method:", winner)
    rows = []
    out_path = os.path.join(RESULTS_DIR, "ablation.csv")
    if os.path.exists(out_path):
        rows = pd.read_csv(out_path).to_dict("records")

    for dataset in args.datasets:
        for seed in args.seeds:
            full = run_selected_method(winner, dataset, seed, args.row_cap)
            full.update({"ablation": "full_method", "dataset": dataset, "seed": seed, "model": winner})
            rows.append(full)

            no_ge = run_no_graph_encoder(winner, dataset, seed, args.row_cap)
            no_ge.update({"ablation": "no_graph_encoder", "dataset": dataset, "seed": seed, "model": winner})
            rows.append(no_ge)

            no_rl = run_no_rl_policy(dataset, seed, args.row_cap)
            no_rl.update({"ablation": "no_rl_policy", "dataset": dataset, "seed": seed, "model": winner})
            rows.append(no_rl)

            for label, override in CANDIDATE_SPECIFIC_ABLATIONS.get(winner, {}).items():
                print(f"  candidate-specific ablation '{label}' requires manual wiring per "
                      f"candidate signature -- see CANDIDATE_SPECIFIC_ABLATIONS.", flush=True)

            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"[{dataset}] seed={seed}: full={full.get('macro_f1')} "
                  f"no_graph_encoder={no_ge.get('macro_f1')} no_rl_policy={no_rl.get('macro_f1')}", flush=True)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
