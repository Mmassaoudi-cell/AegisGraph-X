"""Analytic (closed-form) decision-theoretic baseline for the budgeted
investigation layer -- the single most-requested missing comparison from
the Round-2 review (EIC, Methodology, Domain, Perspective, and the Devil's
Advocate all independently flagged it; the manuscript's own Limitations
section already names it as "the single most important missing comparison
in the paper").

Once Layer 5 produces a calibrated posterior p = P(attack | flow) and the
reward table (AegisRewardWeights) is fixed, the expected-reward-maximizing
action for a single, immediate (zero-telemetry, zero-delay) decision is
available in closed form -- no training required. This script computes that
policy directly from each dataset's calibrated ensemble_proba (the exact
same posterior AegisGraph-X's RL layer conditions on) and evaluates it with
the identical classification/decision-cost metrics used for Candidates
A/B/C, so the analytic policy is directly comparable to both.

Five candidate actions, each scored by the same per-flow reward terms the
RL environment uses (env/pomdp_env_aegis.py, AegisRewardWeights):
  COMMIT=1  E(p) = p*(w_tp + w_minority_bonus) - (1-p)*w_fp
  COMMIT=0  E(p) = -p*w_fn + (1-p)*(w_tp*0.3)
  ISOLATE   E(p) = p*r_isolate_malicious - (1-p)*c_isolate_benign
  ABSTAIN   E    = 0
  ESCALATE  E    = -c_escalate  (escalation is modeled as always resolving
                                  correctly at a fixed cost; see Limitations)
The analytic policy commits immediately (no telemetry request), consistent
with the paper's own finding that telemetry acquisition has zero value of
information in this environment (Section IV-F) -- so the delay bonus is
evaluated at n_steps=0 throughout, matching the RL policies' typical
(near-zero-delay) behavior on this environment.

This is a per-flow myopic policy: it does not model isolation's cross-flow
containment consequence (a real host-level effect on ToN-IoT-Network only,
Section IV-F) beyond the immediate reward term, so it is not claimed to be
the *sequential* optimum, only the single-decision optimum -- exactly the
comparison point the paper's Limitations section asks for.
"""
import sys, os, gc
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix

from aegis.pipeline import fit_aegis_backbone
from env.pomdp_env_aegis import AegisRewardWeights

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "results", "csv")
DATASETS = ["UNSW-NB15", "ToN-IoT-Network", "CICIoT2023"]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
ROW_CAP = 100000  # matches the pinned, row-cap-corrected budget used for the primary comparisons

A_COMMIT1, A_COMMIT0, A_ISOLATE, A_ABSTAIN, A_ESCALATE = 0, 1, 2, 3, 4
ACTION_NAMES = {A_COMMIT1: "commit1", A_COMMIT0: "commit0", A_ISOLATE: "isolate",
                A_ABSTAIN: "abstain", A_ESCALATE: "escalate"}


def analytic_policy(p, rw: AegisRewardWeights):
    """Vectorized closed-form argmax over the five terminal actions."""
    delay_bonus = rw.w_delay  # exp(-0/tau) = 1, n_steps=0 (immediate commit)
    e_commit1 = p * (rw.w_tp + rw.w_minority_bonus + delay_bonus) - (1 - p) * rw.w_fp
    e_commit0 = -p * rw.w_fn + (1 - p) * (rw.w_tp * 0.3)
    e_isolate = p * rw.r_isolate_malicious - (1 - p) * rw.c_isolate_benign
    e_abstain = np.zeros_like(p)
    e_escalate = np.full_like(p, -rw.c_escalate)
    stacked = np.stack([e_commit1, e_commit0, e_isolate, e_abstain, e_escalate], axis=1)
    action = np.argmax(stacked, axis=1)
    expected_reward = stacked[np.arange(len(p)), action]
    return action, expected_reward


def score(y_true, p, action, rw: AegisRewardWeights):
    """Realized reward and classification metrics under the true labels
    (as opposed to the expected reward the policy optimized against)."""
    n = len(y_true)
    predicted = np.full(n, -1, dtype=int)  # -1 = abstain (excluded from classification metrics)
    predicted[action == A_COMMIT1] = 1
    predicted[action == A_COMMIT0] = 0
    predicted[action == A_ISOLATE] = 1  # isolate always resolves as "flagged attack" bookkeeping
    # escalate: excluded from classification metrics (resolved by an oracle analyst), matches
    # the RL rollout convention in aegis/rl_policy.py:aegis_rollout_classification

    realized = np.zeros(n)
    tp = (predicted == 1) & (y_true == 1)
    tn = (predicted == 0) & (y_true == 0)
    fp = (predicted == 1) & (y_true == 0)
    fn = (predicted == 0) & (y_true == 1)
    realized[tp & (action == A_COMMIT1)] = rw.w_tp + rw.w_minority_bonus + rw.w_delay
    realized[tn & (action == A_COMMIT0)] = rw.w_tp * 0.3
    realized[fp & (action == A_COMMIT1)] = -rw.w_fp
    realized[fn & (action == A_COMMIT0)] = -rw.w_fn
    iso = action == A_ISOLATE
    realized[iso & (y_true == 1)] = rw.r_isolate_malicious
    realized[iso & (y_true == 0)] = -rw.c_isolate_benign
    esc = action == A_ESCALATE
    realized[esc] = -rw.c_escalate
    ab = action == A_ABSTAIN
    realized[ab] = 0.0

    mask_classified = predicted >= 0
    y_c, p_c = y_true[mask_classified], predicted[mask_classified]
    if mask_classified.sum() > 0 and len(np.unique(y_c)) > 1:
        macro_f1 = f1_score(y_c, p_c, average="macro", zero_division=0)
        bal_acc = balanced_accuracy_score(y_c, p_c)
        tn_, fp_, fn_, tp_ = confusion_matrix(y_c, p_c, labels=[0, 1]).ravel()
        fpr = fp_ / max(fp_ + tn_, 1)
        fnr = fn_ / max(fn_ + tp_, 1)
        minority_label = 1 if y_c.mean() < 0.5 else 0
        min_mask = y_c == minority_label
        minority_recall = (p_c[min_mask] == minority_label).mean() if min_mask.sum() else float("nan")
    else:
        macro_f1 = bal_acc = fpr = fnr = minority_recall = float("nan")

    return dict(
        macro_f1=macro_f1, balanced_accuracy=bal_acc, fpr=fpr, fnr=fnr, minority_recall=minority_recall,
        mean_realized_reward=float(realized.mean()),
        commit_rate=float(np.isin(action, [A_COMMIT1, A_COMMIT0]).mean()),
        isolation_rate=float((action == A_ISOLATE).mean()),
        abstention_rate=float((action == A_ABSTAIN).mean()),
        escalation_rate=float((action == A_ESCALATE).mean()),
        n_flows=int(n),
    )


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="run only this dataset (subprocess-isolated mode)")
    ap.add_argument("--seed", type=int, default=None, help="run only this seed (subprocess-isolated mode)")
    args = ap.parse_args()
    datasets = [args.dataset] if args.dataset else DATASETS
    seeds = [args.seed] if args.seed is not None else SEEDS

    out_path = os.path.join(RESULTS_DIR, "aegis_analytic_baseline.csv")
    rows = []
    done = set()
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        rows = prev.to_dict("records")
        done = set(zip(prev["dataset"], prev["seed"]))
        print(f"Resuming: {len(done)} (dataset, seed) pairs already complete: {sorted(done)}", flush=True)

    rw = AegisRewardWeights()
    for dataset in datasets:
        for seed in seeds:
            if (dataset, seed) in done:
                print(f"=== analytic baseline {dataset} seed={seed} (skip, already done) ===", flush=True)
                continue
            print(f"=== analytic baseline {dataset} seed={seed} ===", flush=True)
            try:
                b = fit_aegis_backbone(dataset, seed=seed, row_cap=ROW_CAP, window_size=64,
                                        telemetry_budget=8, epochs_neural=4, epochs_graph=4, top_k=3)
                for split in ["val", "test"]:
                    y = b["parts"][split]["y"]
                    p = b[f"{split}_ensemble_p"]
                    action, exp_r = analytic_policy(p, rw)
                    m = score(y, p, action, rw)
                    m.update(dataset=dataset, seed=seed, split=split, model="AnalyticDecisionTheoretic",
                              mean_expected_reward=float(exp_r.mean()))
                    rows.append(m)
                print(f"  test macro_f1={rows[-1]['macro_f1']:.4f} isolation_rate={rows[-1]['isolation_rate']:.3f} "
                      f"commit_rate={rows[-1]['commit_rate']:.3f}", flush=True)
            except Exception as e:
                print("  ERROR:", type(e).__name__, e); continue
            finally:
                b = None
                gc.collect()
            pd.DataFrame(rows).to_csv(out_path, index=False)
    print("Saved results/csv/aegis_analytic_baseline.csv")


if __name__ == "__main__":
    main()
