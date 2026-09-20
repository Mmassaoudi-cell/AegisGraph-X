"""Analytic expected-cost / value-of-information policy for AegisPOMDPEnv.

Addresses Round-1 review MAJOR-11 / DA-6: the paper compares RL against
*no* RL but never against the closed-form Bayes-optimal policy operating on
the same calibrated posterior and the same cost table. This module supplies
that missing baseline.

Given the calibrated ensemble probability p = P(attack | x) and the reward
weights in AegisRewardWeights, the expected reward of every terminal action
is available in closed form, so the optimal action requires no learning.

Notation follows env/pomdp_env_aegis.py:_score_decision exactly.
  db = w_delay * exp(-d / delay_tau)          (delay bonus at step d)

  E[COMMIT, predict 1] = p(w_tp + db + w_minority) - (1-p) w_fp
  E[COMMIT, predict 0] = (1-p)(0.3 w_tp)       - p     w_fn
  E[ESCALATE]          = -c_escalate
  E[ABSTAIN]           = 0
  E[ISOLATE]           = p*r_isolate_malicious - (1-p) c_isolate_benign
                         + E[COMMIT, predict 1]

Equating the two COMMIT branches gives the Bayes-optimal operating point

  p* = (0.3 w_tp + w_fp) / (1.3 w_tp + db + w_minority + w_fp + w_fn)

which is the threshold a cost-minimising agent should use -- NOT 0.5.
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from env.pomdp_env_aegis import (
    AegisRewardWeights, A_COMMIT, A_ABSTAIN, A_ESCALATE, A_ISOLATE,
)


def delay_bonus(rw, d=0):
    return rw.w_delay * np.exp(-d / rw.delay_tau)


def optimal_threshold(rw, d=0):
    """Bayes-optimal commit threshold under the reward table."""
    db = delay_bonus(rw, d)
    num = 0.3 * rw.w_tp + rw.w_fp
    den = 1.3 * rw.w_tp + db + rw.w_minority_bonus + rw.w_fp + rw.w_fn
    return num / den


def action_values(p, rw, d=0):
    """Expected reward of each terminal action at belief p."""
    db = delay_bonus(rw, d)
    e_commit1 = p * (rw.w_tp + db + rw.w_minority_bonus) - (1 - p) * rw.w_fp
    e_commit0 = (1 - p) * (0.3 * rw.w_tp) - p * rw.w_fn
    e_commit = np.maximum(e_commit1, e_commit0)
    e_escalate = np.full_like(np.asarray(p, dtype=float), -rw.c_escalate)
    e_abstain = np.zeros_like(np.asarray(p, dtype=float))
    e_isolate = (p * rw.r_isolate_malicious
                 - (1 - p) * rw.c_isolate_benign
                 + e_commit1)
    return dict(commit=e_commit, commit1=e_commit1, commit0=e_commit0,
                escalate=e_escalate, abstain=e_abstain, isolate=e_isolate)


def value_of_information(rw, d=0):
    """Expected value of the three ACQUISITION actions. Exactly zero.

    Scope note: this applies to REQUEST_TELEMETRY / INSPECT_NODE /
    EXPAND_SUBGRAPH only -- NOT to the whole INFO_ACTIONS set.
    RAISE_THRESHOLD / LOWER_THRESHOLD are also members of INFO_ACTIONS but
    they mutate threshold_idx, which does change _commit_label's cutoff, so
    they have genuine (if coarse) decision value. An earlier version of this
    module conflated the two; the distinction matters because it is
    specifically acquisition, not threshold control, that is inert.

    For the three acquisition actions: AegisPOMDPEnv._commit_label reads
    self.ensemble_proba[i], assigned once at pomdp_env_aegis.py:99 and only
    ever read thereafter (113, 170, 273, 303). Acquisition mutates
    acquired_groups / acquired_node_stats / acquired_subgraph, which alter
    the *observation* (masked features, degree stats) but never the
    posterior the decision consumes. The COMMIT outcome is therefore
    bit-for-bit identical whether the policy took zero acquisition actions
    or all of them.

    Expected Bayes-risk reduction is thus identically zero, while each action
    costs c > 0 (0.05 / 0.08 / 0.15), consumes a budget slot, and -- because
    delay_bonus = w_delay * exp(-n_steps / delay_tau) decays with every step
    -- strictly reduces the eventual terminal reward. Acquisition has
    compounding negative expected value. A rational agent never acquires.

    The same holds in the legacy env (pomdp_env.py), where the fixed array is
    _probe_proba_cache.
    """
    return 0.0


def isolate_dominance_is_structural(rw):
    """Show the ISOLATE dominance cannot be tuned away.

    ISOLATE is scored as r_isolate_malicious / -c_isolate_benign PLUS a full
    _score_decision(predicted=1) (pomdp_env_aegis.py:261-262; identically at
    pomdp_env.py:254-255). So

        E[ISOLATE](p) - E[COMMIT=1](p) = p*r_isolate_malicious
                                         - (1-p)*c_isolate_benign

    The delay term cancels: both branches call _score_decision under the same
    n_steps_on_current. At p = 1 the difference is exactly
    r_isolate_malicious. Hence ANY r_isolate_malicious > 0 makes ISOLATE
    strictly dominate at sufficiently high belief, regardless of how large
    c_isolate_benign is. This is a structural consequence of the additive
    double-credit, not a magnitude that can be retuned.

    Crossover: p* = c_isolate_benign / (r_isolate_malicious + c_isolate_benign)
    """
    r, c = rw.r_isolate_malicious, rw.c_isolate_benign
    return dict(crossover=c / (r + c) if (r + c) else float("nan"),
                gap_at_p1=r,
                tunable=False if r > 0 else True)


def analytic_policy_action(p, rw, d=0):
    """argmax over terminal actions. Never acquires information (VoI = 0)."""
    v = action_values(p, rw, d)
    stacked = np.vstack([v["commit"], v["abstain"],
                         v["escalate"], v["isolate"]])
    idx = np.argmax(stacked, axis=0)
    amap = np.array([A_COMMIT, A_ABSTAIN, A_ESCALATE, A_ISOLATE])
    return amap[idx], stacked


def commit_label(p, rw, d=0):
    return (p >= optimal_threshold(rw, d)).astype(int)


def report(rw=None):
    rw = rw or AegisRewardWeights()
    print("=" * 72)
    print("ANALYTIC BAYES-OPTIMAL POLICY under the paper's default reward table")
    print("=" * 72)
    t = optimal_threshold(rw)
    print(f"\nBayes-optimal commit threshold p* = {t:.4f}")
    print(f"Environment default threshold     = 0.5  (threshold_idx=2)")
    print(f"Available presets                 = [0.3, 0.4, 0.5, 0.6, 0.7]")
    print(f"Nearest reachable preset to p*    = 0.3  "
          f"(requires 2x LOWER_THRESHOLD)")
    print(f"\n  -> the env's default operating point is miscalibrated against")
    print(f"     its own reward table by {0.5 - t:.3f} in probability, and the")
    print(f"     optimum is NOT REACHABLE from the preset grid.")
    print(f"  -> LOWER_THRESHOLD is in INFO_ACTIONS, so each correction also")
    print(f"     consumes a telemetry-budget slot and risks c_budget_violation.")

    print(f"\nValue of ACQUISITION = {value_of_information(rw):.4f} "
          f"(exactly zero, by construction)")
    print("  Applies to REQUEST_TELEMETRY / INSPECT_NODE / EXPAND_SUBGRAPH")
    print("  only. RAISE/LOWER_THRESHOLD are also in INFO_ACTIONS but DO")
    print("  move _commit_label's cutoff, so they retain decision value.")
    print("  ensemble_proba is fixed at construction; no acquisition action")
    print("  revises it, so no acquisition can change any decision. Costs:")
    print("  telemetry 0.05, inspect 0.08, subgraph 0.15 -- and delay_bonus")
    print("  decays each step, so acquisition is compounding negative EV.")
    print("  => optimal acquisition rate is 0.0 on every flow.")

    dom = isolate_dominance_is_structural(rw)
    print("\n" + "-" * 72)
    print("Is the ISOLATE dominance tunable?")
    print(f"  E[ISOLATE] - E[COMMIT=1] = p*{rw.r_isolate_malicious} "
          f"- (1-p)*{rw.c_isolate_benign}   (delay term cancels)")
    print(f"  crossover p* = c_iso/(r_iso + c_iso) = {dom['crossover']:.4f}")
    print(f"  gap at p=1   = r_isolate_malicious = {dom['gap_at_p1']:.4f}")
    print(f"  tunable away by changing weights? {dom['tunable']}")
    print("  ANY r_isolate_malicious > 0 dominates at high p. Structural.")

    print("\n" + "-" * 72)
    print("Action-value crossovers (d=0):")
    grid = np.linspace(0, 1, 100001)
    a, stacked = analytic_policy_action(grid, rw)
    names = {A_COMMIT: "COMMIT", A_ABSTAIN: "ABSTAIN",
             A_ESCALATE: "ESCALATE", A_ISOLATE: "ISOLATE"}
    prev, start = a[0], grid[0]
    for i in range(1, len(grid)):
        if a[i] != prev:
            print(f"  p in [{start:.4f}, {grid[i]:.4f})  -> {names[prev]}")
            prev, start = a[i], grid[i]
    print(f"  p in [{start:.4f}, 1.0000]  -> {names[prev]}")

    print("\n" + "-" * 72)
    print("Is ISOLATE ever optimal?")
    iso_opt = grid[a == A_ISOLATE]
    if len(iso_opt):
        print(f"  YES, for p >= {iso_opt.min():.4f} "
              f"({100*len(iso_opt)/len(grid):.1f}% of the belief range)")
    else:
        print("  NO -- ISOLATE is dominated at every belief. Any policy that")
        print("  isolates is behaving sub-optimally under this reward table.")
    print("=" * 72)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", default=True)
    ap.parse_args()
    report()
