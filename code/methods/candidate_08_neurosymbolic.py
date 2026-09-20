"""
Candidate 8: Neuro-Symbolic ATT&CK-Constrained Graph RL. (Priority-3 candidate.)

Key modules implemented:
 - graph encoder: shared TinyGraphEncoder.
 - symbolic rule constraints: a small hand-coded rule set operating on a
   *predicted* (never ground-truth) coarse attack stage from
   common.label_maps.heuristic_attack_stage, applied to the multiclass label a
   secondary lightweight probe (fit on TRAIN ONLY) assigns the current flow:
     R1: ISOLATE is only permitted once the suspected stage >= 1 (recon-only
         suspicion must not trigger disruptive isolation).
     R2: ESCALATE is disallowed before at least one INFO action has been taken
         on the same flow (an analyst should not be paged with zero evidence).
     R3: ISOLATE/ESCALATE are disallowed while the conformal-style uncertainty
         proxy (|p-0.5|) is below 0.05 (too uncertain to justify a drastic
         action) -- forcing INFO/ABSTAIN instead.
 - constrained action masking: enforced via action_mask_fn passed to the shared
   PPO trainer (illegal actions get -inf logits, never sampled).
 - reward shaping: none beyond the environment's native reward -- the
   contribution here is purely the mask, so any performance difference is
   attributable to the symbolic constraint, not extra reward engineering.
 - explanation generator: `explain()` returns, for one flow, which rule (if any)
   restricted the action set, as a human-readable string trace.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from methods.core import TinyGraphEncoder, PPOLiteTrainer
from methods.metrics_utils import rollout_classification
from env.pomdp_env import N_ACTIONS, A_ISOLATE, A_ESCALATE
from common.label_maps import heuristic_attack_stage


def fit_multiclass_probe(X_train, y_multi_train, seed=0):
    from sklearn.linear_model import LogisticRegression
    if y_multi_train is None:
        return None
    clf = LogisticRegression(max_iter=200, random_state=seed, class_weight="balanced")
    clf.fit(X_train, y_multi_train)
    return clf


def make_mask_fn(multiclass_probe, class_names):
    def mask_fn(env, i):
        mask = np.zeros(N_ACTIONS, dtype=bool)
        if i is None:
            return mask
        p = env._probe_proba_cache[i]
        uncertain = abs(p - 0.5) < 0.05
        # fall back: derive suspected stage purely from binary probe confidence when no
        # multiclass probe/class_names are available for this dataset.
        if class_names is not None and multiclass_probe is not None:
            pred_label = multiclass_probe.predict(env.X[i:i + 1])[0]
            stage = heuristic_attack_stage(pred_label)
        else:
            stage = 1 if p >= 0.5 else -1

        no_info_yet = env.n_steps_on_current == 0
        if stage < 1:
            mask[A_ISOLATE] = True  # R1
        if no_info_yet:
            mask[A_ESCALATE] = True  # R2
        if uncertain:
            mask[A_ISOLATE] = True  # R3
            mask[A_ESCALATE] = True
        return mask
    return mask_fn


def run(env_train, env_val, X_train=None, y_multi_train=None, n_updates=25,
        episodes_per_update=8, seed=0):
    enc = TinyGraphEncoder()
    mc_probe = fit_multiclass_probe(X_train, y_multi_train, seed=seed) if y_multi_train is not None else None
    class_names = list(np.unique(y_multi_train)) if y_multi_train is not None else None
    mask_fn = make_mask_fn(mc_probe, class_names)
    trainer = PPOLiteTrainer(env_train, enc, action_mask_fn=mask_fn, seed=seed)
    for _ in range(n_updates):
        trainer.update(n_episodes=episodes_per_update)
    trainer.env = env_val
    ids, decision = rollout_classification(trainer)
    return {**ids, **decision}
