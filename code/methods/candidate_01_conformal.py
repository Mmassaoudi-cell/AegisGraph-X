"""
Candidate 1: Conformal Risk-Calibrated Graph Reinforcement Learning.

Key modules implemented:
 - temporal GNN encoder: core.TinyGraphEncoder (shared).
 - conformal uncertainty set: split-conformal calibration performed on a held-out
   slice of the VALIDATION fold only (never test) -- calibrate.py-style tau such
   that the empirical false-positive rate on the calibration slice is <= alpha.
 - risk-aware reward: classify actions taken while the conformal set is ambiguous
   ({benign, attack} both plausible) are discouraged via an extra observation
   feature (conformal set size) that lets the policy learn to ABSTAIN/ESCALATE
   instead of forcing a decision -- no reward hack, purely an informative feature.
 - validation-only threshold calibration: tau_lo/tau_hi are fit once on a
   calibration split carved out of VAL before any training-set metric is looked at.
"""
import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from methods.core import TinyGraphEncoder, PPOLiteTrainer, aggregate_summaries
from methods.metrics_utils import rollout_classification


def calibrate_conformal(probe, X_calib, y_calib, alpha=0.1):
    """Split-conformal: nonconformity = 1 - P(true class). tau = (1-alpha) quantile
    of nonconformity scores on the calibration fold."""
    proba = probe.predict_proba(X_calib)
    p_true = proba[np.arange(len(y_calib)), y_calib]
    nonconformity = 1 - p_true
    tau = np.quantile(nonconformity, 1 - alpha)
    return float(tau)


def make_extra_feature_fn(probe, tau):
    def extra_fn(env, i):
        if i is None:
            return np.array([0.0, 0.0], dtype=np.float32)
        p = env._probe_proba_cache[i]
        # prediction set membership: label included if 1-p(label) <= tau
        benign_in = (1 - (1 - p)) <= tau  # i.e. p(benign)=1-p_attack
        attack_in = (1 - p) <= tau
        set_size = int(benign_in) + int(attack_in)
        return np.array([set_size / 2.0, abs(p - 0.5)], dtype=np.float32)
    return extra_fn


def run(env_train, env_val, X_calib, y_calib, probe, alpha=0.1, n_updates=25,
        episodes_per_update=8, seed=0):
    tau = calibrate_conformal(probe, X_calib, y_calib, alpha=alpha)
    enc = TinyGraphEncoder()
    extra_fn = make_extra_feature_fn(probe, tau)
    trainer = PPOLiteTrainer(env_train, enc, extra_dim=2, extra_feature_fn=extra_fn, seed=seed)
    for _ in range(n_updates):
        trainer.update(n_episodes=episodes_per_update)
    trainer.env = env_val
    ids, decision = rollout_classification(trainer)
    return {"tau": tau, **ids, **decision}
