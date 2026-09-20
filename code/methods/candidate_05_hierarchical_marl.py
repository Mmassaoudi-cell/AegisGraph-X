"""
Candidate 5: Hierarchical Multi-Agent Graph RL.

Key modules implemented:
 - local graph policies: each flow's decision is conditioned on a "local group"
   id (hash of its host/role key modulo K), fed as an extra one-hot observation
   feature -- the shared-weight policy specializes its behaviour per group via
   this conditioning, playing the role of K local per-group agents.
 - regional/global coordinator: an extra scalar giving the *global* window-level
   attack-rate estimate (mean probe score over the whole window) is appended to
   every local agent's observation, letting local actors condition on
   cross-group context (a lightweight stand-in for a coordinator message).
 - centralized training with decentralized execution: training uses the shared
   PPO update (centralized, sees all groups' transitions); execution/action
   selection depends only on the local flow's own observation + its group id +
   the (already-computed, cheap) global scalar -- no cross-group communication
   is required at decision time.
 - graph-structured communication: the "global scalar" is derived from the same
   window graph all groups share, so information flows through the graph
   structure rather than an explicit message-passing channel.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from methods.core import TinyGraphEncoder, PPOLiteTrainer
from methods.metrics_utils import rollout_classification

K_GROUPS = 4


def make_extra_feature_fn(env_ref_holder):
    def extra_fn(env, i):
        if i is None:
            return np.zeros(K_GROUPS + 1, dtype=np.float32)
        group = hash(str(env.host_keys[i])) % K_GROUPS
        onehot = np.zeros(K_GROUPS, dtype=np.float32)
        onehot[group] = 1.0
        # cheap global scalar: mean probe score over the current window (coordinator signal)
        global_rate = float(np.mean(env._probe_proba_cache[env.cur_indices])) if env.cur_indices else 0.5
        return np.concatenate([onehot, [global_rate]]).astype(np.float32)
    return extra_fn


def run(env_train, env_val, n_updates=25, episodes_per_update=8, seed=0):
    enc = TinyGraphEncoder()
    extra_fn = make_extra_feature_fn(None)
    trainer = PPOLiteTrainer(env_train, enc, extra_dim=K_GROUPS + 1, extra_feature_fn=extra_fn, seed=seed)
    for _ in range(n_updates):
        trainer.update(n_episodes=episodes_per_update)
    trainer.env = env_val
    ids, decision = rollout_classification(trainer)
    return {**ids, **decision}
