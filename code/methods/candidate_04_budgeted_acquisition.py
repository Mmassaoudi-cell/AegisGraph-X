"""
Candidate 4: Budgeted Active Subgraph Acquisition RL. (Priority-1 candidate per
the task's recommended-priority ordering, since it creates the most genuine
sequential decision problem among the ten.)

Key modules implemented:
 - graph encoder: core.TinyGraphEncoder (shared).
 - acquisition policy: the shared PPO policy choosing among REQUEST_TELEMETRY /
   INSPECT_NODE / EXPAND_SUBGRAPH / a terminal decision, under env's hard
   telemetry_budget.
 - information-gain reward: shaping.info_gain_bonus_fn rewards INFO actions in
   proportion to how much they move the agent's own masked-feature belief
   (probe scored on the *masked*, i.e. currently-acquired-only, feature vector)
   -- an actual value-of-information signal, not a fixed per-action bonus.
 - budget constraint: enforced natively by the POMDP env (telemetry_budget).
 - delayed classification: the agent may spend up to max_steps_per_flow info
   actions before a terminal decision is forced, i.e. classification is
   genuinely delayed pending acquisition.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from methods.core import TinyGraphEncoder, PPOLiteTrainer
from methods.metrics_utils import rollout_classification
from methods.shaping import ShapedEnv, info_gain_bonus_fn


def run(env_train, env_val, n_updates=25, episodes_per_update=8, seed=0, info_gain_weight=0.4):
    shaped_train = ShapedEnv(env_train, reward_bonus_fn=info_gain_bonus_fn(info_gain_weight))
    enc = TinyGraphEncoder()
    trainer = PPOLiteTrainer(shaped_train, enc, seed=seed)
    for _ in range(n_updates):
        trainer.update(n_episodes=episodes_per_update)
    trainer.env = env_val  # evaluate on the unshaped val env (shaping is a training-time device only)
    ids, decision = rollout_classification(trainer)
    return {**ids, **decision}
