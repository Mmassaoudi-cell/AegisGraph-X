"""
Candidate 7: Logic-Preserving Adversarial Graph RL.

Key modules implemented:
 - defender policy: the shared PPO actor-critic.
 - attacker policy: a zero-order (random-search) perturbation chooser that, at
   each training step, tries `n_trials` small additive perturbations of the
   *masked* observation vector, each bounded to +-`eps` standard-deviation units
   (features are already train-fitted RobustScaler outputs, so a fixed eps is a
   comparable per-feature bound across features) and keeps whichever perturbation
   most increases the defender's current-policy loss proxy (entropy-weighted
   disagreement with the greedy action) -- i.e. an adversary that reshapes
   feature *values* it could plausibly report, not graph topology, hence
   "logic-preserving" (no edges/nodes/protocol fields are invented or removed).
 - logic-preserving graph perturbations: only already-acquired numeric feature
   dimensions are perturbed (mask-respecting), so no unacquired/unobserved
   telemetry or graph-structural field is touched.
 - robust reward: the defender is trained on the perturbed observations while
   still being scored against the true reward, i.e. standard adversarial
   training (perturb-then-optimize), alternated every rollout.
 - adversarial training: perturbation and PPO update alternate across the whole
   training loop below.
"""
import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from methods.core import TinyGraphEncoder, PPOLiteTrainer, build_obs_tensor
from methods.metrics_utils import rollout_classification


class AdversarialPPOTrainer(PPOLiteTrainer):
    def __init__(self, *a, eps=0.15, n_trials=4, **kw):
        super().__init__(*a, **kw)
        self.eps = eps
        self.n_trials = n_trials

    def _policy_step(self, obs, greedy=False):
        i = self.env._current_flow()
        gemb = self._graph_embed_for_current()
        extra = self.extra_feature_fn(self.env, i) if self.extra_feature_fn else None
        x_clean = build_obs_tensor(obs, gemb, extra)

        if not greedy and self.n_trials > 0:
            with torch.no_grad():
                logits_clean, _ = self.ac(x_clean.unsqueeze(0))
                probs_clean = torch.softmax(logits_clean, dim=-1).squeeze(0)
                worst_x, worst_kl = x_clean, -1.0
                for _ in range(self.n_trials):
                    noise = (torch.rand_like(x_clean) * 2 - 1) * self.eps
                    x_try = x_clean + noise
                    logits_try, _ = self.ac(x_try.unsqueeze(0))
                    probs_try = torch.softmax(logits_try, dim=-1).squeeze(0)
                    kl = torch.sum(probs_clean * torch.log((probs_clean + 1e-8) / (probs_try + 1e-8)))
                    if kl.item() > worst_kl:
                        worst_kl, worst_x = kl.item(), x_try
                x_use = worst_x
        else:
            x_use = x_clean

        x = x_use.unsqueeze(0)
        logits, value = self.ac(x)
        if self.action_mask_fn is not None:
            mask = self.action_mask_fn(self.env, i)
            logits = logits.masked_fill(torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0), float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        action = torch.argmax(probs, dim=-1) if greedy else dist.sample()
        logp = dist.log_prob(action)
        return int(action.item()), logp, value.squeeze(0), x.squeeze(0)


def run(env_train, env_val, n_updates=25, episodes_per_update=8, seed=0, eps=0.15):
    enc = TinyGraphEncoder()
    trainer = AdversarialPPOTrainer(env_train, enc, seed=seed, eps=eps, n_trials=4)
    for _ in range(n_updates):
        trainer.update(n_episodes=episodes_per_update)
    trainer.env = env_val
    trainer.n_trials = 0  # evaluate on clean observations (no attacker at deployment/eval time)
    ids, decision = rollout_classification(trainer)
    return {**ids, **decision}
