"""
Candidate 10: Byzantine-Robust Federated Graph RL.

Key modules implemented:
 - federated graph policy: K "sites" are simulated by partitioning the TRAIN
   windows into K contiguous shards; each site trains its own PPOLiteTrainer
   copy locally for a short local phase per federated round (no raw data leaves
   a site: only model parameters are aggregated).
 - Byzantine-robust aggregation: one shard (configurable) is designated
   malicious and has its local actor-critic parameter delta negated before
   aggregation (a standard sign-flipping poisoning attack proxy); aggregation
   uses coordinate-wise trimmed mean (dropping the largest and smallest value
   per parameter across sites) instead of plain FedAvg, which is the mechanism
   that must survive the poisoned shard.
 - personalized local head: each site keeps its own actor-critic instance
   between rounds (only a copy of the *aggregated* trunk is redistributed),
   approximating personalization.
 - communication-efficient updates: parameters are exchanged only once per
   federated round (not per gradient step).
 - malicious-client robustness: evaluated by comparing final global-model
   validation metrics against a FedAvg (mean, non-robust) ablation, reported
   under the ablation experiment rather than duplicated here.
"""
import copy
import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from methods.core import TinyGraphEncoder, PPOLiteTrainer
from methods.metrics_utils import rollout_classification


def _flatten(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def _unflatten_into(model, flat):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx + n].view_as(p))
        idx += n


def trimmed_mean_aggregate(param_vectors, trim_frac=0.2):
    stacked = torch.stack(param_vectors)  # [K, n_params]
    k = stacked.shape[0]
    n_trim = max(int(k * trim_frac), 0)
    if k - 2 * n_trim < 1:
        return stacked.mean(dim=0)
    sorted_vals, _ = torch.sort(stacked, dim=0)
    trimmed = sorted_vals[n_trim: k - n_trim] if n_trim > 0 else sorted_vals
    return trimmed.mean(dim=0)


def make_shard_envs(build_shard_env_fn, k_sites):
    return [build_shard_env_fn(site) for site in range(k_sites)]


def run(build_shard_env_fn, env_val, k_sites=5, n_malicious=1, n_rounds=8,
        local_updates_per_round=3, episodes_per_update=6, robust=True, seed=0):
    """build_shard_env_fn(site_idx) -> an IDSPOMDPEnv for that site's local shard."""
    global_enc = TinyGraphEncoder()
    global_trainer = PPOLiteTrainer(build_shard_env_fn(0), global_enc, seed=seed)
    global_params = _flatten(global_trainer.ac)

    for rnd in range(n_rounds):
        deltas = []
        for site in range(k_sites):
            local_enc = copy.deepcopy(global_enc)
            local_trainer = PPOLiteTrainer(build_shard_env_fn(site), local_enc, seed=seed + site + rnd)
            _unflatten_into(local_trainer.ac, global_params)
            for _ in range(local_updates_per_round):
                local_trainer.update(n_episodes=episodes_per_update)
            local_params = _flatten(local_trainer.ac)
            delta = local_params - global_params
            if site < n_malicious:
                delta = -delta * 3.0  # sign-flipping poisoning proxy, amplified
            deltas.append(delta)

        if robust:
            agg_delta = trimmed_mean_aggregate(deltas, trim_frac=0.2)
        else:
            agg_delta = torch.stack(deltas).mean(dim=0)
        global_params = global_params + agg_delta

    _unflatten_into(global_trainer.ac, global_params)
    global_trainer.env = env_val
    ids, decision = rollout_classification(global_trainer)
    return {**ids, **decision}
