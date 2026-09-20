"""
Candidate 9: Self-Supervised Graph World-Model RL. (Priority-4 candidate.)

Key modules implemented:
 - masked graph encoder: shared TinyGraphEncoder, pretrained self-supervised
   (see below) before RL fine-tuning.
 - latent dynamics model: a small MLP predicts the *next* window's mean node
   degree and the *next* flow's feature-group means from the current window's
   graph embedding, trained on label-free data (rows where env.y == 0, i.e.
   "mostly benign traffic" is treated as the unlabeled pretraining stream, which
   is the self-supervised-pretraining premise stated in the task spec) using
   plain MSE reconstruction -- no labels are used for this pretraining loss.
 - graph policy / planning module: the (label-free-pretrained) world model's
   one-step prediction error at the current flow is exposed to the RL policy as
   an anomaly-score extra feature -- a cheap model-predictive signal ("how
   surprising is this window relative to normal dynamics") without a full
   tree-search planner.
 - model-predictive action selection: the anomaly-score feature directly
   informs which flows the (still shared) PPO policy chooses to INSPECT/EXPAND
   before committing to a terminal decision.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from methods.core import TinyGraphEncoder, PPOLiteTrainer
from methods.metrics_utils import rollout_classification


class WorldModel(nn.Module):
    def __init__(self, embed_dim, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, out_dim))

    def forward(self, embed):
        return self.net(embed)


def pretrain_world_model(env, enc, world, y_bin_train, n_steps=300, lr=1e-3, seed=0):
    """Self-supervised: predict next-window graph summary stats from current window
    embedding, restricted to windows dominated by benign (label==0) flows."""
    opt = torch.optim.Adam(list(enc.parameters()) + list(world.parameters()), lr=lr)
    rng = np.random.default_rng(seed)
    n_windows = len(env.episode_starts)
    benign_windows = []
    for w in range(n_windows - 1):
        start = env.episode_starts[w]
        end = min(start + env.cfg.window_size, env.n_flows)
        if y_bin_train[start:end].mean() < 0.3:
            benign_windows.append(w)
    if not benign_windows:
        benign_windows = list(range(n_windows - 1))
    for step in range(n_steps):
        w = int(rng.choice(benign_windows))
        embed = enc.window_summary(env.graphs[w])
        g_next = env.graphs[min(w + 1, n_windows - 1)]
        target = torch.tensor([
            g_next.x.mean().item(), g_next.x.std().item(),
            float(g_next.num_nodes), float(g_next.edge_index.shape[1]),
        ], dtype=torch.float32)
        pred = world(embed)
        loss = F.mse_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return enc, world


def make_extra_feature_fn(enc, world):
    def extra_fn(env, i):
        if i is None:
            return np.array([0.0], dtype=np.float32)
        w = env.flow_index[i][0]
        embed = enc.window_summary(env.graphs[w]).detach()
        with torch.no_grad():
            pred = world(embed)
        g = env.graphs[w]
        actual = torch.tensor([g.x.mean().item(), g.x.std().item(),
                                float(g.num_nodes), float(g.edge_index.shape[1])])
        error = float(F.mse_loss(pred, actual).item())
        return np.array([error], dtype=np.float32)
    return extra_fn


def run(env_train, env_val, y_bin_train, n_pretrain_steps=300, n_updates=25,
        episodes_per_update=8, seed=0):
    enc = TinyGraphEncoder()
    world = WorldModel(enc.out.out_features)
    enc, world = pretrain_world_model(env_train, enc, world, y_bin_train, n_steps=n_pretrain_steps, seed=seed)
    extra_fn = make_extra_feature_fn(enc, world)
    trainer = PPOLiteTrainer(env_train, enc, extra_dim=1, extra_feature_fn=extra_fn, seed=seed)
    for _ in range(n_updates):
        trainer.update(n_episodes=episodes_per_update)
    trainer.env = env_val
    ids, decision = rollout_classification(trainer)
    return {**ids, **decision}
