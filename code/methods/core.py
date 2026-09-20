"""
Shared core for all 10 candidate graph/RL methods (Part 5/6).

Per the task's explicit instruction to "start with lightweight feasible
prototypes," every candidate reuses the same small graph encoder + actor-critic
trunk and the same on-policy PPO-lite trainer (or, for the two candidates whose
whole point is a different training regime -- offline RL and federated RL --
a matching alternative trainer). What distinguishes each candidate is a small,
separately-implemented module documented in its own file under code/methods/,
consistent with the "Key modules" list given for each candidate in the task
spec. This is an explicit, disclosed scope decision: these are screening-scale
prototypes, not independently-engineered SOTA systems.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env.pomdp_env import IDSPOMDPEnv, EnvConfig, N_ACTIONS, fit_probe


class TinyGraphEncoder(nn.Module):
    """1-layer GraphSAGE-style mean-aggregation encoder producing a single window-level
    summary embedding (mean-pooled node embeddings) -- cheap enough to run once per
    episode reset rather than per step."""
    def __init__(self, in_dim=1, hidden=16, out_dim=8):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, hidden)
        self.lin_neigh = nn.Linear(in_dim, hidden)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index):
        # mean neighbor aggregation (undirected treatment)
        n = x.shape[0]
        agg = torch.zeros_like(x)
        deg = torch.zeros(n, 1)
        src, dst = edge_index[0], edge_index[1]
        agg.index_add_(0, dst, x[src])
        deg.index_add_(0, dst, torch.ones(len(src), 1))
        agg = agg / deg.clamp(min=1)
        h = F.relu(self.lin_self(x) + self.lin_neigh(agg))
        h = self.out(h)
        return h  # [n_nodes, out_dim]

    def window_summary(self, graph):
        h = self.forward(graph.x, graph.edge_index)
        return h.mean(dim=0)  # [out_dim]


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, graph_dim=8, hidden=64, n_actions=N_ACTIONS, extra_dim=0):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + graph_dim + extra_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs):
        h = self.trunk(obs)
        return self.actor(h), self.critic(h).squeeze(-1)


def build_obs_tensor(obs, graph_embed, extra=None):
    parts = [torch.as_tensor(obs, dtype=torch.float32), graph_embed]
    if extra is not None:
        parts.append(torch.as_tensor(extra, dtype=torch.float32))
    return torch.cat(parts, dim=-1)


class PPOLiteTrainer:
    """Minimal single-environment PPO (clipped surrogate, GAE) -- deliberately small
    (a few hundred lines shorter than a full PPO implementation) since the screening
    phase only needs a credible, not maximally-tuned, on-policy learner."""
    def __init__(self, env: IDSPOMDPEnv, graph_encoder: TinyGraphEncoder, extra_dim=0,
                 hidden=64, lr=3e-4, gamma=0.97, lam=0.9, clip=0.2, action_mask_fn=None,
                 extra_feature_fn=None, seed=0):
        self.env = env
        self.enc = graph_encoder
        self.extra_dim = extra_dim
        self.obs_dim = env._obs_dim()
        self.ac = ActorCritic(self.obs_dim, graph_dim=self.enc.out.out_features, hidden=hidden,
                               extra_dim=extra_dim)
        params = list(self.ac.parameters()) + list(self.enc.parameters())
        self.opt = torch.optim.Adam(params, lr=lr)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.action_mask_fn = action_mask_fn
        self.extra_feature_fn = extra_feature_fn
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

    def _graph_embed_for_current(self):
        i = self.env._current_flow()
        if i is None:
            return torch.zeros(self.enc.out.out_features)
        w = self.env.flow_index[i][0]
        return self.enc.window_summary(self.env.graphs[w])

    def _policy_step(self, obs, greedy=False):
        i = self.env._current_flow()
        gemb = self._graph_embed_for_current()
        extra = self.extra_feature_fn(self.env, i) if self.extra_feature_fn else None
        x = build_obs_tensor(obs, gemb, extra).unsqueeze(0)
        logits, value = self.ac(x)
        if self.action_mask_fn is not None:
            mask = self.action_mask_fn(self.env, i)
            logits = logits.masked_fill(torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0), float("-inf"))
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        action = torch.argmax(probs, dim=-1) if greedy else dist.sample()
        logp = dist.log_prob(action)
        return int(action.item()), logp, value.squeeze(0), x.squeeze(0)

    def collect_rollout(self, n_episodes):
        obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []
        ep_summaries = []
        for _ in range(n_episodes):
            obs = self.env.reset(episode_idx=int(self.rng.integers(0, len(self.env.episode_starts))))
            done = False
            while not done:
                action, logp, value, x = self._policy_step(obs)
                obs2, r, done, info = self.env.step(action)
                obs_buf.append(x.detach())
                act_buf.append(action)
                logp_buf.append(logp.detach())
                val_buf.append(value.detach())
                rew_buf.append(r)
                done_buf.append(done)
                obs = obs2
            ep_summaries.append(self.env.episode_summary())
        return obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf, ep_summaries

    def _gae(self, rewards, values, dones):
        adv = np.zeros(len(rewards), dtype=np.float32)
        last_gae = 0.0
        values = values + [0.0]
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * values[t + 1] * nonterminal - values[t]
            last_gae = delta + self.gamma * self.lam * nonterminal * last_gae
            adv[t] = last_gae
        returns = adv + np.array(values[:-1])
        return adv, returns

    def update(self, n_episodes=8, epochs=4, minibatch=64):
        obs_buf, act_buf, logp_old, val_buf, rew_buf, done_buf, ep_summaries = self.collect_rollout(n_episodes)
        values = [v.item() for v in val_buf]
        adv, ret = self._gae(rew_buf, values, done_buf)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.stack(obs_buf)
        act_t = torch.as_tensor(act_buf, dtype=torch.long)
        logp_old_t = torch.stack(logp_old)
        adv_t = torch.as_tensor(adv, dtype=torch.float32)
        ret_t = torch.as_tensor(ret, dtype=torch.float32)

        n = len(obs_buf)
        for _ in range(epochs):
            idx = self.rng.permutation(n)
            for start in range(0, n, minibatch):
                mb = idx[start:start + minibatch]
                logits, value = self.ac(obs_t[mb])
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                logp = dist.log_prob(act_t[mb])
                ratio = torch.exp(logp - logp_old_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, ret_t[mb])
                entropy = dist.entropy().mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.opt.step()
        return ep_summaries

    def evaluate(self, n_episodes=30):
        summaries = []
        for ep in range(n_episodes):
            obs = self.env.reset(episode_idx=ep % len(self.env.episode_starts))
            done = False
            preds, trues = [], []
            while not done:
                action, _, _, _ = self._policy_step(obs, greedy=True)
                obs, r, done, info = self.env.step(action)
            summaries.append(self.env.episode_summary())
        return summaries


def aggregate_summaries(summaries):
    keys = summaries[0].keys()
    return {k: float(np.mean([s[k] for s in summaries])) for k in keys}
