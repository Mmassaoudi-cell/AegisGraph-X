"""
Part 8: RL / decision-policy baselines (11), all evaluated on the SAME
IDSPOMDPEnv as the 10 candidates, via metrics_utils.rollout_classification, so
Table VIII/X compare apples to apples with the candidates.

  1. DQN            -- online, replay buffer, single Q-network, epsilon-greedy.
  2. Double-DQN      -- same as DQN but with decoupled action-selection/evaluation
                        target (online net picks argmax action, target net values it).
  3. Dueling-DQN     -- separate value + advantage streams (architecture change only).
  4. PPO             -- core.PPOLiteTrainer, no extra module (the "vanilla" version
                        of what every candidate builds on).
  5. A2C             -- PPOLiteTrainer with 1 epoch/no clipping, i.e. plain
                        advantage actor-critic without PPO's trust-region clip.
  6. CQL (offline)   -- candidate_03's offline conservative Q-learning, imported
                        directly so the "offline RL baseline" and "Candidate 3"
                        are never silently two different implementations.
  7. Random acquisition       -- uniform-random action every step.
  8. Greedy uncertainty       -- rule-based: gather telemetry while |p-0.5| is
                                 small, else classify at the probe threshold.
  9. Greedy centrality        -- rule-based: expand the subgraph once if the
                                 flow's endpoints have high window degree, else
                                 classify directly.
 10. Static threshold policy  -- classify immediately at a fixed 0.5 threshold;
                                 never uses INFO/ESCALATE/ISOLATE/DEFER.
 11. Oracle upper bound       -- classifies using the ground-truth label at zero
                                 cost; a ceiling reference only, never presented
                                 as a deployable system.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from methods.core import TinyGraphEncoder, PPOLiteTrainer, build_obs_tensor
from methods.metrics_utils import rollout_classification
from methods.candidate_03_offline_safe import collect_offline_batch, train_cql
from env.pomdp_env import (N_ACTIONS, A_CLASSIFY_BENIGN, A_CLASSIFY_ATTACK, A_ABSTAIN,
                            A_REQUEST_TELEMETRY, A_INSPECT_NODE, A_EXPAND_SUBGRAPH, A_ISOLATE)


# ---------------------------------------------------------------- DQN family
class QNetPlain(nn.Module):
    def __init__(self, obs_dim, graph_dim, hidden=64, n_actions=N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim + graph_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)


class QNetDueling(nn.Module):
    def __init__(self, obs_dim, graph_dim, hidden=64, n_actions=N_ACTIONS):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim + graph_dim, hidden), nn.ReLU())
        self.value = nn.Linear(hidden, 1)
        self.adv = nn.Linear(hidden, n_actions)

    def forward(self, x):
        h = self.trunk(x)
        v, a = self.value(h), self.adv(h)
        return v + (a - a.mean(dim=-1, keepdim=True))


class DQNTrainer:
    def __init__(self, env, double=False, dueling=False, gamma=0.97, lr=1e-3, seed=0,
                 buffer_size=20000, eps_start=1.0, eps_end=0.05, eps_decay_steps=4000):
        self.env = env
        self.enc = TinyGraphEncoder()
        cls = QNetDueling if dueling else QNetPlain
        obs_dim = env._obs_dim()
        self.q = cls(obs_dim, self.enc.out.out_features)
        self.q_target = cls(obs_dim, self.enc.out.out_features)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(list(self.q.parameters()) + list(self.enc.parameters()), lr=lr)
        self.gamma, self.double = gamma, double
        self.buffer = deque(maxlen=buffer_size)
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        self.eps_start, self.eps_end, self.eps_decay_steps = eps_start, eps_end, eps_decay_steps
        self.step_count = 0

    def _embed_obs(self, obs, i):
        gemb = self.enc.window_summary(self.env.graphs[self.env.flow_index[i][0]]) if i is not None else torch.zeros(self.enc.out.out_features)
        return build_obs_tensor(obs, gemb.detach())

    def _epsilon(self):
        frac = min(self.step_count / self.eps_decay_steps, 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def _policy_step(self, obs, greedy=False):
        i = self.env._current_flow()
        x = self._embed_obs(obs, i)
        if not greedy and self.rng.random() < self._epsilon():
            action = int(self.rng.integers(0, N_ACTIONS))
        else:
            with torch.no_grad():
                action = int(torch.argmax(self.q(x.unsqueeze(0))).item())
        return action, None, None, x

    def train(self, n_episodes, batch_size=64, updates_per_episode=4, target_sync_every=200):
        for ep in range(n_episodes):
            obs = self.env.reset(episode_idx=int(self.rng.integers(0, len(self.env.episode_starts))))
            done = False
            while not done:
                i = self.env._current_flow()
                x = self._embed_obs(obs, i)
                action, _, _, _ = self._policy_step(obs, greedy=False)
                obs2, r, done, info = self.env.step(action)
                i2 = self.env._current_flow()
                x2 = self._embed_obs(obs2, i2)
                self.buffer.append((x.detach(), action, r, x2.detach(), float(done)))
                obs = obs2
                self.step_count += 1

                if len(self.buffer) >= batch_size:
                    for _ in range(updates_per_episode):
                        self._update(batch_size)
                    if self.step_count % target_sync_every == 0:
                        self.q_target.load_state_dict(self.q.state_dict())

    def _update(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        X = torch.stack([b[0] for b in batch])
        A = torch.as_tensor([b[1] for b in batch], dtype=torch.long)
        R = torch.as_tensor([b[2] for b in batch], dtype=torch.float32)
        X2 = torch.stack([b[3] for b in batch])
        D = torch.as_tensor([b[4] for b in batch], dtype=torch.float32)
        q_a = self.q(X).gather(1, A.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double:
                best_a = torch.argmax(self.q(X2), dim=1)
                q_next = self.q_target(X2).gather(1, best_a.unsqueeze(1)).squeeze(1)
            else:
                q_next = self.q_target(X2).max(dim=1).values
            target = R + self.gamma * (1 - D) * q_next
        loss = F.mse_loss(q_a, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


def run_dqn_family(env_train, env_val, variant, n_episodes=150, seed=0):
    double = variant in ("double", "dueling_double")
    dueling = variant in ("dueling", "dueling_double")
    trainer = DQNTrainer(env_train, double=double, dueling=dueling, seed=seed)
    trainer.train(n_episodes=n_episodes)
    trainer.env = env_val
    ids, decision = rollout_classification(trainer)
    return {**ids, **decision}


# --------------------------------------------------------------- PPO / A2C
def run_ppo(env_train, env_val, n_updates=25, episodes_per_update=8, seed=0):
    enc = TinyGraphEncoder()
    trainer = PPOLiteTrainer(env_train, enc, seed=seed)
    for _ in range(n_updates):
        trainer.update(n_episodes=episodes_per_update)
    trainer.env = env_val
    ids, decision = rollout_classification(trainer)
    return {**ids, **decision}


def run_a2c(env_train, env_val, n_updates=25, episodes_per_update=8, seed=0):
    enc = TinyGraphEncoder()
    trainer = PPOLiteTrainer(env_train, enc, clip=1e6, seed=seed)  # no trust-region clip -> plain A2C
    for _ in range(n_updates):
        trainer.update(n_episodes=episodes_per_update, epochs=1)
    trainer.env = env_val
    ids, decision = rollout_classification(trainer)
    return {**ids, **decision}


# ---------------------------------------------------------- offline CQL/IQL
def run_offline_cql(env_train, env_val, n_episodes_offline=150, epochs=10, seed=0):
    enc = TinyGraphEncoder()
    buf, saw_isolate = collect_offline_batch(env_train, enc, n_episodes=n_episodes_offline, seed=seed)
    obs_dim = env_train._obs_dim()
    q = train_cql(buf, obs_dim, enc.out.out_features, cql_alpha=1.0, epochs=epochs, seed=seed)

    class _W:
        def __init__(self, env, enc, q):
            self.env, self.enc, self.q = env, enc, q

        def _policy_step(self, obs, greedy=True):
            i = self.env._current_flow()
            gemb = self.enc.window_summary(self.env.graphs[self.env.flow_index[i][0]]).detach() if i is not None else torch.zeros(self.enc.out.out_features)
            x = build_obs_tensor(obs, gemb)
            action = int(torch.argmax(self.q(x.unsqueeze(0))).item())
            return action, None, None, x

    trainer = _W(env_val, enc, q)
    ids, decision = rollout_classification(trainer)
    return {**ids, **decision}


# --------------------------------------------------------- rule-based policies
class RulePolicyWrapper:
    """Common shape expected by rollout_classification: an object with .env and
    ._policy_step(obs, greedy=True) -> (action, None, None, x)."""
    def __init__(self, env, policy_fn):
        self.env = env
        self.policy_fn = policy_fn

    def _policy_step(self, obs, greedy=True):
        action = self.policy_fn(self.env)
        return action, None, None, obs


def random_acquisition_policy(env):
    return int(np.random.default_rng().integers(0, N_ACTIONS))


def greedy_uncertainty_policy(env, uncertainty_band=0.15):
    i = env._current_flow()
    if i is None:
        return A_CLASSIFY_BENIGN
    p = env._probe_proba_cache[i]
    if abs(p - 0.5) < uncertainty_band and env.telemetry_used < env.cfg.telemetry_budget:
        for act, acquired in [(A_REQUEST_TELEMETRY, env.acquired_groups[i]),
                               (A_INSPECT_NODE, {0} if env.acquired_node_stats[i] else set()),
                               (A_EXPAND_SUBGRAPH, {0} if env.acquired_subgraph[i] else set())]:
            if not acquired:
                return act
    return A_CLASSIFY_ATTACK if p >= 0.5 else A_CLASSIFY_BENIGN


def greedy_centrality_policy(env, degree_percentile=75):
    i = env._current_flow()
    if i is None:
        return A_CLASSIFY_BENIGN
    w, u, v = env.flow_index[i]
    g = env.graphs[w]
    deg = (float(g.x[u, 0]) if u < g.num_nodes else 0.0) + (float(g.x[v, 0]) if v < g.num_nodes else 0.0)
    all_deg = g.x[:, 0].numpy() if g.num_nodes > 0 else np.array([0.0])
    thresh = np.percentile(all_deg, degree_percentile) * 2
    if deg >= thresh and not env.acquired_subgraph[i] and env.telemetry_used < env.cfg.telemetry_budget:
        return A_EXPAND_SUBGRAPH
    p = env._probe_proba_cache[i]
    return A_CLASSIFY_ATTACK if p >= 0.5 else A_CLASSIFY_BENIGN


def static_threshold_policy(env, threshold=0.5):
    i = env._current_flow()
    if i is None:
        return A_CLASSIFY_BENIGN
    p = env._probe_proba_cache[i]
    return A_CLASSIFY_ATTACK if p >= threshold else A_CLASSIFY_BENIGN


def oracle_policy(env):
    """Upper-bound reference only: uses the ground-truth label directly. Never
    a deployable policy -- reported purely as a ceiling in Tables VIII/X."""
    i = env._current_flow()
    if i is None:
        return A_CLASSIFY_BENIGN
    return A_CLASSIFY_ATTACK if env.y[i] == 1 else A_CLASSIFY_BENIGN


def run_rule_policy(env_val, policy_fn):
    trainer = RulePolicyWrapper(env_val, policy_fn)
    ids, decision = rollout_classification(trainer)
    return {**ids, **decision}


def run_all(env_train, env_val, seed=0, n_updates=25, episodes_per_update=8, dqn_episodes=150):
    results = {}
    results["DQN"] = run_dqn_family(env_train, env_val, "plain", n_episodes=dqn_episodes, seed=seed)
    results["DoubleDQN"] = run_dqn_family(env_train, env_val, "double", n_episodes=dqn_episodes, seed=seed)
    results["DuelingDQN"] = run_dqn_family(env_train, env_val, "dueling", n_episodes=dqn_episodes, seed=seed)
    results["PPO"] = run_ppo(env_train, env_val, n_updates=n_updates, episodes_per_update=episodes_per_update, seed=seed)
    results["A2C"] = run_a2c(env_train, env_val, n_updates=n_updates, episodes_per_update=episodes_per_update, seed=seed)
    results["CQL_Offline"] = run_offline_cql(env_train, env_val, seed=seed)
    results["RandomAcquisition"] = run_rule_policy(env_val, random_acquisition_policy)
    results["GreedyUncertainty"] = run_rule_policy(env_val, greedy_uncertainty_policy)
    results["GreedyCentrality"] = run_rule_policy(env_val, greedy_centrality_policy)
    results["StaticThreshold"] = run_rule_policy(env_val, static_threshold_policy)
    results["OracleUpperBound"] = run_rule_policy(env_val, oracle_policy)
    return results
