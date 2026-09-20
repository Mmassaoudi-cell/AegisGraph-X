"""
Candidate 3: Offline Safe Graph RL from Historical SOC Logs.

Key modules implemented:
 - offline dataset of states/actions/rewards: generated once by a fixed logging
   ("historical analyst") policy -- an epsilon-greedy probe-threshold policy --
   interacting with env_train; no further online exploration occurs afterward.
 - conservative Q-learning (CQL-lite): a Q-network trained by TD regression on
   the logged batch plus a conservative penalty that pushes down Q-values for
   actions not seen in the logged data at a given state (approximated via the
   standard CQL logsumexp-minus-data-Q penalty), which is the mechanism that
   keeps the learned policy from overestimating unseen (state, action) pairs.
 - behavior constraint: the CQL penalty coefficient controls how tightly the
   learned policy must stay near the logging policy's action distribution.
 - safety penalty: additional discouragement of ISOLATE actions unless the
   logged behavior policy itself sometimes isolated (i.e. no *novel* disruptive
   actions are learned purely from extrapolation), reflecting the "safe" claim.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from methods.core import TinyGraphEncoder, build_obs_tensor
from methods.metrics_utils import rollout_classification
from env.pomdp_env import N_ACTIONS, A_ISOLATE


class QNet(nn.Module):
    def __init__(self, obs_dim, graph_dim, hidden=64, n_actions=N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + graph_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def behavior_policy_action(env, i, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(0, N_ACTIONS))
    p = env._probe_proba_cache[i]
    if abs(p - 0.5) < 0.1:
        return int(rng.choice([2, 3, 4]))  # abstain / request telemetry / inspect when unsure
    from env.pomdp_env import A_CLASSIFY_ATTACK, A_CLASSIFY_BENIGN
    return A_CLASSIFY_ATTACK if p >= 0.5 else A_CLASSIFY_BENIGN


def collect_offline_batch(env, enc, n_episodes=200, epsilon=0.3, seed=0):
    rng = np.random.default_rng(seed)
    buf = []
    saw_isolate = False
    for ep in range(n_episodes):
        obs = env.reset(episode_idx=ep % len(env.episode_starts))
        done = False
        while not done:
            i = env._current_flow()
            gemb = enc.window_summary(env.graphs[env.flow_index[i][0]]).detach() if i is not None else torch.zeros(enc.out.out_features)
            x = build_obs_tensor(obs, gemb).detach()
            a = behavior_policy_action(env, i, epsilon, rng)
            saw_isolate = saw_isolate or (a == A_ISOLATE)
            obs2, r, done, info = env.step(a)
            gemb2 = enc.window_summary(env.graphs[env.flow_index[env._current_flow()][0]]).detach() if env._current_flow() is not None else torch.zeros(enc.out.out_features)
            x2 = build_obs_tensor(obs2, gemb2).detach()
            buf.append((x, a, r, x2, float(done)))
            obs = obs2
    return buf, saw_isolate


def train_cql(buf, obs_dim, graph_dim, cql_alpha=1.0, epochs=15, batch_size=128, gamma=0.97, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    q = QNet(obs_dim, graph_dim)
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    X = torch.stack([b[0] for b in buf])
    A = torch.as_tensor([b[1] for b in buf], dtype=torch.long)
    R = torch.as_tensor([b[2] for b in buf], dtype=torch.float32)
    X2 = torch.stack([b[3] for b in buf])
    D = torch.as_tensor([b[4] for b in buf], dtype=torch.float32)
    n = len(buf)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            mb = idx[start:start + batch_size]
            q_all = q(X[mb])
            q_a = q_all.gather(1, A[mb].unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                q_next = q(X2[mb]).max(dim=1).values
                target = R[mb] + gamma * (1 - D[mb]) * q_next
            td_loss = F.mse_loss(q_a, target)
            # CQL penalty: push down logsumexp over all actions, push up data action's Q
            logsumexp = torch.logsumexp(q_all, dim=1)
            cql_penalty = (logsumexp - q_a).mean()
            loss = td_loss + cql_alpha * cql_penalty
            opt.zero_grad()
            loss.backward()
            opt.step()
    return q


def run(env_train, env_val, cql_alpha=1.0, n_episodes_offline=200, epochs=15, seed=0, behavior_epsilon=0.3):
    enc = TinyGraphEncoder()
    buf, saw_isolate = collect_offline_batch(env_train, enc, n_episodes=n_episodes_offline,
                                              epsilon=behavior_epsilon, seed=seed)
    obs_dim = env_train._obs_dim()
    q = train_cql(buf, obs_dim, enc.out.out_features, cql_alpha=cql_alpha, epochs=epochs, seed=seed)

    class _Wrapper:
        def __init__(self, env, enc, q, saw_isolate):
            self.env, self.enc, self.q, self.saw_isolate = env, enc, q, saw_isolate

        def _policy_step(self, obs, greedy=True):
            i = self.env._current_flow()
            gemb = self.enc.window_summary(self.env.graphs[self.env.flow_index[i][0]]).detach() if i is not None else torch.zeros(self.enc.out.out_features)
            x = build_obs_tensor(obs, gemb).detach()
            qvals = self.q(x.unsqueeze(0)).squeeze(0)
            if not self.saw_isolate:
                qvals[A_ISOLATE] = -1e9  # safety constraint: never learn a disruptive action never seen in logs
            action = int(torch.argmax(qvals).item())
            return action, None, None, x

    trainer = _Wrapper(env_val, enc, q, saw_isolate)
    ids, decision = rollout_classification(trainer)
    return {**ids, **decision}
