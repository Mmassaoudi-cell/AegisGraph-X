"""
AegisGraph-X Layer 4: the budgeted graph-RL investigation policy.

Candidate B = backbone (Layers 1/2/5, aegis.pipeline) + this PPO-lite policy
trained on env.pomdp_env_aegis.AegisPOMDPEnv, whose COMMIT action always
reads its label off the pretrained, calibrated ensemble (see that module's
docstring for why this structurally bounds classification quality below by
the backbone's own quality).

Candidate C = Candidate B + a neuro-symbolic conformal safety shield
(SafetyShieldPolicy below): at evaluation time, if the current flow's
approximate conformal prediction set has size 2 (i.e. the calibrated
ensemble is not confident enough to exclude either class at the alpha=0.05
level) and the trained policy chose COMMIT, the shield overrides the action
to ESCALATE instead. This is a fixed, non-learned rule layered on top of
the trained policy -- it can only ever trade detection-cost for safety, so
if it does not improve minority recall/FNR or decision utility, that will
show up honestly as C not beating B (Part 16 wording policy), not concealed.

Reuses TinyGraphEncoder/ActorCritic/build_obs_tensor from methods/core.py
almost verbatim; the only structural change is n_actions=10 (Aegis action
space) instead of the legacy 11, and duck-typed compatibility with
AegisPOMDPEnv instead of IDSPOMDPEnv (both expose the same
reset/step/flow_index/graphs/episode_starts/episode_summary interface).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.core import TinyGraphEncoder, build_obs_tensor
from env.pomdp_env_aegis import A_COMMIT, A_ESCALATE, N_ACTIONS


class AegisActorCritic(nn.Module):
    def __init__(self, obs_dim, graph_dim=8, hidden=64, n_actions=N_ACTIONS):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + graph_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs):
        h = self.trunk(obs)
        return self.actor(h), self.critic(h).squeeze(-1)


class AegisPPOTrainer:
    def __init__(self, env, graph_encoder: TinyGraphEncoder, hidden=64, lr=3e-4,
                 gamma=0.97, lam=0.9, clip=0.2, seed=0):
        # seeded BEFORE any nn.Module is constructed, so weight initialization is
        # actually controlled by `seed` and not by whatever global torch RNG state
        # happened to precede this call (a real determinism bug in an earlier
        # version of this class: seeding after AegisActorCritic's construction let
        # its initial weights depend on prior unrelated torch calls elsewhere in
        # the same process, e.g. how many other trainers had already been built).
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.env = env
        self.enc = graph_encoder
        self.obs_dim = env._obs_dim()
        self.ac = AegisActorCritic(self.obs_dim, graph_dim=self.enc.out.out_features, hidden=hidden,
                                    n_actions=N_ACTIONS)
        params = list(self.ac.parameters()) + list(self.enc.parameters())
        self.opt = torch.optim.Adam(params, lr=lr)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.shield = False  # toggled True for Candidate C

    def _graph_embed_for_current(self):
        i = self.env._current_flow()
        if i is None:
            return torch.zeros(self.enc.out.out_features)
        w = self.env.flow_index[i][0]
        return self.enc.window_summary(self.env.graphs[w])

    def _maybe_shield(self, action, i):
        if self.shield and action == A_COMMIT and i is not None:
            if self.env._conformal_set_size[i] >= 2:
                return A_ESCALATE
        return action

    def _policy_step(self, obs, greedy=False):
        i = self.env._current_flow()
        gemb = self._graph_embed_for_current()
        x = build_obs_tensor(obs, gemb).unsqueeze(0)
        logits, value = self.ac(x)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        action = torch.argmax(probs, dim=-1) if greedy else dist.sample()
        logp = dist.log_prob(action)
        action_int = self._maybe_shield(int(action.item()), i)
        return action_int, logp, value.squeeze(0), x.squeeze(0)

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

    def evaluate(self, n_episodes=30, collect_predictions=False):
        summaries = []
        preds, trues = [], []
        for ep in range(n_episodes):
            obs = self.env.reset(episode_idx=ep % len(self.env.episode_starts))
            done = False
            while not done:
                action, _, _, _ = self._policy_step(obs, greedy=True)
                obs, r, done, info = self.env.step(action)
                if collect_predictions and "resolved_flow_row" in info:
                    ra = info["resolved_action"]
                    i = info["resolved_flow_row"]
                    if ra == A_COMMIT:
                        pred = self.env._commit_label(i, False)
                        preds.append(pred); trues.append(int(self.env.y[i]))
            summaries.append(self.env.episode_summary())
        if collect_predictions:
            return summaries, np.array(preds), np.array(trues)
        return summaries


def aggregate_summaries(summaries):
    keys = summaries[0].keys()
    return {k: float(np.mean([s[k] for s in summaries])) for k in keys}


def aegis_rollout_classification(trainer):
    """Greedy pass over every window exactly once, mirroring
    methods.metrics_utils.rollout_classification's convention: COMMIT/ISOLATE
    resolve to a discrete label (scored against y_true), ESCALATE/ABSTAIN are
    excluded from classification metrics but counted as operational decisions
    (so a policy cannot inflate its own macro-F1 by escalating/abstaining on
    every hard case -- that shows up instead as a high escalation/abstention
    rate, reported separately, Part 16 wording policy)."""
    from env.pomdp_env_aegis import A_ISOLATE
    from methods.metrics_utils import ids_metrics_from_scores
    env = trainer.env
    n_windows = len(env.episode_starts)
    y_true_all, y_score_all, y_pred_all = [], [], []
    total_reward = total_decisions = total_delay = 0.0
    telemetry_events = escalations = isolations = abstentions = budget_violations = calibration_violations = commits = 0
    for w in range(n_windows):
        obs = env.reset(episode_idx=w)
        done = False
        while not done:
            i = env._current_flow()
            action, _, _, _ = trainer._policy_step(obs, greedy=True)
            obs, r, done, info = env.step(action)
            resolved_action = info.get("resolved_action", action)
            resolved_i = info.get("resolved_flow_row", i)
            if resolved_action in (A_COMMIT, A_ISOLATE) and resolved_i is not None:
                true_label = int(env.y[resolved_i])
                score = float(env.ensemble_proba[resolved_i])
                pred = 1 if resolved_action == A_ISOLATE else env._commit_label(resolved_i, False)
                y_true_all.append(true_label)
                y_score_all.append(score)
                y_pred_all.append(pred)
        s = env.episode_summary()
        total_reward += s["cumulative_reward"]
        total_decisions += s["n_decisions"]
        total_delay += s["avg_detection_delay"] * s["n_decisions"]
        telemetry_events += s["telemetry_events"]
        escalations += s["escalations"]
        isolations += s["isolations"]
        abstentions += s["abstentions"]
        budget_violations += s["budget_violations"]
        calibration_violations += s["calibration_violations"]
        commits += s["commits"]

    ids = ids_metrics_from_scores(y_true_all, y_score_all, y_pred=y_pred_all) if y_true_all else {}
    decision = {
        "cumulative_reward": total_reward,
        "cumulative_reward_per_window": total_reward / max(n_windows, 1),
        "avg_detection_delay": total_delay / max(total_decisions, 1),
        "telemetry_cost": telemetry_events / max(total_decisions, 1),
        "escalation_rate": escalations / max(total_decisions, 1),
        "isolation_rate": isolations / max(total_decisions, 1),
        "abstention_rate": abstentions / max(total_decisions, 1),
        "budget_violation_rate": budget_violations / max(total_decisions, 1),
        "calibration_violation_rate": calibration_violations / max(total_decisions, 1),
        "commit_rate": commits / max(total_decisions, 1),
    }
    return ids, decision
