"""
Part 4: POMDP environment for sequential intrusion investigation.

Episode = one graph window (a contiguous block of `window_size` flows, see
graphs/graph_builder.py). Within an episode the agent must dispose of every
flow in the window -- by classifying it, abstaining, or escalating -- while it
may first spend a bounded telemetry/step budget acquiring more information
(feature groups, node/subgraph neighborhood stats). Isolating a host has a
real, simulated consequence: subsequent flows from that host within the same
episode are (a) prevented from causing a false-negative penalty if they were
truly malicious (credited as "prevented"), and (b) penalized as disruption if
they were actually benign. This is what makes isolation a real sequential
decision rather than a relabeled classification.

State/observation, action set, and reward terms follow Part 4 / Part 13 of the
task spec. The environment is dataset-agnostic: it consumes an already
leakage-safe-split, train-only-fitted feature matrix (X), binary labels
(y_bin), optional multiclass labels (y_multi, used only for logging attack
stage / diagnostics, never as a reward input beyond what y_bin already
supplies), a host/role key per row (for isolation bookkeeping), and the
flow_index + graphs produced by graph_builder.build_graph_sequence.

The "uncertainty score" and "suspected attack stage" observation components
come from a lightweight probe classifier fit on TRAIN ONLY (never on the
episode's own labels at run time) -- see fit_probe(). This avoids leaking
ground truth into the observation.
"""
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np

# ----------------------------- action space -----------------------------
A_CLASSIFY_BENIGN = 0
A_CLASSIFY_ATTACK = 1
A_ABSTAIN = 2
A_REQUEST_TELEMETRY = 3
A_INSPECT_NODE = 4
A_EXPAND_SUBGRAPH = 5
A_ESCALATE = 6
A_ISOLATE = 7
A_RAISE_THRESHOLD = 8
A_LOWER_THRESHOLD = 9
A_DEFER = 10
N_ACTIONS = 11

TERMINAL_ACTIONS = {A_CLASSIFY_BENIGN, A_CLASSIFY_ATTACK, A_ABSTAIN, A_ESCALATE}
INFO_ACTIONS = {A_REQUEST_TELEMETRY, A_INSPECT_NODE, A_EXPAND_SUBGRAPH, A_RAISE_THRESHOLD, A_LOWER_THRESHOLD}


@dataclass
class RewardWeights:
    w_tp: float = 1.0
    w_delay: float = 0.3
    delay_tau: float = 3.0
    w_fn: float = 3.0
    w_fp: float = 1.5
    c_telemetry: float = 0.05
    c_inspect: float = 0.08
    c_subgraph: float = 0.15
    c_escalate: float = 0.4
    c_isolate_benign: float = 1.2   # disruption penalty: isolating an innocent host
    r_isolate_malicious: float = 0.6  # credit: isolating a genuinely malicious host prevents future harm
    c_budget_violation: float = 0.5
    c_step: float = 0.01             # tiny per-step cost discourages endless deferral/investigation


@dataclass
class EnvConfig:
    window_size: int = 64
    n_feature_groups: int = 4        # feature vector split into this many "purchasable" groups
    telemetry_budget: int = 8        # max info-actions per episode
    max_steps_per_flow: int = 4      # forces a terminal decision after this many info-actions on one flow
    reward: RewardWeights = field(default_factory=RewardWeights)
    seed: int = 42


def fit_probe(X_train, y_train, seed=42):
    """Lightweight train-only probe used purely to populate the agent's
    uncertainty/calibration observation features (never used to compute reward)."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=200, random_state=seed, class_weight="balanced")
    clf.fit(X_train, y_train)
    return clf


class IDSPOMDPEnv:
    """Minimal dependency-free (no gymnasium requirement) sequential IDS environment."""

    def __init__(self, X, y_bin, host_keys, flow_index, graphs, probe, y_multi=None,
                 config: Optional[EnvConfig] = None):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y_bin, dtype=np.int64)
        self.y_multi = y_multi
        self.host_keys = np.asarray(host_keys)
        self.flow_index = flow_index
        self.graphs = graphs
        self.probe = probe
        self.cfg = config or EnvConfig()
        self.n_flows, self.n_features = self.X.shape
        self.group_bounds = np.linspace(0, self.n_features, self.cfg.n_feature_groups + 1).astype(int)
        self.rng = np.random.default_rng(self.cfg.seed)
        self._probe_proba_cache = self._safe_probe_proba(self.X)
        self.threshold_presets = [0.3, 0.5, 0.7]
        self.episode_starts = list(range(0, self.n_flows, self.cfg.window_size))

    def _safe_probe_proba(self, X):
        try:
            return self.probe.predict_proba(X)[:, 1]
        except Exception:
            return np.full(len(X), 0.5, dtype=np.float32)

    # ------------------------------------------------------------------
    def reset(self, episode_idx: Optional[int] = None):
        if episode_idx is None:
            episode_idx = self.rng.integers(0, len(self.episode_starts))
        start = self.episode_starts[episode_idx]
        end = min(start + self.cfg.window_size, self.n_flows)
        self.cur_indices = list(range(start, end))
        # FIFO queue of flow row-indices still awaiting a terminal decision this episode.
        self.queue = list(self.cur_indices)
        self.deferred_once = set()  # each flow may be DEFERred at most once (avoid infinite loops)
        self.telemetry_used = 0
        self.threshold_idx = 1  # start at 0.5
        self.isolated_hosts = set()
        self.acquired_groups = {i: set() for i in self.cur_indices}
        self.acquired_node_stats = {i: False for i in self.cur_indices}
        self.acquired_subgraph = {i: False for i in self.cur_indices}
        self.n_steps_on_current = 0
        self.recent_alerts = []
        self.prev_action = -1
        self.budget_violations = 0
        self.trajectory_reward = 0.0
        self.n_decisions = 0
        self.n_correct = 0
        self.total_delay = 0
        self.escalations = 0
        self.isolations = 0
        self.abstentions = 0
        self.telemetry_events = 0
        self.done = False
        return self._observe()

    def _current_flow(self):
        return self.queue[0] if self.queue else None

    def _observe(self):
        i = self._current_flow()
        if i is None:
            return np.zeros(self._obs_dim(), dtype=np.float32)
        mask = np.zeros(self.n_features, dtype=np.float32)
        for g in self.acquired_groups[i]:
            lo, hi = self.group_bounds[g], self.group_bounds[g + 1]
            mask[lo:hi] = 1.0
        masked_x = self.X[i] * mask
        deg_u = deg_v = win_nodes = win_edges = 0.0
        if self.acquired_node_stats[i] or self.acquired_subgraph[i]:
            w, u, v = self.flow_index[i]
            g = self.graphs[w]
            deg_u = float(g.x[u, 0]) if u < g.num_nodes else 0.0
            deg_v = float(g.x[v, 0]) if v < g.num_nodes else 0.0
            win_nodes, win_edges = g.num_nodes, g.edge_index.shape[1]
        uncertainty = abs(self._probe_proba_cache[i] - 0.5) * 2  # 0 = max uncertain, 1 = confident
        alert_rate = np.mean(self.recent_alerts[-10:]) if self.recent_alerts else 0.0
        host_isolated = float(self.host_keys[i] in self.isolated_hosts)
        obs = np.concatenate([
            masked_x,
            [mask.mean()],
            [deg_u, deg_v, win_nodes, win_edges],
            [1.0 - uncertainty],           # calibration/uncertainty score (higher = more uncertain)
            [alert_rate],
            [self.telemetry_used / max(self.cfg.telemetry_budget, 1)],
            [self.threshold_presets[self.threshold_idx]],
            [host_isolated],
            [self.prev_action / N_ACTIONS],
            [self.n_steps_on_current / self.cfg.max_steps_per_flow],
        ]).astype(np.float32)
        return obs

    def _obs_dim(self):
        # Computed from the actual concatenation in _observe() rather than hand-counted,
        # so the two can never silently drift out of sync again.
        if not hasattr(self, "_cached_obs_dim"):
            self._cached_obs_dim = self.n_features + 12
        return self._cached_obs_dim

    # ------------------------------------------------------------------
    def step(self, action: int):
        assert not self.done
        i = self._current_flow()
        r = -self.cfg.reward.c_step
        info = {}
        rw = self.cfg.reward

        if i is None:
            self.done = True
            return self._observe(), 0.0, True, {}

        if action in INFO_ACTIONS:
            self.n_steps_on_current += 1
            if self.telemetry_used >= self.cfg.telemetry_budget:
                r -= rw.c_budget_violation
                self.budget_violations += 1
            else:
                self.telemetry_used += 1
                self.telemetry_events += 1
                if action == A_REQUEST_TELEMETRY:
                    remaining = [g for g in range(self.cfg.n_feature_groups) if g not in self.acquired_groups[i]]
                    if remaining:
                        self.acquired_groups[i].add(remaining[0])
                    r -= rw.c_telemetry
                elif action == A_INSPECT_NODE:
                    self.acquired_node_stats[i] = True
                    r -= rw.c_inspect
                elif action == A_EXPAND_SUBGRAPH:
                    self.acquired_subgraph[i] = True
                    r -= rw.c_subgraph
                elif action == A_RAISE_THRESHOLD:
                    self.threshold_idx = min(self.threshold_idx + 1, len(self.threshold_presets) - 1)
                elif action == A_LOWER_THRESHOLD:
                    self.threshold_idx = max(self.threshold_idx - 1, 0)
            if self.n_steps_on_current >= self.cfg.max_steps_per_flow:
                action = A_CLASSIFY_ATTACK if self._probe_proba_cache[i] >= self.threshold_presets[self.threshold_idx] else A_CLASSIFY_BENIGN

        if action in TERMINAL_ACTIONS or action == A_ISOLATE or action == A_DEFER:
            host = self.host_keys[i]
            host_isolated = host in self.isolated_hosts
            # Record which flow row and which action actually resolved this step (which
            # may differ from the action argument if max_steps_per_flow or an exhausted
            # DEFER forced an internal fallback classification above), so callers that
            # need ground-truth-vs-prediction pairs (e.g. metrics_utils.rollout_classification)
            # can see what genuinely happened rather than what was originally requested.
            info["resolved_flow_row"] = i
            info["resolved_action"] = action

            if action == A_DEFER and i not in self.deferred_once and len(self.queue) > 1:
                self.deferred_once.add(i)
                self.queue.pop(0)
                self.queue.append(i)
                self.n_steps_on_current = 0
            elif action == A_DEFER:
                # cannot defer again (already deferred once, or it's the last flow in
                # the episode) -- fall back to a forced classification so the episode
                # still makes progress instead of raising on an unhandled action.
                action = (A_CLASSIFY_ATTACK if self._probe_proba_cache[i] >= self.threshold_presets[self.threshold_idx]
                          else A_CLASSIFY_BENIGN)
                predicted = 1 if action == A_CLASSIFY_ATTACK else 0
                r += self._score_decision(i, predicted, host_isolated)
                self.queue.pop(0)
                self.n_steps_on_current = 0
            elif action == A_ISOLATE:
                self.isolations += 1
                self.isolated_hosts.add(host)
                true_label = self.y[i]
                r += rw.r_isolate_malicious if true_label == 1 else -rw.c_isolate_benign
                # Bookkeeping only (n_decisions/n_correct/recent_alerts/total_delay) -- the
                # return value is NOT added to r. Previously it was, which stacked a full TP
                # reward on top of the isolation credit for the same flow, making ISOLATE
                # strictly dominate an honest CLASSIFY_ATTACK for any true_label==1 flow once
                # the probe's implied confidence exceeded p=2/3 (derivable analytically from
                # the reward weights) -- independent of whether isolating that flow specifically
                # was informative. Fixed so isolation is scored only by its containment
                # consequence, not also as a duplicate classification credit.
                self._score_decision(i, predicted=1, host_isolated=host_isolated)
                self.queue.pop(0)
                self.n_steps_on_current = 0
            else:
                if host_isolated and action != A_ESCALATE:
                    # host already isolated this episode: remaining flows from it are contained
                    predicted = 1
                else:
                    predicted = {A_CLASSIFY_BENIGN: 0, A_CLASSIFY_ATTACK: 1,
                                 A_ABSTAIN: -1, A_ESCALATE: 2}[action]
                r += self._score_decision(i, predicted, host_isolated)
                if action == A_ESCALATE:
                    self.escalations += 1
                if action == A_ABSTAIN:
                    self.abstentions += 1
                self.queue.pop(0)
                self.n_steps_on_current = 0

            if not self.queue:
                self.done = True

        self.prev_action = action
        self.trajectory_reward += r
        obs = self._observe()
        done = self.done
        return obs, r, done, info

    def _score_decision(self, i, predicted, host_isolated):
        rw = self.cfg.reward
        true_label = self.y[i]
        delay_bonus = rw.w_delay * np.exp(-self.n_steps_on_current / rw.delay_tau)
        r = 0.0
        self.n_decisions += 1
        self.total_delay += self.n_steps_on_current
        if predicted == 2:  # escalate -> resolved correctly at a flat cost (analyst is accurate but expensive)
            r -= rw.c_escalate
            self.n_correct += 1
            self.recent_alerts.append(1 if true_label == 1 else 0)
            return r
        if predicted == -1:  # abstain
            self.recent_alerts.append(0)
            return r  # no tp/fn/fp scored; abstention itself already discouraged via c_step
        if predicted == 1 and true_label == 1:
            r += rw.w_tp + delay_bonus
            self.n_correct += 1
            self.recent_alerts.append(1)
        elif predicted == 0 and true_label == 0:
            r += rw.w_tp * 0.3  # smaller credit for correct benign (avoid degenerate always-attack policy)
            self.n_correct += 1
            self.recent_alerts.append(0)
        elif predicted == 1 and true_label == 0:
            r -= rw.w_fp
            self.recent_alerts.append(1)
        elif predicted == 0 and true_label == 1:
            r -= rw.w_fn
            self.recent_alerts.append(0)
        return r

    def episode_summary(self):
        n = max(self.n_decisions, 1)
        return {
            "cumulative_reward": self.trajectory_reward,
            "n_decisions": self.n_decisions,
            "accuracy_proxy": self.n_correct / n,
            "avg_detection_delay": self.total_delay / n,
            "telemetry_events": self.telemetry_events,
            "escalations": self.escalations,
            "isolations": self.isolations,
            "abstentions": self.abstentions,
            "budget_violations": self.budget_violations,
        }
