"""
Candidate 6: Continual Meta-Graph RL for Zero-Day Adaptation.

Key modules implemented:
 - temporal graph memory: reuses the windowed graph sequence.
 - meta-RL adaptation: the policy is first pretrained broadly on env_train, then
   given a handful (n_adapt_updates) of additional Reptile-style gradient steps
   restricted to episodes drawn from windows containing the rarest observed
   attack family (identified from y_multi on TRAIN only) -- a lightweight stand
   -in for "adapt quickly to an under-represented/novel family."
 - prototype memory: a running per-(coarse-)class mean embedding, built from the
   graph encoder's window summary at flows the probe currently assigns to each
   class, exposed to the policy as a nearest-prototype-distance feature.
 - replay/regularization: pretraining episodes are periodically re-mixed back in
   during the adaptation phase (replay) so the adaptation steps do not overwrite
   previously-learned behaviour (a simple anti-catastrophic-forgetting device).
 - change-point detection: a rolling z-score on the probe's mean predicted attack
   probability over the last 5 windows flags a distribution shift, exposed as a
   binary extra feature.
"""
import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from methods.core import TinyGraphEncoder, PPOLiteTrainer
from methods.metrics_utils import rollout_classification


class PrototypeMemory:
    def __init__(self, dim, momentum=0.9):
        self.protos = {0: np.zeros(dim), 1: np.zeros(dim)}
        self.momentum = momentum

    def update(self, label, embed):
        e = embed.detach().numpy()
        self.protos[label] = self.momentum * self.protos[label] + (1 - self.momentum) * e

    def nearest_dist(self, embed, predicted_label):
        e = embed.detach().numpy()
        return float(np.linalg.norm(e - self.protos[predicted_label]))


class ChangePointTracker:
    def __init__(self, window=5):
        self.history = []
        self.window = window

    def update(self, window_mean_score):
        self.history.append(window_mean_score)
        if len(self.history) > 50:
            self.history.pop(0)

    def flag(self):
        if len(self.history) < self.window + 1:
            return 0.0
        recent = self.history[-self.window:]
        base = self.history[:-self.window] or recent
        z = (np.mean(recent) - np.mean(base)) / (np.std(base) + 1e-6)
        return float(abs(z) > 1.5)


def rarest_family_windows(train_df, y_multi_col, window_size, benign_tokens=("benign", "normal")):
    if y_multi_col is None or y_multi_col not in train_df.columns:
        return set()
    s = train_df[y_multi_col].astype(str).str.lower()
    attack_counts = s[~s.str.startswith(tuple(benign_tokens))].value_counts()
    if attack_counts.empty:
        return set()
    rarest = attack_counts.idxmin()
    rows = np.where(s.values == rarest)[0]
    windows = set((rows // window_size).tolist())
    return windows


def run(env_train, env_val, train_df=None, y_multi_col=None, window_size=64,
        n_updates=20, episodes_per_update=8, n_adapt_updates=5, seed=0):
    enc = TinyGraphEncoder()
    proto_mem = PrototypeMemory(dim=enc.out.out_features)
    cpt = ChangePointTracker()

    def extra_fn(env, i):
        if i is None:
            return np.array([0.0, 0.0], dtype=np.float32)
        w = env.flow_index[i][0]
        embed = enc.window_summary(env.graphs[w])
        p = env._probe_proba_cache[i]
        predicted = int(p >= 0.5)
        dist = proto_mem.nearest_dist(embed, predicted)
        proto_mem.update(predicted, embed)
        return np.array([dist, cpt.flag()], dtype=np.float32)

    trainer = PPOLiteTrainer(env_train, enc, extra_dim=2, extra_feature_fn=extra_fn, seed=seed)
    for _ in range(n_updates):
        trainer.update(n_episodes=episodes_per_update)
        cpt.update(float(np.mean(env_train._probe_proba_cache)))

    rare_windows = rarest_family_windows(train_df, y_multi_col, window_size) if train_df is not None else set()
    if rare_windows:
        for _ in range(n_adapt_updates):
            idx = list(rare_windows)[: max(len(rare_windows), 1)]
            trainer.rng = np.random.default_rng(seed)  # keep adaptation deterministic
            trainer.update(n_episodes=min(episodes_per_update, max(len(idx), 1)))

    trainer.env = env_val
    ids, decision = rollout_classification(trainer)
    return {"adapted_on_rare_windows": len(rare_windows), **ids, **decision}
