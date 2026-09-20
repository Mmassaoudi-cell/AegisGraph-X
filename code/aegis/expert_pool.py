"""
AegisGraph-X Layer 1, component (a): the expert pool.

Nine experts spanning tree/boosting, tabular-neural, temporal-neural, and
graph families, matching the task brief's required pool:
  XGBoost, LightGBM, CatBoost, RandomForest, ExtraTrees, HistGradientBoosting,
  FT-Transformer(-lite), TCN, protocol-role-graph GraphSAGE.

Leakage-safe stacking split: TRAIN is split ONCE into expert_fit (70%) and
router_fit (30%), both disjoint, both still inside the original TRAIN
partition (val/test are never touched here). Experts are fit on expert_fit
and scored out-of-sample on router_fit to produce the meta-features the
router (router.py) trains on. For the version deployed at val/test time,
every expert is *refit on the full TRAIN set* (expert_fit + router_fit) so
no training signal is wasted at deployment; the router itself is not
refit at that point (trained once on the smaller held-out split) -- a
disclosed, standard simplification of full k-fold stacking, traded for
tractable runtime (see manifests/technical_audit_report.md and the
Method section of the manuscript for the explicit disclosure).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               HistGradientBoostingClassifier)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baselines.deep_baselines import FTTransformer, TCN, SeqFlowDataset, FlowDataset, train_and_eval
from graphs.graph_builder import build_feature_rich_graphs
from graphs.graph_builder import port_group  # noqa: F401 (re-exported for callers)

TREE_EXPERT_NAMES = ["XGBoost", "LightGBM", "CatBoost", "RandomForest", "ExtraTrees", "HistGradientBoosting"]
NEURAL_EXPERT_NAMES = ["FT-Transformer", "TCN"]
GRAPH_EXPERT_NAMES = ["ProtocolRoleGraphSAGE"]
ALL_EXPERT_NAMES = TREE_EXPERT_NAMES + NEURAL_EXPERT_NAMES + GRAPH_EXPERT_NAMES


def build_tree_experts(seed=42):
    experts = {
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=18, class_weight="balanced",
                                                random_state=seed, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, max_depth=18, class_weight="balanced",
                                            random_state=seed, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=300, random_state=seed),
    }
    try:
        import xgboost as xgb
        experts["XGBoost"] = xgb.XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.08,
                                                eval_metric="logloss", random_state=seed, n_jobs=-1,
                                                tree_method="hist")
    except ImportError:
        pass
    try:
        import lightgbm as lgb
        experts["LightGBM"] = lgb.LGBMClassifier(n_estimators=300, max_depth=10, learning_rate=0.08,
                                                  class_weight="balanced", random_state=seed, verbosity=-1)
    except ImportError:
        pass
    try:
        import catboost as cb
        experts["CatBoost"] = cb.CatBoostClassifier(iterations=300, depth=8, learning_rate=0.08,
                                                     random_state=seed, verbose=False)
    except ImportError:
        pass
    return experts


def fit_tree_experts(X, y, seed=42):
    experts = build_tree_experts(seed=seed)
    for name, clf in experts.items():
        clf.fit(X, y)
    return experts


def tree_expert_proba(experts, X):
    out = {}
    for name, clf in experts.items():
        p = clf.predict_proba(X)
        out[name] = p[:, 1] if p.shape[1] == 2 else p
    return out


class NeuralExpertBundle:
    """Wraps the two neural experts (FT-Transformer-lite over single flows,
    TCN over a short sliding window of consecutive flows) with a common
    fit/predict_proba interface matching the tree experts."""
    def __init__(self, in_dim, seq_len=16, seed=42, epochs=6):
        torch.manual_seed(seed)
        self.in_dim = in_dim
        self.seq_len = seq_len
        self.epochs = epochs
        self.ft = FTTransformer(in_dim)
        self.tcn = TCN(in_dim)

    def fit(self, X, y):
        class_weight = torch.tensor([1.0, max((y == 0).sum() / max((y == 1).sum(), 1), 0.2)], dtype=torch.float32)
        flat_ds = FlowDataset(X, y)
        seq_ds = SeqFlowDataset(X, y, seq_len=self.seq_len)
        train_and_eval(self.ft, flat_ds, flat_ds, epochs=self.epochs, class_weight=class_weight)
        train_and_eval(self.tcn, seq_ds, seq_ds, epochs=self.epochs, class_weight=class_weight)
        return self

    def predict_proba(self, X):
        """Returns dict {"FT-Transformer": p, "TCN": p}. TCN needs a sequence
        window; for rows before position seq_len-1 within X we left-pad by
        repeating the first available window (documented approximation for
        the first `seq_len-1` rows of each dataset partition only)."""
        self.ft.eval(); self.tcn.eval()
        with torch.no_grad():
            Xt = torch.as_tensor(X, dtype=torch.float32)
            ft_p = F.softmax(self.ft(Xt), dim=-1)[:, 1].numpy()

            n = len(X)
            seqs = np.zeros((n, self.seq_len, X.shape[1]), dtype=np.float32)
            for i in range(n):
                lo = max(0, i - self.seq_len + 1)
                window = X[lo:i + 1]
                if len(window) < self.seq_len:
                    pad = np.repeat(window[:1], self.seq_len - len(window), axis=0) if len(window) > 0 else np.zeros((self.seq_len, X.shape[1]))
                    window = np.concatenate([pad, window], axis=0)
                seqs[i] = window
            tcn_p = F.softmax(self.tcn(torch.as_tensor(seqs, dtype=torch.float32)), dim=-1)[:, 1].numpy()
        return {"FT-Transformer": ft_p, "TCN": tcn_p}


class GraphExpertBundle:
    """The protocol-role/host-flow graph expert: GraphSAGE over windowed
    feature-rich graphs (same construction used for the graph baselines),
    read out per-edge (per-flow)."""
    def __init__(self, in_dim, seed=42, epochs=6, window_size=64):
        from baselines.graph_baselines import SAGEModel
        torch.manual_seed(seed)
        self.model = SAGEModel(in_dim, in_dim)
        self.window_size = window_size
        self.epochs = epochs

    def fit(self, df, spec, X, y):
        from baselines.graph_baselines import train_and_eval_graph_model
        graphs, _ = build_feature_rich_graphs(df, spec, X, y, window_size=self.window_size)
        y_all = np.concatenate([g.y.numpy() for g in graphs])
        class_weight = torch.tensor([1.0, max((y_all == 0).sum() / max((y_all == 1).sum(), 1), 0.2)],
                                     dtype=torch.float32)
        self.model, *_ = self._train(graphs, class_weight)
        return self

    def _train(self, graphs, class_weight):
        from torch_geometric.loader import DataLoader as PyGDataLoader
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss(weight=class_weight)
        loader = PyGDataLoader(graphs, batch_size=8, shuffle=True)
        self.model.train()
        for _ in range(self.epochs):
            for batch in loader:
                opt.zero_grad()
                logits = self.model(batch)
                loss = loss_fn(logits, batch.y)
                loss.backward()
                opt.step()
        return self.model,

    def predict_proba(self, df, spec, X, y):
        from torch_geometric.loader import DataLoader as PyGDataLoader
        graphs, _ = build_feature_rich_graphs(df, spec, X, y, window_size=self.window_size)
        self.model.eval()
        scores = []
        loader = PyGDataLoader(graphs, batch_size=8, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                logits = self.model(batch)
                p = F.softmax(logits, dim=-1)[:, 1]
                scores.append(p.numpy())
        return {"ProtocolRoleGraphSAGE": np.concatenate(scores) if scores else np.zeros(len(X))}


def stacking_split(X, y, router_frac=0.3, seed=42):
    """Contiguous (not shuffled) 70/30 split of TRAIN into expert_fit/router_fit.
    Contiguity is deliberate, not an oversight: the graph expert's windows
    (graph_builder.build_feature_rich_graphs) are built over runs of
    consecutive rows, so a shuffled/stratified split would silently break
    window structure for that one expert while leaving the other eight
    row-independent -- using the same contiguous split for every expert
    keeps the stacking procedure uniform and avoids that inconsistency,
    at the documented cost of expert_fit/router_fit not being class-balanced
    by construction (both still drawn from the same TRAIN distribution)."""
    n = len(y)
    cut = int(n * (1 - router_frac))
    expert_idx = np.arange(0, cut)
    router_idx = np.arange(cut, n)
    return expert_idx, router_idx
