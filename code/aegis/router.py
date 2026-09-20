"""
AegisGraph-X Layer 2: the graph mixture-of-experts router.

A small gating network that reads (a) each expert's calibrated-scale
probability, (b) a compact per-flow graph/context embedding (degree,
neighborhood attack fraction, protocol/port group -- reused from
graphs.graph_builder.neighborhood_stats/port_group, already computed by
build_envs/build_feature_rich_graphs), and (c) the raw flow features, and
outputs a sparse (top-k) softmax gate over the expert pool. The router's
own output probability is the gate-weighted sum of expert probabilities
(not a fresh classifier), so nothing about the pool's supervised training
signal is discarded -- routing only reweights it per-instance.

Trained on router_fit (the held-out 30% of TRAIN, see expert_pool.py),
using the experts' OUT-OF-SAMPLE predictions on router_fit (never
in-sample), so the router cannot learn to lean on an overfit expert.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoERouter(nn.Module):
    def __init__(self, n_experts, context_dim, hidden=32, top_k=3):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.gate_net = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_experts),
        )

    def forward(self, context):
        logits = self.gate_net(context)
        if self.top_k < self.n_experts:
            topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(1, topk_idx, topk_vals)
            logits = mask
        gates = F.softmax(logits, dim=-1)
        return gates


def build_context_matrix(expert_prob_matrix, graph_ctx, raw_feat_summary=None):
    """expert_prob_matrix: (N, n_experts) each expert's P(attack).
    graph_ctx: (N, k) graph/topology context features (degree, neighborhood
    attack fraction, port-group one-hot, etc.) -- already leakage-safe since
    it is derived only from structural/topological quantities, not labels.
    raw_feat_summary: optional (N, m) extra scalar summary of the raw flow
    (e.g. a few top-variance features) to let the router condition routing
    on flow content, not just expert opinions."""
    parts = [expert_prob_matrix, graph_ctx]
    if raw_feat_summary is not None:
        parts.append(raw_feat_summary)
    return np.concatenate(parts, axis=1).astype(np.float32)


def train_router(context_train, expert_prob_train, y_train, n_experts, epochs=40, lr=1e-3,
                  top_k=3, seed=42, entropy_reg=0.01):
    torch.manual_seed(seed)
    context_dim = context_train.shape[1]
    router = MoERouter(n_experts, context_dim, top_k=top_k)
    opt = torch.optim.Adam(router.parameters(), lr=lr)

    Xc = torch.as_tensor(context_train, dtype=torch.float32)
    Ep = torch.as_tensor(expert_prob_train, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
    y = torch.as_tensor(y_train, dtype=torch.float32)

    n_pos = max(float((y == 1).sum()), 1.0)
    n_neg = max(float((y == 0).sum()), 1.0)
    pos_weight = n_neg / n_pos

    router.train()
    for _ in range(epochs):
        opt.zero_grad()
        gates = router(Xc)
        mixed_p = (gates * Ep).sum(dim=-1).clamp(1e-6, 1 - 1e-6)
        w = torch.where(y == 1, torch.full_like(y, pos_weight), torch.ones_like(y))
        bce = F.binary_cross_entropy(mixed_p, y, weight=w)
        gate_entropy = -(gates * (gates.clamp_min(1e-8)).log()).sum(dim=-1).mean()
        loss = bce - entropy_reg * gate_entropy
        loss.backward()
        opt.step()
    return router


def route_predict(router, context, expert_prob_matrix):
    router.eval()
    with torch.no_grad():
        Xc = torch.as_tensor(context, dtype=torch.float32)
        Ep = torch.as_tensor(expert_prob_matrix, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
        gates = router(Xc)
        mixed_p = (gates * Ep).sum(dim=-1)
    return mixed_p.numpy(), gates.numpy()
