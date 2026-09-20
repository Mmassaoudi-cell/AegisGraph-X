"""
Part 8: Graph IDS baselines (5 models): GCN, GraphSAGE, GAT, Protocol-role GAT,
Hypergraph NN. All operate on the feature-rich windowed graphs from
graph_builder.build_feature_rich_graphs (node.x = mean incident-flow features,
edge_attr = the flow's own feature vector, y = per-flow/edge binary label), and
all read out a per-edge (per-flow) prediction as concat(h_u, h_v, edge_attr) ->
MLP -> logits, trained/evaluated over many windows via PyG mini-batching.

"Protocol-role GAT" differs from plain GAT only in actually using edge_attr in
its attention computation (GATConv's edge_dim path) instead of ignoring it --
this is the specific, checkable architectural difference the name refers to,
not a relabeled duplicate.

"Hypergraph NN" builds genuine >2-node hyperedges per window (grouping nodes
that share a coarse attribute -- protocol for protocol-role graphs, /24 subnet
for host-flow graphs) rather than treating pairwise edges as degenerate
hyperedges, so it captures a different (multi-way) relational structure than
the pairwise-edge baselines.
"""
import time, tracemalloc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, HypergraphConv
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Data

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from methods.metrics_utils import ids_metrics_from_scores


class EdgeReadout(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden=32, n_classes=2):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * node_dim + edge_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, n_classes))

    def forward(self, h, edge_index, edge_attr):
        hu, hv = h[edge_index[0]], h[edge_index[1]]
        return self.mlp(torch.cat([hu, hv, edge_attr], dim=-1))


class GCNModel(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden=32, n_classes=2):
        super().__init__()
        self.c1 = GCNConv(in_dim, hidden)
        self.c2 = GCNConv(hidden, hidden)
        self.readout = EdgeReadout(hidden, edge_dim, hidden, n_classes)

    def forward(self, data):
        h = F.relu(self.c1(data.x, data.edge_index))
        h = F.relu(self.c2(h, data.edge_index))
        return self.readout(h, data.edge_index, data.edge_attr)


class SAGEModel(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden=32, n_classes=2):
        super().__init__()
        self.c1 = SAGEConv(in_dim, hidden)
        self.c2 = SAGEConv(hidden, hidden)
        self.readout = EdgeReadout(hidden, edge_dim, hidden, n_classes)

    def forward(self, data):
        h = F.relu(self.c1(data.x, data.edge_index))
        h = F.relu(self.c2(h, data.edge_index))
        return self.readout(h, data.edge_index, data.edge_attr)


class GATModel(nn.Module):
    """Plain GAT: attention ignores edge_attr (edge_dim=None)."""
    def __init__(self, in_dim, edge_dim, hidden=32, n_classes=2, heads=2):
        super().__init__()
        self.c1 = GATConv(in_dim, hidden, heads=heads, concat=False)
        self.c2 = GATConv(hidden, hidden, heads=heads, concat=False)
        self.readout = EdgeReadout(hidden, edge_dim, hidden, n_classes)

    def forward(self, data):
        h = F.elu(self.c1(data.x, data.edge_index))
        h = F.elu(self.c2(h, data.edge_index))
        return self.readout(h, data.edge_index, data.edge_attr)


class ProtocolRoleGAT(nn.Module):
    """GAT whose attention is explicitly conditioned on edge_attr (the flow's own
    protocol/port/flag features) via GATConv's edge_dim path -- the "protocol-role"
    distinction versus plain GAT."""
    def __init__(self, in_dim, edge_dim, hidden=32, n_classes=2, heads=2):
        super().__init__()
        self.c1 = GATConv(in_dim, hidden, heads=heads, concat=False, edge_dim=edge_dim)
        self.c2 = GATConv(hidden, hidden, heads=heads, concat=False, edge_dim=edge_dim)
        self.readout = EdgeReadout(hidden, edge_dim, hidden, n_classes)

    def forward(self, data):
        h = F.elu(self.c1(data.x, data.edge_index, edge_attr=data.edge_attr))
        h = F.elu(self.c2(h, data.edge_index, edge_attr=data.edge_attr))
        return self.readout(h, data.edge_index, data.edge_attr)


class HypergraphModel(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden=32, n_classes=2):
        super().__init__()
        self.c1 = HypergraphConv(in_dim, hidden)
        self.c2 = HypergraphConv(hidden, hidden)
        self.readout = EdgeReadout(hidden, edge_dim, hidden, n_classes)

    def forward(self, data):
        h = F.relu(self.c1(data.x, data.hyperedge_index))
        h = F.relu(self.c2(h, data.hyperedge_index))
        return self.readout(h, data.edge_index, data.edge_attr)


def add_hyperedges(graphs, entities_per_graph):
    """entities_per_graph[i] -> list of entity-name strings aligned to node order
    for graph i, used to bucket nodes into >2-node hyperedges by coarse attribute."""
    for g, entities in zip(graphs, entities_per_graph):
        buckets = {}
        for node_idx, ent in enumerate(entities):
            key = ent.split(":")[0] if ":" in ent else ".".join(ent.split(".")[:3])
            buckets.setdefault(key, []).append(node_idx)
        node_list, edge_list = [], []
        for he_id, (key, nodes) in enumerate(buckets.items()):
            for node_idx in nodes:
                node_list.append(node_idx)
                edge_list.append(he_id)
        g.hyperedge_index = torch.tensor([node_list, edge_list], dtype=torch.long)
    return graphs


def train_and_eval_graph_model(model_cls, train_graphs, eval_graphs, in_dim, edge_dim,
                                epochs=6, batch_size=8, lr=1e-3, seed=0, class_weight=None):
    torch.manual_seed(seed)
    model = model_cls(in_dim, edge_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    train_loader = PyGDataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    t0 = time.time()
    model.train()
    for ep in range(epochs):
        for batch in train_loader:
            opt.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, batch.y)
            loss.backward()
            opt.step()
    train_time = time.time() - t0

    model.eval()
    eval_loader = PyGDataLoader(eval_graphs, batch_size=batch_size, shuffle=False)
    scores, trues = [], []
    t1 = time.time()
    n_edges_total = 0
    with torch.no_grad():
        for batch in eval_loader:
            logits = model(batch)
            p = F.softmax(logits, dim=-1)[:, 1]
            scores.append(p.numpy())
            trues.append(batch.y.numpy())
            n_edges_total += batch.y.shape[0]
    infer_time = (time.time() - t1) / max(n_edges_total, 1)
    scores, trues = np.concatenate(scores), np.concatenate(trues)
    n_params = sum(p.numel() for p in model.parameters())
    return trues, scores, train_time, infer_time, n_params


def run_all(train_graphs, eval_graphs, in_dim, edge_dim, train_entities=None, eval_entities=None,
            seed=42, epochs=6, eval_split="val"):
    y_all_train = np.concatenate([g.y.numpy() for g in train_graphs])
    class_weight = torch.tensor([1.0, max((y_all_train == 0).sum() / max((y_all_train == 1).sum(), 1), 0.2)],
                                 dtype=torch.float32)

    specs = [("GCN", GCNModel), ("GraphSAGE", SAGEModel), ("GAT", GATModel),
             ("Protocol-role-GAT", ProtocolRoleGAT)]
    results = []
    for name, cls in specs:
        tracemalloc.start()
        trues, scores, train_time, infer_time, n_params = train_and_eval_graph_model(
            cls, train_graphs, eval_graphs, in_dim, edge_dim, epochs=epochs, seed=seed,
            class_weight=class_weight)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        metrics = ids_metrics_from_scores(trues, scores)
        metrics.update({"model": name, "family": "graph", "train_time_sec": train_time,
                         "inference_latency_sec_per_sample": infer_time, "peak_memory_MB": peak / 1e6,
                         "n_params": n_params, "seed": seed, "eval_split": eval_split})
        results.append(metrics)
        print(f"  {name}: macro_f1={metrics['macro_f1']:.4f} minority_recall={metrics['minority_recall']:.4f} "
              f"pr_auc={metrics['pr_auc']:.4f}", flush=True)

    if train_entities is not None and eval_entities is not None:
        train_hg = add_hyperedges([g for g in train_graphs], train_entities)
        eval_hg = add_hyperedges([g for g in eval_graphs], eval_entities)
        tracemalloc.start()
        trues, scores, train_time, infer_time, n_params = train_and_eval_graph_model(
            HypergraphModel, train_hg, eval_hg, in_dim, edge_dim, epochs=epochs, seed=seed,
            class_weight=class_weight)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        metrics = ids_metrics_from_scores(trues, scores)
        metrics.update({"model": "HypergraphNN", "family": "graph", "train_time_sec": train_time,
                         "inference_latency_sec_per_sample": infer_time, "peak_memory_MB": peak / 1e6,
                         "n_params": n_params, "seed": seed, "eval_split": eval_split})
        results.append(metrics)
        print(f"  HypergraphNN: macro_f1={metrics['macro_f1']:.4f} minority_recall={metrics['minority_recall']:.4f} "
              f"pr_auc={metrics['pr_auc']:.4f}", flush=True)
    return results
