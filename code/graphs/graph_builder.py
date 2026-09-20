"""
Part 10: Graph and state construction.

Builds windowed graphs from a preprocessed flow table. Two graph types are
supported depending on the dataset's available identifier columns:

 - host_flow_graph: nodes = hosts (src/dst IP), edges = flows between them,
   used when spec.src_ip_col / dst_ip_col are available (e.g. ToN-IoT, Edge-IIoTset).
 - protocol_role_graph: nodes = (protocol, role, port-group) tuples, edges =
   flows connecting a source-role node to a destination-role node, used when
   raw IPs are not available (e.g. UNSW-NB15, CICIoT2023, RT-IoT2022).

Both produce a sequence of per-window torch_geometric.data.Data graphs plus
a flow->(window_id, node_u, node_v) index used by the POMDP environment to
look up "neighborhood statistics" for a given flow's state representation.

A window is a fixed number of consecutive flows (row order in the leakage-safe
TRAIN/VAL/TEST partition, which for datasets with real timestamps is chronological
within each split, and for datasets without timestamps is the file's row order --
this limitation is stated explicitly in the paper's limitations section).
"""
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def port_group(port):
    try:
        p = int(port)
    except (ValueError, TypeError):
        return "na"
    if p == 0:
        return "na"
    if p < 1024:
        return "well_known"
    if p < 49152:
        return "registered"
    return "dynamic"


def build_host_flow_graph(df: pd.DataFrame, spec, window_size=256):
    """Nodes = unique host identifiers (src/dst IP). Edge features = flow numeric features."""
    assert spec.src_ip_col and spec.dst_ip_col
    src = df[spec.src_ip_col].astype(str).values
    dst = df[spec.dst_ip_col].astype(str).values
    n = len(df)
    graphs = []
    flow_index = np.zeros((n, 3), dtype=np.int64)  # window_id, u, v
    for w_start in range(0, n, window_size):
        w_end = min(w_start + window_size, n)
        w_src = src[w_start:w_end]
        w_dst = dst[w_start:w_end]
        hosts = pd.unique(np.concatenate([w_src, w_dst]))
        host_to_idx = {h: i for i, h in enumerate(hosts)}
        u = np.array([host_to_idx[h] for h in w_src])
        v = np.array([host_to_idx[h] for h in w_dst])
        edge_index = torch.tensor(np.stack([u, v]), dtype=torch.long)
        # node feature = degree (in+out) within the window, standardized later
        deg = np.zeros(len(hosts))
        for uu in u: deg[uu] += 1
        for vv in v: deg[vv] += 1
        x = torch.tensor(deg.reshape(-1, 1), dtype=torch.float)
        graphs.append(Data(x=x, edge_index=edge_index, num_nodes=len(hosts)))
        w_idx = len(graphs) - 1
        flow_index[w_start:w_end, 0] = w_idx
        flow_index[w_start:w_end, 1] = u
        flow_index[w_start:w_end, 2] = v
    return graphs, flow_index


def build_protocol_role_graph(df: pd.DataFrame, spec, window_size=256):
    """Nodes = (protocol, port-group) role tuples for src and dst side."""
    proto = df[spec.proto_col].astype(str).values if spec.proto_col in df.columns else np.full(len(df), "na")
    dport = df[spec.dst_port_col].astype(str).values if spec.dst_port_col and spec.dst_port_col in df.columns else np.full(len(df), "na")
    sport = df[spec.src_port_col].astype(str).values if spec.src_port_col and spec.src_port_col in df.columns else np.full(len(df), "na")
    src_role = np.array([f"{p}:{port_group(sp)}:src" for p, sp in zip(proto, sport)])
    dst_role = np.array([f"{p}:{port_group(dp)}:dst" for p, dp in zip(proto, dport)])
    n = len(df)
    graphs, flow_index = [], np.zeros((n, 3), dtype=np.int64)
    for w_start in range(0, n, window_size):
        w_end = min(w_start + window_size, n)
        w_src = src_role[w_start:w_end]
        w_dst = dst_role[w_start:w_end]
        roles = pd.unique(np.concatenate([w_src, w_dst]))
        role_to_idx = {r: i for i, r in enumerate(roles)}
        u = np.array([role_to_idx[r] for r in w_src])
        v = np.array([role_to_idx[r] for r in w_dst])
        edge_index = torch.tensor(np.stack([u, v]), dtype=torch.long)
        deg = np.zeros(len(roles))
        for uu in u: deg[uu] += 1
        for vv in v: deg[vv] += 1
        x = torch.tensor(deg.reshape(-1, 1), dtype=torch.float)
        graphs.append(Data(x=x, edge_index=edge_index, num_nodes=len(roles)))
        w_idx = len(graphs) - 1
        flow_index[w_start:w_end, 0] = w_idx
        flow_index[w_start:w_end, 1] = u
        flow_index[w_start:w_end, 2] = v
    return graphs, flow_index


def build_graph_sequence(df: pd.DataFrame, spec, window_size=256):
    """Dispatch to host-flow graph if raw host identifiers exist, else protocol-role graph."""
    if spec.src_ip_col and spec.dst_ip_col and spec.src_ip_col in df.columns and spec.dst_ip_col in df.columns:
        graphs, flow_index = build_host_flow_graph(df, spec, window_size)
        graph_type = "host_flow"
    else:
        graphs, flow_index = build_protocol_role_graph(df, spec, window_size)
        graph_type = "protocol_role"
    return graphs, flow_index, graph_type


def build_feature_rich_graphs(df: pd.DataFrame, spec, X, y, window_size=256):
    """For the graph BASELINE family (GCN/GraphSAGE/GAT/...), unlike the RL env's
    cheap degree-only node feature, node.x = mean of the (already train-fit-scaled)
    feature vectors of flows incident to that node within the window -- giving the
    GNN baselines a fair, information-rich input instead of a single scalar. Each
    window becomes one Data graph with edge_attr = the flow's own feature vector
    and edge_label = that flow's binary label (edge/flow-level classification
    target), read out via concat(node_u_embed, node_v_embed, edge_attr)."""
    if spec.src_ip_col and spec.dst_ip_col and spec.src_ip_col in df.columns and spec.dst_ip_col in df.columns:
        src = df[spec.src_ip_col].astype(str).values
        dst = df[spec.dst_ip_col].astype(str).values
    else:
        proto = df[spec.proto_col].astype(str).values if spec.proto_col in df.columns else np.full(len(df), "na")
        dport = df[spec.dst_port_col].astype(str).values if spec.dst_port_col and spec.dst_port_col in df.columns else np.full(len(df), "na")
        sport = df[spec.src_port_col].astype(str).values if spec.src_port_col and spec.src_port_col in df.columns else np.full(len(df), "na")
        src = np.array([f"{p}:{port_group(sp)}:src" for p, sp in zip(proto, sport)])
        dst = np.array([f"{p}:{port_group(dp)}:dst" for p, dp in zip(proto, dport)])

    n = len(df)
    graphs = []
    entities_per_graph = []
    for w_start in range(0, n, window_size):
        w_end = min(w_start + window_size, n)
        w_src, w_dst = src[w_start:w_end], dst[w_start:w_end]
        w_X, w_y = X[w_start:w_end], y[w_start:w_end]
        entities = pd.unique(np.concatenate([w_src, w_dst]))
        e2i = {e: i for i, e in enumerate(entities)}
        u = np.array([e2i[e] for e in w_src])
        v = np.array([e2i[e] for e in w_dst])

        node_sum = np.zeros((len(entities), w_X.shape[1]))
        node_cnt = np.zeros(len(entities))
        for k in range(len(w_X)):
            node_sum[u[k]] += w_X[k]; node_cnt[u[k]] += 1
            node_sum[v[k]] += w_X[k]; node_cnt[v[k]] += 1
        node_feat = node_sum / np.clip(node_cnt, 1, None)[:, None]

        edge_index = torch.tensor(np.stack([u, v]), dtype=torch.long)
        x = torch.tensor(node_feat, dtype=torch.float32)
        edge_attr = torch.tensor(w_X, dtype=torch.float32)
        edge_label = torch.tensor(w_y, dtype=torch.long)
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=edge_label,
                            num_nodes=len(entities)))
        entities_per_graph.append(list(entities))
    return graphs, entities_per_graph


def neighborhood_stats(graphs, flow_index, row_i):
    """Graph-neighborhood statistics for a given flow row: window node count, edge count,
    degree of its two endpoint nodes -- used as part of the POMDP observation."""
    w, u, v = flow_index[row_i]
    g = graphs[w]
    deg_u = float(g.x[u, 0]) if u < g.num_nodes else 0.0
    deg_v = float(g.x[v, 0]) if v < g.num_nodes else 0.0
    return {
        "window_n_nodes": g.num_nodes,
        "window_n_edges": g.edge_index.shape[1],
        "deg_u": deg_u,
        "deg_v": deg_v,
    }
