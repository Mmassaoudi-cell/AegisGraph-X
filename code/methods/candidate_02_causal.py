"""
Candidate 2: Causal Counterfactual Temporal Graph RL.

Key modules implemented:
 - temporal causal graph: reuses the windowed host-flow/protocol-role graph
   (graphs/graph_builder.py) as the temporal structure investigated.
 - counterfactual edge masking: for the current flow's endpoint nodes, the
   degree-based node feature is recomputed with the flow's own edge removed;
   the resulting change in the window's local degree statistics is exposed as
   a "counterfactual contribution score" -- an approximation of "would the
   suspicious pattern remain if this edge were removed" using only structural
   (non-label) information.
 - causal contribution score: combined with the probe's uncertainty to form a
   2-D extra feature guiding the RL investigation policy toward flows whose
   removal would most change the local graph structure.
 - RL investigation policy: shared PPOLiteTrainer.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from methods.core import TinyGraphEncoder, PPOLiteTrainer, aggregate_summaries
from methods.metrics_utils import rollout_classification


def make_extra_feature_fn(probe):
    def extra_fn(env, i):
        if i is None:
            return np.array([0.0, 0.0], dtype=np.float32)
        w, u, v = env.flow_index[i]
        g = env.graphs[w]
        deg_u = float(g.x[u, 0]) if u < g.num_nodes else 0.0
        deg_v = float(g.x[v, 0]) if v < g.num_nodes else 0.0
        # counterfactual: remove this one edge's contribution to both endpoint degrees
        cf_deg_u = max(deg_u - 1, 0.0)
        cf_deg_v = max(deg_v - 1, 0.0)
        rel_change = ((deg_u - cf_deg_u) + (deg_v - cf_deg_v)) / max(deg_u + deg_v, 1.0)
        p = env._probe_proba_cache[i]
        causal_score = rel_change * abs(p - 0.5) * 2
        return np.array([rel_change, causal_score], dtype=np.float32)
    return extra_fn


def run(env_train, env_val, probe, n_updates=25, episodes_per_update=8, seed=0):
    enc = TinyGraphEncoder()
    extra_fn = make_extra_feature_fn(probe)
    trainer = PPOLiteTrainer(env_train, enc, extra_dim=2, extra_feature_fn=extra_fn, seed=seed)
    for _ in range(n_updates):
        trainer.update(n_episodes=episodes_per_update)
    trainer.env = env_val
    ids, decision = rollout_classification(trainer)
    return {**ids, **decision}
