"""
AegisGraph-X end-to-end pipeline: builds the Layer-1/2/5 classification
backbone (Candidate A) for one dataset, leakage-safe throughout.

Stages:
  1. build_envs.build_all() -> leakage-safe TRAIN/VAL/TEST partitions,
     already-fit-on-TRAIN-only feature matrices, and the cheap windowed
     graphs used elsewhere in this project (RL env observations, context
     features).
  2. Split TRAIN into expert_fit/router_fit (aegis.expert_pool.stacking_split,
     contiguous to preserve graph-window structure).
  3. Fit all 9 experts on expert_fit; score them OUT-OF-SAMPLE on router_fit;
     train the MoE router on those out-of-sample scores + graph/context
     features (aegis.router).
  4. Refit all 9 experts on the FULL TRAIN split for deployment (the router
     itself is not refit -- disclosed simplification, see expert_pool.py).
  5. Score VAL and TEST through the refit experts + the once-trained router
     to get the calibrated-eligible ensemble probability.
  6. Fit calibration (temperature/isotonic/Platt/thresholds/conformal) on
     VAL only (aegis.calibrate). TEST is only ever scored, never fit on.

Returns a single dict bundle consumed by the RL layer (env/pomdp_env_aegis.py)
and by the metrics/reporting scripts.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from aegis.expert_pool import (fit_tree_experts, tree_expert_proba, NeuralExpertBundle,
                                GraphExpertBundle, stacking_split)
from aegis.router import build_context_matrix, train_router, route_predict
from aegis.calibrate import CalibrationBundle
from experiments.build_envs import build_all


def compute_graph_context(flow_index, graphs, indices):
    ctx = np.zeros((len(indices), 4), dtype=np.float32)
    for j, i in enumerate(indices):
        w, u, v = flow_index[i]
        g = graphs[w]
        deg_u = float(g.x[u, 0]) if u < g.num_nodes else 0.0
        deg_v = float(g.x[v, 0]) if v < g.num_nodes else 0.0
        ctx[j] = [deg_u, deg_v, float(g.num_nodes), float(g.edge_index.shape[1])]
    return ctx


def _fit_all_experts(X, y, df, spec, seed, epochs_neural, epochs_graph, window_size):
    tree_experts = fit_tree_experts(X, y, seed=seed)
    neural_bundle = NeuralExpertBundle(in_dim=X.shape[1], seed=seed, epochs=epochs_neural)
    neural_bundle.fit(X, y)
    graph_bundle = GraphExpertBundle(in_dim=X.shape[1], seed=seed, epochs=epochs_graph, window_size=window_size)
    graph_bundle.fit(df, spec, X, y)
    return tree_experts, neural_bundle, graph_bundle


def _score_all_experts(tree_experts, neural_bundle, graph_bundle, X, df, spec, y, expert_names):
    prob_dict = {}
    prob_dict.update(tree_expert_proba(tree_experts, X))
    prob_dict.update(neural_bundle.predict_proba(X))
    prob_dict.update(graph_bundle.predict_proba(df, spec, X, y))
    prob_matrix = np.stack([prob_dict[name] for name in expert_names], axis=1)
    return prob_matrix


def fit_aegis_backbone(dataset_name, seed=42, row_cap="default", window_size=64,
                        telemetry_budget=8, epochs_neural=4, epochs_graph=4, top_k=3,
                        router_frac=0.3, label_fraction=1.0):
    """label_fraction < 1.0 (Part 9 low-label study): stratified-subsamples
    the TRAIN split's labels ONLY (contiguous prefix per class, preserving
    each row's original within-class order so graph-window contiguity is
    disturbed as little as possible) before the expert_fit/router_fit
    stacking split -- val/test are never touched, mirroring
    experiments/run_low_label.py's convention for the baseline models."""
    t0 = time.time()
    bundle = build_all(dataset_name, window_size=window_size, telemetry_budget=telemetry_budget,
                        seed=seed, row_cap=row_cap)
    parts, spec, feat_names = bundle["parts"], bundle["spec"], bundle["feat_names"]
    train, val, test = parts["train"], parts["val"], parts["test"]
    X_train, y_train, df_train = train["X"], train["y"], train["df"]

    train_graphs, train_flow_index = train["graphs"], train["flow_index"]
    if label_fraction < 1.0:
        from sklearn.model_selection import train_test_split
        from graphs.graph_builder import build_graph_sequence
        idx_all = np.arange(len(y_train))
        keep_idx, _ = train_test_split(idx_all, train_size=label_fraction, random_state=seed, stratify=y_train)
        keep_idx = np.sort(keep_idx)
        X_train, y_train = X_train[keep_idx], y_train[keep_idx]
        df_train = df_train.iloc[keep_idx].reset_index(drop=True)
        # train["graphs"]/train["flow_index"] index into the FULL train sequence and would
        # silently misalign once rows are subsampled -- rebuild cheap windowed graphs over
        # the subsampled+reindexed sequence so router context features stay correct.
        train_graphs, train_flow_index, _ = build_graph_sequence(df_train, spec, window_size=window_size)

    expert_idx, router_idx = stacking_split(X_train, y_train, router_frac=router_frac, seed=seed)
    Xe, ye, df_e = X_train[expert_idx], y_train[expert_idx], df_train.iloc[expert_idx].reset_index(drop=True)
    Xr, yr, df_r = X_train[router_idx], y_train[router_idx], df_train.iloc[router_idx].reset_index(drop=True)

    t_expert0 = time.time()
    tree_e, neural_e, graph_e = _fit_all_experts(Xe, ye, df_e, spec, seed, epochs_neural, epochs_graph, window_size)
    expert_names = list(tree_e.keys()) + ["FT-Transformer", "TCN"] + ["ProtocolRoleGraphSAGE"]
    router_prob_matrix = _score_all_experts(tree_e, neural_e, graph_e, Xr, df_r, spec, yr, expert_names)
    expert_fit_time_s = time.time() - t_expert0

    graph_ctx_router = compute_graph_context(train_flow_index, train_graphs, router_idx)
    raw_summary_router = Xr[:, :min(5, Xr.shape[1])]
    context_router = build_context_matrix(router_prob_matrix, graph_ctx_router, raw_summary_router)

    t_router0 = time.time()
    router = train_router(context_router, router_prob_matrix, yr, n_experts=len(expert_names),
                           top_k=top_k, seed=seed)
    router_fit_time_s = time.time() - t_router0

    t_refit0 = time.time()
    tree_full, neural_full, graph_full = _fit_all_experts(X_train, y_train, df_train, spec, seed,
                                                            epochs_neural, epochs_graph, window_size)
    refit_time_s = time.time() - t_refit0

    def score_split(part):
        X = part["X"]
        prob_matrix = _score_all_experts(tree_full, neural_full, graph_full, X, part["df"], spec,
                                          part["y"], expert_names)
        ctx = compute_graph_context(part["flow_index"], part["graphs"], np.arange(len(X)))
        raw_summary = X[:, :min(5, X.shape[1])]
        context = build_context_matrix(prob_matrix, ctx, raw_summary)
        ensemble_p, gates = route_predict(router, context, prob_matrix)
        return prob_matrix, gates, ensemble_p, ctx, raw_summary

    # Train-split scoring is in-sample (experts were fit on this exact data) and is used
    # ONLY to build the RL-training environment's observations/COMMIT labels -- never
    # reported as a classification-quality number (those come from val_*/test_* below,
    # which are genuinely out-of-sample for every component including the router).
    train_prob_matrix, train_gates, train_ensemble_p, train_ctx, train_raw = score_split(train)
    val_prob_matrix, val_gates, val_ensemble_p, val_ctx, val_raw = score_split(val)
    test_prob_matrix, test_gates, test_ensemble_p, test_ctx, test_raw = score_split(test)

    calib = CalibrationBundle().fit(val_ensemble_p, val["y"])
    total_time_s = time.time() - t0

    return dict(
        dataset_name=dataset_name, spec=spec, feat_names=feat_names, parts=parts,
        expert_names=expert_names,
        tree_experts=tree_full, neural_bundle=neural_full, graph_bundle=graph_full, router=router,
        calib=calib,
        train_prob_matrix=train_prob_matrix, train_gates=train_gates, train_ensemble_p=train_ensemble_p,
        val_prob_matrix=val_prob_matrix, val_gates=val_gates, val_ensemble_p=val_ensemble_p,
        test_prob_matrix=test_prob_matrix, test_gates=test_gates, test_ensemble_p=test_ensemble_p,
        val_ctx=val_ctx, val_raw=val_raw, test_ctx=test_ctx, test_raw=test_raw,
        router_prob_matrix=router_prob_matrix, router_ctx=graph_ctx_router, router_raw=raw_summary_router,
        router_y=yr,
        timings=dict(expert_fit_time_s=expert_fit_time_s, router_fit_time_s=router_fit_time_s,
                     refit_time_s=refit_time_s, total_time_s=total_time_s),
        n_experts=len(expert_names), n_expert_fit_rows=len(expert_idx), n_router_fit_rows=len(router_idx),
    )


def ablated_router_ensemble(backbone, drop_experts=(), top_k=3, seed=42):
    """Retrains a router on the router_fit out-of-sample scores with one or
    more expert columns removed (e.g. drop_experts=("ProtocolRoleGraphSAGE",)
    for the "no graph branch" ablation, or =("TCN",) for "no temporal
    branch"), then re-scores VAL/TEST with the same columns dropped. Reuses
    the already-fitted experts and their already-computed probabilities --
    no re-fitting of any expert -- so ablation is cheap."""
    names = backbone["expert_names"]
    keep_idx = [i for i, n in enumerate(names) if n not in drop_experts]

    def sub(prob_matrix, ctx, raw):
        pm = prob_matrix[:, keep_idx]
        return build_context_matrix(pm, ctx, raw), pm

    ctx_r, pm_r = sub(backbone["router_prob_matrix"], backbone["router_ctx"], backbone["router_raw"])
    router = train_router(ctx_r, pm_r, backbone["router_y"], n_experts=len(keep_idx), top_k=min(top_k, len(keep_idx)),
                           seed=seed)

    out = {}
    for split, ens_key in [("val", "val"), ("test", "test")]:
        ctx_s, pm_s = sub(backbone[f"{split}_prob_matrix"], backbone[f"{split}_ctx"], backbone[f"{split}_raw"])
        p, _ = route_predict(router, ctx_s, pm_s)
        out[split] = p
    return out


def equal_weight_ensemble(backbone):
    """'No MoE router' ablation: unweighted mean of all expert probabilities
    (still the same 9-expert pool, just without learned per-instance routing)."""
    out = {}
    for split in ["val", "test"]:
        out[split] = backbone[f"{split}_prob_matrix"].mean(axis=1)
    return out


def best_single_expert(backbone, split="val"):
    """'No expert ensemble' ablation: the single best-performing tree expert
    alone (selected on VAL macro-F1 at threshold 0.5, never on test)."""
    from sklearn.metrics import f1_score
    names = backbone["expert_names"]
    val_pm = backbone["val_prob_matrix"]
    y_val = backbone["parts"]["val"]["y"]
    best_name, best_f1 = None, -1.0
    for i, name in enumerate(names):
        f1 = f1_score(y_val, (val_pm[:, i] >= 0.5).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_name = f1, name
    idx = names.index(best_name)
    return best_name, {"val": backbone["val_prob_matrix"][:, idx], "test": backbone["test_prob_matrix"][:, idx]}


def build_aegis_envs(backbone, window_size=64, telemetry_budget=8, reward_weights=None, seed=42):
    from env.pomdp_env_aegis import AegisPOMDPEnv, AegisEnvConfig, AegisRewardWeights
    parts = backbone["parts"]
    envs = {}
    for split_name, prob_key, gate_key, ens_key in [
        ("train", "train_prob_matrix", "train_gates", "train_ensemble_p"),
        ("val", "val_prob_matrix", "val_gates", "val_ensemble_p"),
        ("test", "test_prob_matrix", "test_gates", "test_ensemble_p"),
    ]:
        p = parts[split_name]
        rw = AegisRewardWeights(**reward_weights) if reward_weights else AegisRewardWeights()
        cfg = AegisEnvConfig(window_size=window_size, telemetry_budget=telemetry_budget, seed=seed, reward=rw)
        envs[split_name] = AegisPOMDPEnv(
            p["X"], p["y"], p["host_keys"], p["flow_index"], p["graphs"],
            backbone[ens_key], backbone[prob_key], backbone[gate_key], backbone["calib"],
            config=cfg,
        )
    return envs


def simple_classification_metrics(p, y_true, threshold):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                                  precision_recall_fscore_support, roc_auc_score,
                                  average_precision_score, confusion_matrix)
    y_pred = (p >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None,
                                                         labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    try:
        roc_auc = roc_auc_score(y_true, p)
        pr_auc = average_precision_score(y_true, p)
    except ValueError:
        roc_auc, pr_auc = float("nan"), float("nan")
    return dict(
        accuracy=accuracy_score(y_true, y_pred), balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        weighted_f1=f1_score(y_true, y_pred, average="weighted", zero_division=0),
        precision_attack=prec[1], recall_attack=rec[1], minority_recall=rec[1],
        fpr=fpr, fnr=fnr, roc_auc=roc_auc, pr_auc=pr_auc, threshold=float(threshold),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def backbone_metrics(ensemble_p, y_true, calib, threshold_key="temperature"):
    """Candidate A classification metrics using the calibrated operating point
    selected on VAL only (threshold_key picks which calibration variant's
    threshold to apply: uncalibrated/temperature/isotonic/platt)."""
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                                  precision_recall_fscore_support, roc_auc_score,
                                  average_precision_score, confusion_matrix)
    p_variants = calib.apply_all(ensemble_p)
    p = p_variants[threshold_key]
    t = {"uncalibrated": calib.threshold_uncalibrated, "temperature": calib.threshold_temperature,
         "isotonic": calib.threshold_isotonic, "platt": calib.threshold_platt}[threshold_key]
    y_pred = (p >= t).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None,
                                                         labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    try:
        roc_auc = roc_auc_score(y_true, p)
        pr_auc = average_precision_score(y_true, p)
    except ValueError:
        roc_auc, pr_auc = float("nan"), float("nan")

    return dict(
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        weighted_f1=f1_score(y_true, y_pred, average="weighted", zero_division=0),
        precision_benign=prec[0], recall_benign=rec[0], f1_benign=f1[0],
        precision_attack=prec[1], recall_attack=rec[1], f1_attack=f1[1],
        minority_recall=rec[1],
        fpr=fpr, fnr=fnr, roc_auc=roc_auc, pr_auc=pr_auc,
        threshold=t, threshold_key=threshold_key,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )
