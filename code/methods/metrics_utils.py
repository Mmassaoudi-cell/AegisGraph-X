"""
Shared metric computation used by every candidate method AND every baseline, so
Tables IV/VI/VII/VIII/IX/X are computed identically regardless of model family.
"""
import numpy as np
from sklearn.metrics import (f1_score, balanced_accuracy_score, accuracy_score,
                              precision_score, recall_score, roc_auc_score,
                              average_precision_score, confusion_matrix)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env.pomdp_env import (A_CLASSIFY_BENIGN, A_CLASSIFY_ATTACK, A_ABSTAIN, A_ESCALATE,
                            A_ISOLATE, INFO_ACTIONS, A_DEFER)


def ids_metrics_from_scores(y_true, y_score, y_pred=None, threshold=0.5):
    """Standard supervised IDS metrics.

    y_score is a continuous attack-probability/ranking signal used ONLY for
    roc_auc/pr_auc. Discrete metrics (accuracy, macro_f1, recall, FPR/FNR, ...)
    are computed from y_pred when given -- the policy's/model's ACTUAL discrete
    decision -- and only fall back to thresholding y_score at `threshold` when
    y_pred is not supplied (the normal case for plain probability-output
    classifiers, where the decision genuinely *is* a threshold on the score).
    Passing the two separately matters for RL policies: a policy's discrete
    action (e.g. an oracle that reads the ground-truth label, or a policy that
    only classifies after several INFO actions) can differ from a shared
    environment probe's raw score, and conflating them silently mis-scores any
    policy whose decision isn't literally "threshold this score"."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred) if y_pred is not None else (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    minority_label = 1 if y_true.mean() < 0.5 else 0
    minority_recall = recall_score(y_true, y_pred, pos_label=minority_label, zero_division=0)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "attack_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "minority_recall": minority_recall,
        "precision_attack": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "fpr": fpr, "fnr": fnr,
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["pr_auc"] = average_precision_score(y_true, y_score)
    except ValueError:
        out["roc_auc"], out["pr_auc"] = float("nan"), float("nan")
    return out


def rollout_classification(trainer, n_passes_over_windows=None):
    """Run the (greedy) policy over every window exactly once, recording the true
    label and a soft attack-score for every flow that reaches a terminal decision,
    plus decision-level operational counters, for Table IV/VIII/IX/X style metrics."""
    env = trainer.env
    n_windows = len(env.episode_starts)
    y_true_all, y_score_all, y_pred_all = [], [], []
    total_reward, total_decisions, total_delay = 0.0, 0, 0
    telemetry_events = escalations = isolations = abstentions = budget_violations = 0
    for w in range(n_windows):
        obs = env.reset(episode_idx=w)
        done = False
        while not done:
            i = env._current_flow()
            action, _, value, _ = trainer._policy_step(obs, greedy=True)
            obs, r, done, info = env.step(action)
            # Use the *resolved* action/row the env actually scored (which can differ
            # from the requested action when an internal forced fallback fired -- see
            # pomdp_env.step) so classification metrics never silently drop those cases.
            resolved_action = info.get("resolved_action", action)
            resolved_i = info.get("resolved_flow_row", i)
            true_label = env.y[resolved_i] if resolved_i is not None else None
            probe_score = float(env._probe_proba_cache[resolved_i]) if resolved_i is not None else 0.5
            if resolved_action in (A_CLASSIFY_BENIGN, A_CLASSIFY_ATTACK, A_ISOLATE) and true_label is not None:
                # The discrete decision actually taken (1 = flagged attack/isolated,
                # 0 = flagged benign) -- this, not the shared probe score, is what
                # macro_f1/precision/recall/FPR/FNR must be computed from, otherwise
                # every policy that ever reaches a classify/isolate action collapses
                # onto "whatever the environment's shared probe says," which silently
                # erases genuine differences between policies (e.g. an oracle that
                # decides from ground truth would otherwise be scored as if it had
                # merely thresholded the probe, like every other policy).
                pred = 1 if resolved_action in (A_CLASSIFY_ATTACK, A_ISOLATE) else 0
                score = probe_score if resolved_action != A_ISOLATE else max(probe_score, 0.51)
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

    ids = ids_metrics_from_scores(y_true_all, y_score_all, y_pred=y_pred_all) if y_true_all else {}
    decision = {
        "cumulative_reward": total_reward,
        "cumulative_reward_per_window": total_reward / max(n_windows, 1),
        "avg_detection_delay": total_delay / max(total_decisions, 1),
        "telemetry_cost": telemetry_events / max(n_windows, 1),
        "escalation_rate": escalations / max(total_decisions, 1),
        "isolation_rate": isolations / max(total_decisions, 1),
        "abstention_rate": abstentions / max(total_decisions, 1),
        "budget_violation_rate": budget_violations / max(total_decisions, 1),
    }
    return ids, decision
