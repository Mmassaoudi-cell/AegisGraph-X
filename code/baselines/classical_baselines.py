"""
Part 8: Classical/tree IDS baselines (12 models), evaluated on the SAME
leakage-safe train/val/test tabular splits used everywhere else (build_envs.py),
using the SAME metrics_utils.ids_metrics_from_scores as the RL candidates for
Tables VI/VII/IX. These are flat-feature, single-shot classifiers -- they have
no sequential/decision-cost semantics, so decision-level columns (Table VIII)
are left as "not applicable" for this family rather than fabricated.
"""
import time, tracemalloc
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               GradientBoostingClassifier, HistGradientBoostingClassifier)
from sklearn.calibration import CalibratedClassifierCV

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from methods.metrics_utils import ids_metrics_from_scores


def _proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        d = model.decision_function(X)
        return 1 / (1 + np.exp(-d))
    return model.predict(X).astype(float)


def build_models(seed=42):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=300, class_weight="balanced", random_state=seed),
        "LinearSVM": CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=seed, max_iter=2000), cv=3),
        "GaussianNB": GaussianNB(),
        "kNN": KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
        "DecisionTree": DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=16, class_weight="balanced",
                                                random_state=seed, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=200, max_depth=16, class_weight="balanced",
                                            random_state=seed, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=seed),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=200, random_state=seed),
    }
    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                               eval_metric="logloss", random_state=seed, n_jobs=-1,
                                               tree_method="hist")
    except ImportError:
        pass
    try:
        import lightgbm as lgb
        models["LightGBM"] = lgb.LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                                                 class_weight="balanced", random_state=seed, verbosity=-1)
    except ImportError:
        pass
    try:
        import catboost as cb
        models["CatBoost"] = cb.CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1,
                                                     random_state=seed, verbose=False)
    except ImportError:
        pass
    return models


def run_all(X_train, y_train, X_val, y_val, X_test, y_test, seed=42, eval_split="val"):
    """eval_split: 'val' for screening-consistent scoring, 'test' for the final comparison."""
    X_eval, y_eval = (X_val, y_val) if eval_split == "val" else (X_test, y_test)
    results = []
    for name, model in build_models(seed=seed).items():
        t0 = time.time()
        tracemalloc.start()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        t1 = time.time()
        scores = _proba(model, X_eval)
        infer_time = (time.time() - t1) / max(len(X_eval), 1)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        n_params = getattr(model, "n_features_in_", X_train.shape[1])
        metrics = ids_metrics_from_scores(y_eval, scores)
        metrics.update({"model": name, "family": "classical_tree", "train_time_sec": train_time,
                         "inference_latency_sec_per_sample": infer_time, "peak_memory_MB": peak / 1e6,
                         "seed": seed, "eval_split": eval_split})
        results.append(metrics)
        print(f"  {name}: macro_f1={metrics['macro_f1']:.4f} minority_recall={metrics['minority_recall']:.4f} "
              f"pr_auc={metrics['pr_auc']:.4f}", flush=True)
    return results
