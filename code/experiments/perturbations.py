"""
Part 12, Experiment 10: robustness / distribution-shift perturbations.

Generic, model-agnostic perturbation functions applied to an already
train-fit-scaled feature matrix X (and, where relevant, labels y) at
EVALUATION time only -- training is never touched, so any degradation
reported reflects genuine test-time robustness rather than retraining on
corrupted data. Each function takes (X, y, rng) and returns (X', y').
"""
import numpy as np


def feature_noise(X, y, rng, sigma=0.5):
    """Additive Gaussian noise scaled to each (already RobustScaler-fit) feature's
    own spread within this batch -- sigma is in units of that spread."""
    scale = np.std(X, axis=0, keepdims=True) + 1e-6
    noise = rng.normal(0, sigma, size=X.shape) * scale
    return X + noise, y


def missing_feature_injection(X, y, rng, frac=0.2):
    """Zero out a random `frac` of feature columns per row (simulating dropped
    telemetry fields), consistent with the RobustScaler's centered-at-median scale."""
    X2 = X.copy()
    mask = rng.random(X.shape) < frac
    X2[mask] = 0.0
    return X2, y


def label_noise(X, y, rng, frac=0.05):
    """Flip a random `frac` of evaluation labels -- used only to stress-test
    metric stability under noisy ground truth, never to alter training."""
    y2 = y.copy()
    idx = rng.choice(len(y2), size=int(len(y2) * frac), replace=False)
    y2[idx] = 1 - y2[idx]
    return X, y2


def class_imbalance_stress(X, y, rng, minority_keep_frac=0.3):
    """Subsample the minority class to stress-test recall under heavier imbalance
    than the natural test distribution."""
    minority_label = 1 if y.mean() < 0.5 else 0
    min_idx = np.where(y == minority_label)[0]
    maj_idx = np.where(y != minority_label)[0]
    keep_min = rng.choice(min_idx, size=max(int(len(min_idx) * minority_keep_frac), 1), replace=False)
    idx = np.concatenate([maj_idx, keep_min])
    rng.shuffle(idx)
    return X[idx], y[idx]


def protocol_port_masking(X, y, rng, col_indices, frac=1.0):
    """Zero out specific (protocol/port-related) columns for a random `frac` of rows,
    simulating loss of protocol/port telemetry specifically rather than random fields."""
    X2 = X.copy()
    n_affected = int(len(X2) * frac)
    idx = rng.choice(len(X2), size=n_affected, replace=False)
    for c in col_indices:
        if c < X2.shape[1]:
            X2[idx, c] = 0.0
    return X2, y


PERTURBATIONS = {
    "feature_noise_low": lambda X, y, rng: feature_noise(X, y, rng, sigma=0.2),
    "feature_noise_high": lambda X, y, rng: feature_noise(X, y, rng, sigma=0.8),
    "missing_features_20pct": lambda X, y, rng: missing_feature_injection(X, y, rng, frac=0.2),
    "missing_features_50pct": lambda X, y, rng: missing_feature_injection(X, y, rng, frac=0.5),
    "label_noise_5pct": lambda X, y, rng: label_noise(X, y, rng, frac=0.05),
    "class_imbalance_stress": lambda X, y, rng: class_imbalance_stress(X, y, rng, minority_keep_frac=0.3),
    "clean_baseline": lambda X, y, rng: (X, y),
}
