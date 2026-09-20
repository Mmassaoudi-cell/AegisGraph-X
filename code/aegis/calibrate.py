"""
AegisGraph-X Layer 5: calibration and low-FPR control.

All fitting here happens on a VALIDATION split only, never test. Provides:
  - temperature scaling (1-parameter logistic recalibration)
  - isotonic regression
  - Platt scaling (logistic regression on the raw score)
  - class-wise / operating-point threshold selection (max F1, or FPR-budget)
  - split-conformal benign-FPR control (returns a p-value/threshold such
    that non-conformity on held-out benign flows controls FPR at level alpha)
  - ECE / Brier / NLL diagnostics
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fit_temperature(p_val, y_val, n_iter=200, lr=0.05):
    z = _logit(p_val)
    T = 1.0
    y = y_val.astype(np.float64)
    for _ in range(n_iter):
        p = _sigmoid(z / T)
        grad_T = np.mean((p - y) * (-z / (T ** 2)))
        T -= lr * grad_T
        T = float(np.clip(T, 0.05, 20.0))
    return T


def apply_temperature(p, T):
    return _sigmoid(_logit(p) / T)


def fit_isotonic(p_val, y_val):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val, y_val)
    return iso


def fit_platt(p_val, y_val):
    lr = LogisticRegression()
    lr.fit(_logit(p_val).reshape(-1, 1), y_val)
    return lr


def apply_platt(p, lr):
    return lr.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


def select_threshold_max_f1(p_val, y_val, grid=None):
    if grid is None:
        grid = np.linspace(0.01, 0.99, 197)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        pred = (p_val >= t).astype(int)
        tp = np.sum((pred == 1) & (y_val == 1))
        fp = np.sum((pred == 1) & (y_val == 0))
        fn = np.sum((pred == 0) & (y_val == 1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def select_threshold_for_fpr_budget(p_val, y_val, target_fpr=0.01):
    benign_scores = np.sort(p_val[y_val == 0])
    if len(benign_scores) == 0:
        return 0.5
    idx = int(np.ceil((1 - target_fpr) * len(benign_scores))) - 1
    idx = np.clip(idx, 0, len(benign_scores) - 1)
    return float(benign_scores[idx])


def conformal_benign_threshold(p_val_benign, alpha=0.05):
    """Split-conformal control of benign FPR at level alpha: returns a
    threshold t such that, asymptotically, P(score >= t | benign) <= alpha
    on exchangeable held-out benign data."""
    scores = np.sort(p_val_benign)
    n = len(scores)
    if n == 0:
        return 0.5
    k = int(np.ceil((1 - alpha) * (n + 1))) - 1
    k = np.clip(k, 0, n - 1)
    return float(scores[k])


def expected_calibration_error(p, y, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def brier_score(p, y):
    return float(np.mean((p - y) ** 2))


def nll_score(p, y, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_diagram_data(p, y, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    centers, accs, confs, counts = [], [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        centers.append((lo + hi) / 2)
        if mask.sum() == 0:
            accs.append(np.nan); confs.append(np.nan); counts.append(0)
        else:
            accs.append(float(y[mask].mean()))
            confs.append(float(p[mask].mean()))
            counts.append(int(mask.sum()))
    return {"bin_center": centers, "accuracy": accs, "confidence": confs, "count": counts}


class CalibrationBundle:
    """Fits every calibration method on validation data and can produce all
    variants on demand for the calibration-comparison experiment (Part 10)."""
    def __init__(self):
        self.temperature = None
        self.isotonic = None
        self.platt = None
        self.threshold_uncalibrated = 0.5
        self.threshold_temperature = 0.5
        self.threshold_isotonic = 0.5
        self.threshold_platt = 0.5
        self.conformal_threshold_alpha05 = 0.5

    def fit(self, p_val, y_val):
        self.temperature = fit_temperature(p_val, y_val)
        self.isotonic = fit_isotonic(p_val, y_val)
        self.platt = fit_platt(p_val, y_val)

        self.threshold_uncalibrated = select_threshold_max_f1(p_val, y_val)
        p_temp = apply_temperature(p_val, self.temperature)
        self.threshold_temperature = select_threshold_max_f1(p_temp, y_val)
        p_iso = self.isotonic.predict(p_val)
        self.threshold_isotonic = select_threshold_max_f1(p_iso, y_val)
        p_platt = apply_platt(p_val, self.platt)
        self.threshold_platt = select_threshold_max_f1(p_platt, y_val)

        self.conformal_threshold_alpha05 = conformal_benign_threshold(p_val[y_val == 0], alpha=0.05)
        return self

    def apply_all(self, p):
        return {
            "uncalibrated": p,
            "temperature": apply_temperature(p, self.temperature),
            "isotonic": self.isotonic.predict(p),
            "platt": apply_platt(p, self.platt),
        }
