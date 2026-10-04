"""Metrics we report for every model. One place, so every model is judged the same way."""
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             brier_score_loss, precision_recall_curve)


def ks_statistic(y, p):
    """Biggest gap between the 'bad' and 'good' score curves. 0 = no separation, 1 = perfect."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def recall_at_precision(y, p, target=0.5):
    """Share of defaulters caught while being right at least `target` of the time."""
    precision, recall, _ = precision_recall_curve(y, p)
    ok = recall[precision >= target]
    return float(ok.max()) if len(ok) else 0.0


def evaluate(y, p, target_precision=0.5):
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "ks": ks_statistic(y, p),
        f"recall_at_precision_{int(target_precision * 100)}": recall_at_precision(y, p, target_precision),
        "brier": float(brier_score_loss(y, p)),
    }


def expected_calibration_error(y, p, n_bins=10):
    """Average gap between predicted risk and actual default rate, bin by bin. 0 = honest probabilities."""
    import pandas as pd
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    bins = pd.qcut(p, q=n_bins, duplicates="drop")
    g = pd.DataFrame({"y": y, "p": p, "bin": bins}).groupby("bin", observed=True)
    return float(sum(len(d) * abs(d.p.mean() - d.y.mean()) for _, d in g) / len(p))