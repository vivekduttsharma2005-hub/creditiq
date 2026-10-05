"""Calibrator classes live in their OWN module so saved models can always be loaded.
(If they lived in a script run with `python -m`, pickle would store them as __main__.X and loading would fail.)"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """Squeeze/stretch the model's scores with a tiny logistic regression (a smooth S-curve)."""
    def fit(self, p, y):
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        return self

    def predict(self, p):
        return self.lr_.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Flexible staircase: 'when the model says ~X, the real rate was Y'. Needs more data."""
    def fit(self, p, y):
        self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        return self

    def predict(self, p):
        return self.iso_.predict(p)
