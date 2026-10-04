import numpy as np
import pytest
from src.evaluate import ks_statistic, recall_at_precision, evaluate
from src.data import clean, time_split
from src.features import FeatureBuilder
from src.sample_data import make_sample
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

Y = np.array([0, 0, 1, 1])


def test_perfect_ranking_scores_one():
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = evaluate(Y, p)
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0 and m["ks"] == 1.0


def test_constant_scores_are_useless():
    m = evaluate(Y, np.full(4, 0.5))
    assert m["roc_auc"] == 0.5 and m["ks"] == 0.0


def test_recall_at_precision_simple_case():
    y = np.array([1, 1, 0, 0]); p = np.array([0.9, 0.8, 0.7, 0.1])
    assert recall_at_precision(y, p, target=1.0) == 1.0   # top 2 are both defaulters


def test_brier_zero_for_perfect_probabilities():
    assert evaluate(Y, np.array([0.0, 0.0, 1.0, 1.0]))["brier"] == 0.0


def test_evaluate_returns_all_metrics():
    keys = evaluate(Y, np.array([0.1, 0.2, 0.8, 0.9])).keys()
    assert {"roc_auc", "pr_auc", "ks", "brier"} <= set(keys)


def test_logistic_beats_base_rate_on_pr_auc():
    train, val, _ = time_split(clean(make_sample(20000, seed=3)))
    fb = FeatureBuilder().fit(train)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(fb.transform(train), train.default)
    p = model.predict_proba(fb.transform(val))[:, 1]
    base = evaluate(val.default, np.full(len(val), train.default.mean()))
    assert evaluate(val.default, p)["pr_auc"] > base["pr_auc"]
