import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.calibrate import PlattCalibrator, IsotonicCalibrator, split_validation_by_time
from src.data import clean, time_split
from src.evaluate import evaluate, expected_calibration_error
from src.features import FeatureBuilder
from src.sample_data import make_sample
from src.threshold import (expected_cost, theoretical_threshold, best_threshold, decision)


def _honest_world(n=200000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.beta(1, 6, n)                    # true risks, mostly small
    y = (rng.random(n) < p).astype(int)      # outcomes drawn from those risks
    return y, p


def test_calibrators_output_valid_monotone_probabilities():
    y, p = _honest_world(20000)
    grid = np.linspace(0, 1, 200)
    for cal in (PlattCalibrator().fit(p, y), IsotonicCalibrator().fit(p, y)):
        out = cal.predict(grid)
        assert ((out >= 0) & (out <= 1)).all() and (np.diff(out) >= -1e-9).all()


def test_calibration_fixes_a_weighted_model():
    train, val, _ = time_split(clean(make_sample(30000, seed=3)))
    fb = FeatureBuilder().fit(train)
    m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    m.fit(fb.transform(train), train.default)
    p = m.predict_proba(fb.transform(val))[:, 1]; y = val.default.to_numpy()
    a, b = split_validation_by_time(val)
    fixed = PlattCalibrator().fit(p[a], y[a]).predict(p[b])
    assert evaluate(y[b], fixed)["brier"] < evaluate(y[b], p[b])["brier"] * 0.5


def test_ece_is_zero_for_honest_constant_prediction():
    y = np.array([0, 1] * 50)
    assert expected_calibration_error(y, np.full(100, 0.5)) == 0.0


def test_expected_cost_hand_example():
    y = np.array([1, 0, 1, 0]); p = np.array([0.9, 0.8, 0.2, 0.1])
    assert expected_cost(y, p, 0.5, cost_fn=5, cost_fp=1) == 6   # 1 FN (x5) + 1 FP (x1)


def test_best_threshold_matches_theory_when_probabilities_are_honest():
    y, p = _honest_world()
    thr, _ = best_threshold(y, p, cost_fn=5, cost_fp=1)
    assert abs(thr - theoretical_threshold(5, 1)) < 0.03


def test_higher_miss_cost_lowers_the_threshold():
    y, p = _honest_world()
    assert best_threshold(y, p, 10, 1)[0] < best_threshold(y, p, 2, 1)[0]


def test_decision_bands():
    assert decision(0.30, 0.2) == "decline"
    assert decision(0.15, 0.2) == "review"
    assert decision(0.05, 0.2) == "approve"