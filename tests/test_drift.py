import numpy as np
from src.data import clean, time_split
from src.drift import psi, drift_report, status
from src.features import FeatureBuilder
from src.sample_data import make_sample

RNG = np.random.default_rng(0)


def test_identical_distributions_have_near_zero_psi():
    assert psi(RNG.normal(0, 1, 50000), RNG.normal(0, 1, 50000)) < 0.01


def test_shifted_distribution_raises_alert():
    assert psi(RNG.normal(0, 1, 50000), RNG.normal(1, 1, 50000)) > 0.25


def test_bigger_shift_means_bigger_psi():
    base = RNG.normal(0, 1, 50000)
    assert psi(base, RNG.normal(0.2, 1, 50000)) < psi(base, RNG.normal(0.6, 1, 50000)) < psi(base, RNG.normal(1.5, 1, 50000))


def test_flags_constants_and_nans_do_not_break_psi():
    flags_a, flags_b = RNG.integers(0, 2, 5000), RNG.integers(0, 2, 5000)
    assert np.isfinite(psi(flags_a, flags_b))
    assert np.isfinite(psi(np.zeros(100), np.zeros(100)))
    with_nan = np.r_[RNG.normal(0, 1, 1000), [np.nan] * 50]
    assert np.isfinite(psi(with_nan, with_nan))


def test_status_thresholds():
    assert (status(0.05), status(0.15), status(0.40)) == ("stable", "watch", "ALERT")


def test_report_flags_only_the_shifted_features():
    train, val, _ = time_split(clean(make_sample(30000, seed=6)))
    fb = FeatureBuilder().fit(train)
    shifted = val.copy(); shifted["dti"] += 10
    rep = drift_report(fb.transform(train), fb.transform(shifted)).set_index("feature")
    assert rep.loc["dti", "status"] == "ALERT"
    assert rep.loc["loan_amnt", "status"] == "stable"