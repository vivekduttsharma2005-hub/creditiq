import pandas as pd
import pytest
from src.data import clean, time_split, LEAKY
from src.features import FeatureBuilder
from src.sample_data import make_sample


@pytest.fixture(scope="module")
def splits():
    return time_split(clean(make_sample(20000, seed=1)))


def test_split_is_in_time_order(splits):
    train, val, test = splits
    assert train.issue_d.max() < val.issue_d.min() <= val.issue_d.max() < test.issue_d.min()
    assert min(len(train), len(val), len(test)) > 0


def test_medians_and_caps_come_from_train_only(splits):
    train, val, _ = splits
    fb = FeatureBuilder().fit(train)
    assert fb.medians_["annual_inc"] == train.annual_inc.median()
    assert fb.caps_["annual_inc"] == train.annual_inc.quantile(0.99)


def test_no_missing_values_after_transform(splits):
    train, val, test = splits
    fb = FeatureBuilder().fit(train)
    for part in (train, val, test):
        assert fb.transform(part).isna().sum().sum() == 0


def test_missing_flags_match_raw_missing(splits):
    train, _, _ = splits
    out = FeatureBuilder().fit(train).transform(train)
    assert (out["revol_util_missing"] == train.revol_util.isna().astype(int)).all()


def test_outlier_is_capped_not_deleted(splits):
    train, val, _ = splits
    fb = FeatureBuilder().fit(train)
    v = val.copy(); v.iloc[0, v.columns.get_loc("annual_inc")] = 9_000_000
    out = fb.transform(v)
    assert len(out) == len(v)                                   # row kept
    assert out["annual_inc"].max() <= fb.caps_["annual_inc"]    # value capped


def test_same_columns_in_train_val_test(splits):
    train, val, test = splits
    fb = FeatureBuilder().fit(train)
    assert list(fb.transform(train).columns) == list(fb.transform(val).columns) == list(fb.transform(test).columns)


def test_no_leaky_columns_in_final_features(splits):
    train, _, _ = splits
    assert not (LEAKY & set(FeatureBuilder().fit(train).transform(train).columns))


def test_lender_columns_are_optional(splits):
    train, _, _ = splits
    assert "int_rate" not in FeatureBuilder().fit(train).transform(train).columns
    assert "int_rate" in FeatureBuilder(use_lender_cols=True).fit(train).transform(train).columns