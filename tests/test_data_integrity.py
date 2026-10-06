import pandas as pd
from src.data import clean, time_split, RESOLVED
from src.features import FeatureBuilder
from src.sample_data import make_sample


def test_split_is_a_clean_partition_of_the_data():
    df = clean(make_sample(20000, seed=5))
    train, val, test = time_split(df)
    assert len(train) + len(val) + len(test) == len(df)
    assert not (set(train.index) & set(val.index)) and not (set(val.index) & set(test.index))
    assert not (set(train.index) & set(test.index))


def test_cleaned_data_only_has_resolved_loans_and_valid_rate():
    df = clean(make_sample(10000, seed=5))
    assert set(df["loan_status"]) <= set(RESOLVED) and 0 < df["default"].mean() < 1


def test_features_are_all_numeric_and_keep_row_order():
    train, val, _ = time_split(clean(make_sample(20000, seed=5)))
    out = FeatureBuilder().fit(train).transform(val)
    assert all(pd.api.types.is_numeric_dtype(t) for t in out.dtypes)
    assert list(out.index) == list(val.index)