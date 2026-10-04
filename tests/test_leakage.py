import pandas as pd
from src.data import FEATURES, LEAKY, RESOLVED, COLS, clean
from src.sample_data import make_sample

def test_no_leaky_columns_in_features():
    assert not (LEAKY & set(FEATURES))

def test_no_leaky_columns_are_loaded():
    assert not (LEAKY & set(COLS))

def test_target_has_only_resolved_loans():
    df = clean(make_sample(5000))
    assert set(df["loan_status"].unique()) <= set(RESOLVED)

def test_target_is_binary_and_matches_status():
    df = clean(make_sample(5000))
    assert set(df["default"].unique()) <= {0, 1}
    assert (df.loc[df.default == 1, "loan_status"] == "Charged Off").all()

def test_text_columns_became_numbers():
    df = clean(make_sample(5000))
    assert pd.api.types.is_numeric_dtype(df["term_months"])
    assert pd.api.types.is_numeric_dtype(df["emp_years"])