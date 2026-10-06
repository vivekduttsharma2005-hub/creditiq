import pytest
from src.data import clean, load_data, LEAKY
from src.sample_data import make_sample
from src.validate import validate_clean, validate_applicant


@pytest.fixture()
def good():
    df = clean(make_sample(3000, seed=8))
    return df.drop(columns=[c for c in LEAKY if c in df.columns])   # load_data never reads leaky columns


def test_good_data_passes(good):
    assert validate_clean(good) is good


@pytest.mark.parametrize("column,bad_value", [
    ("loan_amnt", -500), ("term_months", 48), ("default", 2), ("dti", 5000), ("fico_range_low", 100)])
def test_bad_values_are_caught_and_named(good, column, bad_value):
    bad = good.copy(); bad.iloc[0, bad.columns.get_loc(column)] = bad_value
    with pytest.raises(ValueError, match=column):
        validate_clean(bad)


def test_leaky_column_is_rejected(good):
    bad = good.copy(); bad["total_pymnt"] = 1.0
    with pytest.raises(ValueError, match="leaky"):
        validate_clean(bad)


def test_all_problems_reported_at_once(good):
    bad = good.copy()
    bad.iloc[0, bad.columns.get_loc("loan_amnt")] = -1
    bad.iloc[1, bad.columns.get_loc("term_months")] = 99
    with pytest.raises(ValueError) as e:
        validate_clean(bad)
    assert "loan_amnt" in str(e.value) and "term_months" in str(e.value)


def test_too_many_missing_values_is_caught(good):
    bad = good.copy(); bad.loc[bad.index[: int(len(bad) * 0.3)], "revol_util"] = float("nan")
    with pytest.raises(ValueError, match="revol_util"):
        validate_clean(bad)


def test_applicant_schema_accepts_features_only_row(good):
    row = good.drop(columns=["default", "loan_status"]).head(3)
    assert len(validate_applicant(row)) == 3


def test_applicant_schema_rejects_missing_column(good):
    row = good.drop(columns=["default", "loan_status", "loan_amnt"]).head(3)
    with pytest.raises(ValueError, match="loan_amnt"):
        validate_applicant(row)


def test_load_data_validates_by_default(tmp_path):
    df = make_sample(2000, seed=2); df.loc[:300, "loan_amnt"] = -5
    csv = tmp_path / "bad.csv"; df.to_csv(csv, index=False)
    with pytest.raises(ValueError):
        load_data("sample", path=csv)
    assert len(load_data("sample", path=csv, validate=False)) > 0     # escape hatch works