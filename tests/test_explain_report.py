import numpy as np
import pandas as pd
import pytest
from src.sample_data import make_sample
from src.train import run as train_run
from src.calibrate import run as calibrate_run
from src.explain import describe, reason_codes, shap_matrix
from src.final_report import diagnostics, final_test, segment_table
from src.data import clean, time_split


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    d = tmp_path_factory.mktemp("p"); csv = d / "s.csv"
    make_sample(20000, seed=21).to_csv(csv, index=False)
    kw = dict(models_dir=d / "models", reports_dir=d / "reports")
    train_run("sample", path=csv, **kw)
    calibrate_run("sample", path=csv, **kw)
    return d, csv, kw


def test_describe_uses_direction_of_the_value():
    assert describe("annual_inc", 20000, {"annual_inc": 60000}) == "Low annual income"
    assert describe("revol_util", 95, {"revol_util": 50}) == "High credit card utilization"
    assert describe("purpose_car", 1, {}) == "Loan purpose: car"
    assert describe("dti_missing", 1, {}) == "Missing information: dti"
    assert describe("pub_rec", 0, {"pub_rec": 1}) == "Public records"      # never a raw column name


def test_shap_shape_and_reasons_are_unique_and_at_most_k(pipeline):
    import joblib
    d, csv, _ = pipeline
    art = joblib.load(d / "models" / "calibrated_sample.joblib")
    fb = art["feature_builder"]
    train, val, _ = time_split(clean(make_sample(20000, seed=21)))
    X = fb.transform(val).head(50)
    assert shap_matrix(art["model"], X).shape == X.shape
    for r in reason_codes(art["model"], X, fb.transform(train).median(), k=3):
        assert len(r) <= 3 and len(set(r)) == len(r)


def test_segment_table_has_expected_columns():
    train, val, _ = time_split(clean(make_sample(20000, seed=4)))
    t = segment_table(val, np.full(len(val), 0.05), 0.15, train)
    assert {"segment", "group", "n", "calibration_gap_pp", "declined_%"} <= set(t.columns)
    assert (t["declined_%"] == 0).all()


def test_diagnostics_runs_and_writes_tables(pipeline):
    d, csv, kw = pipeline
    diagnostics("sample", path=csv, **kw)
    assert (d / "reports" / "tables" / "segments_sample.csv").exists()


def test_final_test_runs_once_then_refuses(pipeline):
    d, csv, kw = pipeline
    res = final_test("sample", path=csv, **kw)
    assert {"lgbm_calibrated", "logistic_no_lender", "base_rate", "decision"} <= set(res)
    with pytest.raises(RuntimeError):
        final_test("sample", path=csv, **kw)