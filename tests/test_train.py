import joblib
import numpy as np
import pytest
from src.sample_data import make_sample
from src.train import run


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    csv = d / "sample.csv"
    make_sample(15000, seed=11).to_csv(csv, index=False)
    table = run("sample", path=csv, models_dir=d / "models", reports_dir=d / "reports")
    return d, table


def test_trains_all_four_configs(trained):
    _, table = trained
    assert len(table) == 4 and table["pr_auc"].astype(float).min() > 0


def test_early_stopping_uses_fewer_trees_than_max(trained):
    _, table = trained
    assert (table["trees_used"].astype(int) < 2000).all()


def test_saved_model_loads_and_predicts_probabilities(trained):
    d, _ = trained
    art = joblib.load(d / "models" / "no_lender_sample.joblib")
    assert art["use_lender_cols"] is False
    from src.data import clean
    df = clean(make_sample(500, seed=2))
    p = art["model"].predict_proba(art["feature_builder"].transform(df))[:, 1]
    assert p.shape == (len(df),) and ((p >= 0) & (p <= 1)).all()


def test_no_lender_model_does_not_see_lender_columns(trained):
    d, _ = trained
    art = joblib.load(d / "models" / "no_lender_sample.joblib")
    assert "int_rate" not in art["features"] and "grade_num" not in art["features"]
    art2 = joblib.load(d / "models" / "with_lender_sample.joblib")
    assert "int_rate" in art2["features"]