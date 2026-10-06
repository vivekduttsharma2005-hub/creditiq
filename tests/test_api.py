import pytest
from fastapi.testclient import TestClient
from src.api import create_app
from src.calibrate import run as calibrate_run
from src.sample_data import make_sample
from src.threshold import decision
from src.train import run as train_run

SAFE = dict(loan_amnt=5000, term_months=36, installment=160, annual_inc=150000, dti=5, revol_util=10,
            revol_bal=1000, delinq_2yrs=0, open_acc=8, pub_rec=0, total_acc=25, fico_range_low=400,
            fico_range_high=404, emp_years=10, credit_history_years=20, home_ownership="MORTGAGE",
            verification_status="Verified", purpose="car")
RISKY = dict(SAFE, loan_amnt=35000, term_months=60, installment=900, annual_inc=25000, dti=45,
             revol_util=95, delinq_2yrs=3, emp_years=0, credit_history_years=2, home_ownership="RENT",
             purpose="small_business")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("api"); csv = d / "s.csv"
    make_sample(20000, seed=31).to_csv(csv, index=False)
    kw = dict(models_dir=d / "models", reports_dir=d / "reports")
    train_run("sample", path=csv, **kw); calibrate_run("sample", path=csv, **kw)
    with TestClient(create_app(d / "models" / "calibrated_sample.joblib", log_path=None)) as c:
        yield c


def test_health_reports_model_and_fake_data_warning(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["trained_on"] == "sample" and "FAKE" in r["warning"]
    assert r["uses_lender_columns"] is False


def test_score_returns_complete_valid_answer(client):
    r = client.post("/score", json=SAFE)
    assert r.status_code == 200
    j = r.json()
    assert 0 <= j["default_probability"] <= 1
    assert j["risk_band"] in {"Low", "Medium", "High", "Very high"}
    assert j["decision"] == decision(j["default_probability"], j["cutoff"])
    assert len(j["top_risk_factors"]) <= 3 and j["disclaimer"]


def test_reasons_only_for_review_or_decline(client):
    for body in (SAFE, RISKY):
        j = client.post("/score", json=body).json()
        if j["decision"] == "approve":
            assert j["top_risk_factors"] == []
        else:
            assert 1 <= len(j["top_risk_factors"]) <= 3


def test_riskier_applicant_gets_higher_probability(client):
    safe = client.post("/score", json=SAFE).json()["default_probability"]
    risky = client.post("/score", json=RISKY).json()["default_probability"]
    assert risky > safe


def test_optional_fields_can_be_missing(client):
    body = {k: v for k, v in SAFE.items() if k not in ("dti", "revol_util", "emp_years", "credit_history_years")}
    assert client.post("/score", json=body).status_code == 200


@pytest.mark.parametrize("change", [
    {"term_months": 48}, {"loan_amnt": -5}, {"fico_range_high": 350, "fico_range_low": 500},
    {"home_ownership": "CASTLE"}, {"total_pymnt": 9999}])          # last one: a leaky field is rejected
def test_bad_input_is_rejected_with_422(client, change):
    assert client.post("/score", json={**SAFE, **change}).status_code == 422


def test_missing_required_field_is_rejected(client):
    body = {k: v for k, v in SAFE.items() if k != "loan_amnt"}
    assert client.post("/score", json=body).status_code == 422


def test_batch_keeps_order_and_limits_size(client):
    r = client.post("/score/batch", json=[SAFE, RISKY])
    assert r.status_code == 200 and len(r.json()) == 2
    assert r.json()[0]["default_probability"] < r.json()[1]["default_probability"]
    assert client.post("/score/batch", json=[SAFE] * 101).status_code == 422
