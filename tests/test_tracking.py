import mlflow
from src.sample_data import make_sample
from src.tracking import log_run
from src.train import run


def _client(uri):
    mlflow.set_tracking_uri(uri)
    return mlflow.tracking.MlflowClient()


def test_log_run_records_params_metrics_and_tags(tmp_path):
    uri = f"sqlite:///{tmp_path}/t.db"
    rid = log_run("demo", {"lr": 0.03}, {"pr_auc": 0.5, "flag": True}, tags={"k": "v"}, tracking_uri=uri)
    r = _client(uri).get_run(rid)
    assert r.data.params["lr"] == "0.03" and r.data.metrics["pr_auc"] == 0.5
    assert "flag" not in r.data.metrics and r.data.tags["k"] == "v" and "git_commit" in r.data.tags


def test_training_with_tracking_logs_four_runs_and_one_champion_per_lender_setting(tmp_path):
    csv = tmp_path / "s.csv"; make_sample(12000, seed=13).to_csv(csv, index=False)
    uri = f"sqlite:///{tmp_path}/t.db"
    run("sample", path=csv, models_dir=tmp_path / "m", reports_dir=tmp_path / "r", track=True, tracking_uri=uri)
    c = _client(uri)
    runs = c.search_runs([c.get_experiment_by_name("creditiq").experiment_id])
    assert len(runs) == 4
    assert sum(r.data.tags["champion"] == "True" for r in runs) == 2