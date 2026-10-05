"""Day 7c: experiment tracking with MLflow. Every training run leaves a record: what, how, how well, which code."""
import subprocess
import mlflow

# SQLite database file (newer MLflow no longer allows the old ./mlruns folder as the store)
TRACKING_URI = "sqlite:///mlflow.db"


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def log_run(run_name, params, metrics, tags=None, experiment="creditiq", tracking_uri=None):
    mlflow.set_tracking_uri(tracking_uri or TRACKING_URI)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)})
        mlflow.set_tags({"git_commit": git_commit(), **(tags or {})})
    return run.info.run_id
