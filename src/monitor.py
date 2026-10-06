
import argparse
from pathlib import Path
import joblib
from src.data import load_data, time_split
from src.drift import drift_report
from src.request_log import read_log
from src.service import prepare_frame


def run(source="real", path=None, log_path="logs/requests.jsonl", min_rows=100, models_dir="models", reports_dir="reports"):
    art = joblib.load(Path(models_dir) / f"calibrated_{source}.joblib")
    fb = art["feature_builder"]
    rows = read_log(log_path)
    if len(rows) < min_rows:
        print(f"Sirf {len(rows)} requests mili. Kam se kam {min_rows} chahiye, warna PSI bharosemand nahi hota.")
        return None, "INSUFFICIENT DATA"

    train, _val, _test = time_split(load_data(source, path))          # training data = "normal" kaisa dikhta hai
    live = fb.transform(prepare_frame([r["input"] for r in rows]))   # nayi requests, wahi feature steps
    report = drift_report(fb.transform(train), live)

    n_alert, n_watch = (report.status == "ALERT").sum(), (report.status == "watch").sum()
    status = "ALERT" if n_alert else ("WATCH" if n_watch else "OK")
    print(f"{len(rows)} requests vs training data\n")
    print(report.head(10).to_string(index=False))
    print(f"\nSTATUS: {status}  ({n_alert} features alert, {n_watch} watch)")
    if n_alert:
        print("Alert wale features:", ", ".join(report[report.status == "ALERT"].feature))
    Path(reports_dir).mkdir(exist_ok=True)
    report.to_csv(Path(reports_dir) / f"monitor_{source}.csv", index=False)
    return report, status


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="real", choices=["real", "sample"])
    ap.add_argument("--path", default=None)
    ap.add_argument("--log", default="logs/requests.jsonl")
    a = ap.parse_args()
    run(a.source, a.path, a.log)
