"""Day 5: make probabilities honest, then pick the cutoff by cost.
Run: python -m src.calibrate --source sample   (or real)   (needs Day 4 models)
Fit on validation (first half), judge on validation (second half). TEST stays untouched."""
import argparse, json
from pathlib import Path
import joblib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from src.calibrators import PlattCalibrator, IsotonicCalibrator
from src.data import load_data, time_split
from src.evaluate import evaluate, expected_calibration_error
from src.threshold import threshold_summary, best_threshold


def split_validation_by_time(val):
    """Earlier half fits the calibrator, later half judges it (never judge on what you fit)."""
    order = np.argsort(val["issue_d"].values, kind="stable")
    mid = len(order) // 2
    return order[:mid], order[mid:]


def run(source="real", path=None, models_dir="models", reports_dir="reports", cost_fn=5.0, cost_fp=1.0):
    models_dir, reports_dir = Path(models_dir), Path(reports_dir)
    (reports_dir / "figures").mkdir(parents=True, exist_ok=True)
    art = joblib.load(models_dir / f"no_lender_{source}.joblib")      # honest model (no lender columns)
    fb, model = art["feature_builder"], art["model"]
    df = load_data(source, path)
    train, val, _test = time_split(df)
    p = model.predict_proba(fb.transform(val))[:, 1]
    y = val["default"].to_numpy()
    a, b = split_validation_by_time(val)                               # a = fit part, b = judge part

    cals = {"platt": PlattCalibrator().fit(p[a], y[a]), "isotonic": IsotonicCalibrator().fit(p[a], y[a])}
    scored = {"raw": p[b], **{k: c.predict(p[b]) for k, c in cals.items()}}
    rows = {}
    for name, s in scored.items():
        rows[name] = {**evaluate(y[b], s), "ece": expected_calibration_error(y[b], s)}
    table = pd.DataFrame(rows).T[["brier", "ece", "roc_auc", "pr_auc"]].round(4)
    print("CALIBRATION (judged on the later half of validation)\n", table.to_string())

    best = min(cals, key=lambda k: rows[k]["brier"])
    print(f"\nChosen calibrator (lowest Brier): {best}")

    # reliability curve
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    top = 0.0
    for name, s in scored.items():
        frac, mean_p = calibration_curve(y[b], s, n_bins=10, strategy="quantile")
        ax.plot(mean_p, frac, marker="o", label=name)
        top = max(top, mean_p.max(), frac.max())
    top = min(1.0, top * 1.1)                       # zoom to where the data is
    ax.plot([0, top], [0, top], "k--", label="perfect")
    ax.set_xlim(0, top); ax.set_ylim(0, top)
    ax.set_xlabel("Predicted default probability"); ax.set_ylabel("Actual default rate")
    ax.set_title("Reliability curve (closer to the diagonal = more honest)"); ax.legend()
    plt.tight_layout(); plt.savefig(reports_dir / "figures" / f"reliability_{source}.png", dpi=90)

    # cost-based threshold, chosen on held-out calibrated probabilities
    held_out = scored[best]
    rows_thr = [threshold_summary(y[b], held_out, r * cost_fp, cost_fp) for r in (2, 5, 10)]
    thr_table = pd.DataFrame(rows_thr)
    print("\nTHRESHOLD BY COST RATIO (FN cost : FP cost)\n", thr_table.to_string(index=False))
    chosen, _ = best_threshold(y[b], held_out, cost_fn, cost_fp)

    # final calibrator uses ALL validation data; save everything the API will need
    final_cal = (PlattCalibrator if best == "platt" else IsotonicCalibrator)().fit(p, y)
    out = {**art, "calibrator": final_cal, "calibration_method": best,
           "decline_threshold": chosen, "cost_fn": cost_fn, "cost_fp": cost_fp,
           "reference": fb.transform(train).median().to_dict()}    # typical values, used to word the reasons
    joblib.dump(out, models_dir / f"calibrated_{source}.joblib")
    table.to_csv(reports_dir / f"calibration_{source}.csv")
    thr_table.to_csv(reports_dir / f"threshold_table_{source}.csv", index=False)
    (reports_dir / f"metrics_calibration_{source}.json").write_text(json.dumps(
        {"method": best, "decline_threshold": chosen, "cost_fn": cost_fn, "cost_fp": cost_fp}, indent=2))
    print(f"\nSaved models/calibrated_{source}.joblib (cutoff {chosen:.2f} at {cost_fn:.0f}:{cost_fp:.0f} cost)")
    return table, thr_table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="real", choices=["real", "sample"])
    ap.add_argument("--path", default=None)
    ap.add_argument("--cost-fn", type=float, default=5.0)
    a = ap.parse_args()
    run(a.source, a.path, cost_fn=a.cost_fn)
