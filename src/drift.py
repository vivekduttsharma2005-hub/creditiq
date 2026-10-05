"""Day 7b: drift check with PSI (Population Stability Index).
PSI asks: 'does the data I see NOW look like the data I trained on?'
  < 0.10 stable | 0.10-0.25 watch | > 0.25 ALERT
Run: python -m src.drift --source sample [--simulate]      (features and scores only, no labels, no test set)"""
import argparse
from pathlib import Path
import joblib
import numpy as np, pandas as pd
from src.data import load_data, time_split


def psi(expected, actual, bins=10, eps=1e-4):
    e = np.asarray(expected, dtype=float); e = e[~np.isnan(e)]
    a = np.asarray(actual, dtype=float); a = a[~np.isnan(a)]
    values = np.unique(e)
    if len(values) <= bins:                                   # flags / few distinct values: compare each value
        e_cnt = np.array([(e == v).sum() for v in values], dtype=float)
        a_cnt = np.array([(a == v).sum() for v in values], dtype=float)
        a_cnt[-1] += (~np.isin(a, values)).sum()              # unseen values go to the last bucket
    else:                                                     # numbers: bins from the TRAINING quantiles
        edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e_cnt, a_cnt = np.histogram(e, edges)[0].astype(float), np.histogram(a, edges)[0].astype(float)
    pe, pa_ = e_cnt / e_cnt.sum() + eps, a_cnt / a_cnt.sum() + eps
    return float(np.sum((pa_ - pe) * np.log(pa_ / pe)))


def status(value):
    return "stable" if value < 0.10 else ("watch" if value < 0.25 else "ALERT")


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = [{"feature": c, "psi": round(psi(reference[c], current[c]), 4)} for c in reference.columns]
    out = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    out["status"] = out["psi"].map(status)
    return out


def run(source="real", path=None, models_dir="models", reports_dir="reports", simulate=False):
    reports_dir = Path(reports_dir); reports_dir.mkdir(exist_ok=True)
    art = joblib.load(Path(models_dir) / f"calibrated_{source}.joblib")
    fb, model, cal = art["feature_builder"], art["model"], art["calibrator"]
    train, val, _test = time_split(load_data(source, path))   # test untouched
    if simulate:                                              # pretend the economy got worse
        val = val.copy(); val["dti"] += 10; val["annual_inc"] *= 0.6
        print("SIMULATED SHIFT: dti +10, income x0.6\n")
    X_tr, X_cur = fb.transform(train), fb.transform(val)
    rep = drift_report(X_tr, X_cur)
    score = psi(cal.predict(model.predict_proba(X_tr)[:, 1]), cal.predict(model.predict_proba(X_cur)[:, 1]))
    print("FEATURE DRIFT (train vs validation), top 10:\n", rep.head(10).to_string(index=False))
    print(f"\nSCORE DRIFT (predicted risk): PSI = {score:.4f} -> {status(score)}")
    tag = "simulated" if simulate else "validation"
    rep.to_csv(reports_dir / f"drift_{source}_{tag}.csv", index=False)
    return rep, score


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="real", choices=["real", "sample"])
    ap.add_argument("--path", default=None)
    ap.add_argument("--simulate", action="store_true")
    a = ap.parse_args()
    run(a.source, a.path, simulate=a.simulate)
