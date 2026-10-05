"""Day 6b: diagnostics on VALIDATION (we can still act on them) and ONE final look at TEST.
  python -m src.final_report diagnostics --source sample
  python -m src.final_report test --source sample        # refuses to run twice unless --force"""
import argparse, json
from pathlib import Path
import joblib
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.data import load_data, time_split
from src.evaluate import evaluate, expected_calibration_error
from src.explain import reason_codes
from src.features import FeatureBuilder
from src.threshold import expected_cost


def _load(source, path, models_dir):
    art = joblib.load(Path(models_dir) / f"calibrated_{source}.joblib")
    train, val, test = time_split(load_data(source, path))
    return art, train, val, test


def _score(art, df):
    X = art["feature_builder"].transform(df)
    raw = art["model"].predict_proba(X)[:, 1]
    return X, raw, art["calibrator"].predict(raw)


def segment_table(df, p, thr, train):
    """Does the model behave similarly across groups? Compare actual vs predicted risk and decline rate."""
    d = df.copy(); d["p"] = p; d["declined"] = p >= thr
    edges = np.r_[-np.inf, train["annual_inc"].quantile([.25, .5, .75]).to_numpy(), np.inf]
    d["income_band"] = pd.cut(d["annual_inc"], edges, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])
    d["emp_band"] = pd.cut(d["emp_years"].fillna(-1), [-2, -0.5, 1.5, 9.5, 99],
                           labels=["unknown", "0-1 yrs", "2-9 yrs", "10+ yrs"])
    d["issue_year"] = d["issue_d"].dt.year
    rows = []
    for col in ["income_band", "emp_band", "term_months", "home_ownership", "purpose", "issue_year"]:
        for level, g in d.groupby(col, observed=True):
            if len(g) < 30:
                continue
            rows.append({"segment": col, "group": str(level), "n": len(g),
                         "actual_default_%": round(100 * g["default"].mean(), 1),
                         "avg_predicted_%": round(100 * g["p"].mean(), 1),
                         "declined_%": round(100 * g["declined"].mean(), 1)})
    t = pd.DataFrame(rows)
    t["calibration_gap_pp"] = (t["avg_predicted_%"] - t["actual_default_%"]).round(1)
    return t


def diagnostics(source="real", path=None, models_dir="models", reports_dir="reports"):
    reports_dir = Path(reports_dir); (reports_dir / "tables").mkdir(parents=True, exist_ok=True)
    art, train, val, _ = _load(source, path, models_dir)       # test is not used here
    X, raw, p = _score(art, val)
    thr = art["decline_threshold"]
    print(f"Diagnostics on VALIDATION | cutoff {thr:.2f} | declined {100*(p >= thr).mean():.1f}%\n")

    seg = segment_table(val, p, thr, train)
    seg.to_csv(reports_dir / "tables" / f"segments_{source}.csv", index=False)
    print("SEGMENT CHECK (calibration gap = predicted% - actual%):\n", seg.to_string(index=False))
    worst = seg.reindex(seg["calibration_gap_pp"].abs().sort_values(ascending=False).index).head(3)
    print("\nLargest gaps:\n", worst[["segment", "group", "n", "calibration_gap_pp"]].to_string(index=False))

    # error analysis
    y = val["default"].to_numpy(); decline = p >= thr
    kind = np.select([decline & (y == 1), ~decline & (y == 1), decline & (y == 0)],
                     ["caught defaulter", "MISSED defaulter", "wrongly declined"], "approved good")
    cols = ["dti", "revol_util", "annual_inc", "fico_avg", "loan_to_income"]
    prof = X.assign(kind=kind)[cols + ["kind"]].groupby("kind").median().round(2)
    prof["count"] = pd.Series(kind).value_counts()
    print("\nERROR PROFILE (median values per outcome):\n", prof.to_string())
    prof.to_csv(reports_dir / "tables" / f"error_profile_{source}.csv")

    reference = art["feature_builder"].transform(train).median()
    miss = np.where(kind == "MISSED defaulter")[0]
    miss = miss[np.argsort(p[miss])][:5]                          # lowest-scored missed defaulters
    wrong = np.where(kind == "wrongly declined")[0]
    wrong = wrong[np.argsort(-p[wrong])][:5]                      # highest-scored good customers
    out = []
    for label, idx in [("MISSED defaulter", miss), ("wrongly declined", wrong)]:
        if len(idx):
            for i, r in zip(idx, reason_codes(art["model"], X.iloc[idx], reference)):
                out.append({"type": label, "risk_%": round(100 * p[i], 1), "top_risk_reasons": " | ".join(r) or "(none)"})
    pd.DataFrame(out).to_csv(reports_dir / "tables" / f"error_examples_{source}.csv", index=False)
    print("\nWorst mistakes:\n", pd.DataFrame(out).to_string(index=False))


def final_test(source="real", path=None, models_dir="models", reports_dir="reports", force=False):
    reports_dir = Path(reports_dir); reports_dir.mkdir(exist_ok=True)
    out = reports_dir / f"final_test_{source}.json"
    if out.exists() and not force:
        raise RuntimeError(f"{out} already exists. The test set is for ONE look. Use --force only if you accept that.")
    art, train, val, test = _load(source, path, models_dir)
    X, raw, p = _score(art, test)
    y, thr = test["default"].to_numpy(), art["decline_threshold"]

    fb = FeatureBuilder().fit(train)                               # baseline logistic, same train data
    lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    lr.fit(fb.transform(train), train["default"])
    res = {"base_rate": evaluate(y, np.full(len(y), train["default"].mean())),
           "logistic_no_lender": evaluate(y, lr.predict_proba(fb.transform(test))[:, 1]),
           "lgbm_raw": {**evaluate(y, raw), "ece": expected_calibration_error(y, raw)},
           "lgbm_calibrated": {**evaluate(y, p), "ece": expected_calibration_error(y, p)}}
    decline = p >= thr
    c_fn, c_fp = art["cost_fn"], art["cost_fp"]
    res["decision"] = {"cutoff": thr, "n_test": int(len(y)), "test_default_rate_%": round(100 * y.mean(), 2),
                       "declined_%": round(100 * decline.mean(), 1),
                       "defaulters_caught_%": round(100 * (decline & (y == 1)).sum() / max((y == 1).sum(), 1), 1),
                       "cost_with_model": expected_cost(y, p, thr, c_fn, c_fp),
                       "cost_approve_all": expected_cost(y, p, 1.01, c_fn, c_fp)}
    out.write_text(json.dumps(res, indent=2))
    print(pd.DataFrame({k: v for k, v in res.items() if k != "decision"}).T.round(4).to_string())
    print("\nDecision summary:", json.dumps(res["decision"], indent=2))
    print(f"\nSaved {out}  (test set now used)")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["diagnostics", "test"])
    ap.add_argument("--source", default="real", choices=["real", "sample"])
    ap.add_argument("--path", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    diagnostics(a.source, a.path) if a.mode == "diagnostics" else final_test(a.source, a.path, force=a.force)
