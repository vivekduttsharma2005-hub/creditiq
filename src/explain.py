"""Day 6a: explain the model. SHAP = how much each feature pushed ONE applicant's risk up or down.
Run: python -m src.explain --source sample   (or real)   (needs Day 5 model)
Uses train + validation only."""
import argparse
from pathlib import Path
import joblib
import numpy as np, pandas as pd
from src.data import load_data, time_split

# feature -> (phrase when applicant's value is HIGH, phrase when it is LOW). None = use the neutral label.
LABELS = {
    "revol_util": ("High credit card utilization", None),
    "high_util": ("Credit cards nearly maxed out", None),
    "util_x_dti": ("High utilization combined with heavy debt", None),
    "dti": ("High debt-to-income ratio", None),
    "loan_to_income": ("Loan is large relative to income", None),
    "installment_to_income": ("Loan is large relative to income", None),
    "annual_inc": (None, "Low annual income"),
    "delinq_2yrs": ("Recent missed payments", None),
    "pub_rec": ("Public derogatory records", None),
    "fico_avg": (None, "Low credit score"),
    "credit_history_years": (None, "Short credit history"),
    "emp_years": (None, "Short employment history"),
    "term_months": ("Longer loan term", None),
    "loan_amnt": ("Large loan amount", None),
    "installment": ("High monthly installment", None),
    "revol_bal": ("High revolving balance", None),
    "open_acc": ("Many open credit lines", None),
    "total_acc": (None, "Few credit accounts on file"),
    "int_rate": ("High interest rate", None),
    "grade_num": ("Weak lender risk grade", None),
}
NEUTRAL = {"revol_util": "Credit utilization", "dti": "Debt-to-income ratio", "annual_inc": "Annual income",
           "fico_avg": "Credit score", "emp_years": "Employment length", "delinq_2yrs": "Past delinquencies",
           "term_months": "Loan term", "credit_history_years": "Credit history length",
           "total_acc": "Number of credit accounts", "open_acc": "Number of open credit lines",
           "pub_rec": "Public records", "revol_bal": "Revolving balance", "loan_amnt": "Loan amount",
           "installment": "Monthly installment", "loan_to_income": "Loan size relative to income",
           "installment_to_income": "Loan size relative to income", "high_util": "Credit utilization",
           "util_x_dti": "Utilization combined with debt", "int_rate": "Interest rate",
           "grade_num": "Lender risk grade"}
ONE_HOT = {"purpose_": "Loan purpose", "home_ownership_": "Home ownership", "verification_status_": "Income verification"}


def describe(feature, value, reference):
    """Turn (feature, applicant value) into a plain-language reason."""
    for prefix, name in ONE_HOT.items():
        if feature.startswith(prefix):
            return f"{name}: {feature[len(prefix):].replace('_', ' ')}"
    if feature.endswith("_missing"):
        return f"Missing information: {feature[:-8].replace('_', ' ')}"
    high, low = LABELS.get(feature, (None, None))
    is_high = value > reference.get(feature, value)
    text = high if is_high else low
    return text or NEUTRAL.get(feature, feature.replace("_", " ").capitalize())


def shap_matrix(model, X):
    """SHAP values as a (rows x features) array; positive = pushes risk UP.
    Uses LightGBM's built-in TreeSHAP (identical to the shap library for tree models, but no extra
    dependency, so the serving image stays small). Values are in log-odds; last column is the baseline."""
    return np.asarray(model.booster_.predict(X, pred_contrib=True))[:, :-1]


def reason_codes(model, X, reference, k=3):
    """Top-k plain-language reasons per applicant (only features that INCREASE risk, no duplicates)."""
    sv = shap_matrix(model, X)
    cols, out = list(X.columns), []
    for i in range(len(X)):
        reasons = []
        for j in np.argsort(-sv[i]):
            if sv[i, j] <= 0 or len(reasons) == k:
                break
            text = describe(cols[j], X.iloc[i, j], reference)
            if text not in reasons:
                reasons.append(text)
        out.append(reasons)
    return out


def run(source="real", path=None, models_dir="models", reports_dir="reports", sample_rows=5000):
    import matplotlib                                              # plotting only needed here, not when serving
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models_dir, reports_dir = Path(models_dir), Path(reports_dir)
    (reports_dir / "figures").mkdir(parents=True, exist_ok=True)
    art = joblib.load(models_dir / f"calibrated_{source}.joblib")
    fb, model = art["feature_builder"], art["model"]
    train, val, _test = time_split(load_data(source, path))       # test untouched
    reference = fb.transform(train).median()
    X = fb.transform(val)
    Xs = X.sample(min(sample_rows, len(X)), random_state=0)

    sv = shap_matrix(model, Xs)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False)
    print("Top 10 features by mean |SHAP|:\n", imp.head(10).round(4).to_string())
    top = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index, top.values); ax.set_xlabel("Mean |SHAP| (average push on risk)")
    ax.set_title("What drives the model (validation sample)")
    plt.tight_layout(); plt.savefig(reports_dir / "figures" / f"shap_importance_{source}.png", dpi=90)

    # reasons for the 5 riskiest applicants in the sample
    p = art["calibrator"].predict(model.predict_proba(Xs)[:, 1])
    risky = Xs.iloc[np.argsort(-p)[:5]]
    rows = [{"risk_pct": round(100 * float(pr), 1), "reasons": " | ".join(r)}
            for pr, r in zip(np.sort(p)[::-1][:5], reason_codes(model, risky, reference))]
    ex = pd.DataFrame(rows)
    print("\nExample reason codes (5 riskiest applicants):\n", ex.to_string(index=False))
    ex.to_csv(reports_dir / f"reason_examples_{source}.csv", index=False)
    imp.round(5).to_csv(reports_dir / f"shap_importance_{source}.csv", header=["mean_abs_shap"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="real", choices=["real", "sample"])
    ap.add_argument("--path", default=None)
    a = ap.parse_args()
    run(a.source, a.path)