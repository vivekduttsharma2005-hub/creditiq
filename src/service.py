"""Day 8a: the scoring logic (no web code here, so it is easy to test and reuse)."""
import joblib
import pandas as pd
from src.explain import reason_codes
from src.threshold import decision, risk_band
from src.validate import validate_applicant

DISCLAIMER = "Educational model, not a lending tool. Not validated for real credit decisions."
NUMERIC = ["loan_amnt", "term_months", "installment", "annual_inc", "dti", "revol_util", "revol_bal",
           "delinq_2yrs", "open_acc", "pub_rec", "total_acc", "fico_range_low", "fico_range_high", "emp_years"]


class ScoringService:
    def __init__(self, model_path):
        art = joblib.load(model_path)
        if "reference" not in art:
            raise RuntimeError("Model file is missing 'reference'. Re-run: python -m src.calibrate --source <real|sample>")
        self.fb, self.model, self.calibrator = art["feature_builder"], art["model"], art["calibrator"]
        self.threshold, self.reference = art["decline_threshold"], art["reference"]
        self.name, self.trained_on = art["name"], art["trained_on"]
        self.method = art["calibration_method"]

    @property
    def warning(self):
        return None if self.trained_on == "real" else "Model was trained on FAKE sample data. Scores are NOT real."

    def info(self):
        return {"model": self.name, "trained_on": self.trained_on, "calibration": self.method,
                "decline_cutoff": round(self.threshold, 4), "n_features": len(self.fb.columns_),
                "uses_lender_columns": self.fb.use_lender_cols, "warning": self.warning, "disclaimer": DISCLAIMER}

    def score(self, applicants):
        df = pd.DataFrame(applicants)
        df[NUMERIC] = df[NUMERIC].astype(float)                      # None -> NaN (will be flagged + imputed)
        today = pd.Timestamp.today().normalize()                     # application date = today
        df["issue_d"] = today
        years = df.pop("credit_history_years").astype(float)
        df["earliest_cr_line"] = today - pd.to_timedelta(years * 365.25, unit="D")
        validate_applicant(df)                                       # Pandera: second line of defence
        X = self.fb.transform(df)
        p = self.calibrator.predict(self.model.predict_proba(X)[:, 1])
        reasons = reason_codes(self.model, X, self.reference, k=3)
        decisions = [decision(pi, self.threshold) for pi in p]
        # Reasons are for ADVERSE outcomes (review/decline). For an approval they would be tiny, misleading pushes.
        reasons = [r if d != "approve" else [] for r, d in zip(reasons, decisions)]
        return [{"default_probability": round(float(pi), 4),
                 "risk_band": risk_band(pi, self.threshold),
                 "decision": d,
                 "top_risk_factors": r,
                 "cutoff": round(self.threshold, 4),
                 "model": self.name, "warning": self.warning, "disclaimer": DISCLAIMER}
                for pi, d, r in zip(p, decisions, reasons)]
