"""FAKE Lending-Club-shaped data for practice and tests. Not real loans."""
import numpy as np, pandas as pd
from pathlib import Path

def make_sample(n=30000, seed=0):
    rng = np.random.default_rng(seed)
    issue = pd.Timestamp("2012-01-01") + pd.to_timedelta(rng.integers(0, 365*6, n), unit="D")
    df = pd.DataFrame({
        "loan_amnt": rng.integers(1000, 40000, n),
        "term": rng.choice([" 36 months", " 60 months"], n, p=[.7, .3]),
        "int_rate": rng.uniform(6, 28, n).round(2),
        "grade": rng.choice(list("ABCDEFG"), n),
        "emp_length": rng.choice(["< 1 year", "1 year", "5 years", "10+ years", None], n),
        "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN"], n),
        "annual_inc": rng.lognormal(11, 0.6, n).round(),
        "verification_status": rng.choice(["Verified", "Not Verified", "Source Verified"], n),
        "purpose": rng.choice(["debt_consolidation", "credit_card", "car", "small_business"], n),
        "dti": rng.normal(18, 8, n).clip(0, 60).round(2),
        "delinq_2yrs": rng.poisson(0.3, n),
        "open_acc": rng.integers(2, 30, n),
        "pub_rec": rng.poisson(0.1, n),
        "revol_bal": rng.integers(0, 60000, n),
        "revol_util": rng.uniform(0, 100, n).round(1),
        "total_acc": rng.integers(4, 60, n),
        "fico_range_low": (rng.integers(66, 85, n) * 5),
        "issue_d": issue.strftime("%b-%Y"),
    })
    df["installment"] = (df.loan_amnt / df.term.str.extract(r"(\d+)")[0].astype(int) * 1.15).round(2)
    df["fico_range_high"] = df.fico_range_low + 4
    df["earliest_cr_line"] = (issue - pd.to_timedelta(rng.integers(365*2, 365*30, n), unit="D")).strftime("%b-%Y")
    z = (-4.5 + 0.04*df.dti + 0.015*df.revol_util + 0.5*df.term.str.contains("60")
         + 0.4*df.delinq_2yrs - 0.00001*df.annual_inc + 0.1*(df.int_rate - 15))
    bad = rng.random(n) < 1/(1+np.exp(-z))
    status = np.where(bad, "Charged Off", "Fully Paid")
    still_open = (issue > pd.Timestamp("2016-06-01")) & (rng.random(n) < 0.6)   # unresolved loans
    status = np.where(still_open, rng.choice(["Current", "Late (31-120 days)", "In Grace Period"], n), status)
    df["loan_status"] = status
    # LEAKY columns (recorded after issue) - present in real file, we must never use them
    df["total_pymnt"] = np.where(bad, df.loan_amnt * 0.4, df.loan_amnt * 1.15)
    df["recoveries"] = np.where(bad, df.loan_amnt * 0.05, 0.0)
    df["last_pymnt_d"] = "Dec-2017"
    return df

SAMPLE_PATH = Path("data/sample/sample_accepted.csv")

if __name__ == "__main__":
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    make_sample(n=5000, seed=0).to_csv(SAMPLE_PATH, index=False)
    print(f"Wrote FAKE data -> {SAMPLE_PATH}")
