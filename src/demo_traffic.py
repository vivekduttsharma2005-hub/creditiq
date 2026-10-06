import argparse, json
from pathlib import Path
import pandas as pd
from src.data import load_data, time_split


def to_payload(r):
    """Ek row ko API wale format mein badlo."""
    f = lambda v: None if pd.isna(v) else float(v)       # optional: missing -> null
    i = lambda v: 0 if pd.isna(v) else int(v)            # counts: missing -> 0
    return {"loan_amnt": float(r.loan_amnt), "term_months": int(r.term_months), "installment": float(r.installment),
            "annual_inc": float(r.annual_inc), "dti": f(r.dti), "revol_util": f(r.revol_util),
            "revol_bal": float(r.revol_bal), "delinq_2yrs": i(r.delinq_2yrs), "open_acc": i(r.open_acc),
            "pub_rec": i(r.pub_rec), "total_acc": i(r.total_acc), "fico_range_low": int(r.fico_range_low),
            "fico_range_high": int(r.fico_range_high), "emp_years": f(r.emp_years),
            "credit_history_years": f((r.issue_d - r.earliest_cr_line).days / 365.25),
            "home_ownership": r.home_ownership if r.home_ownership in {"RENT", "MORTGAGE", "OWN"} else "OTHER",
            "verification_status": r.verification_status, "purpose": str(r.purpose)}


def main(source="real", path=None, n=300, shift=False):
    _, val, _ = time_split(load_data(source, path))
    out = Path("logs") / ("demo_shifted.jsonl" if shift else "demo_normal.jsonl")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        for r in val.sample(min(n, len(val)), random_state=0).itertuples():
            p = to_payload(r)
            if shift:                                     # dti badh gaya, income ghat gayi
                p["dti"] = None if p["dti"] is None else min(p["dti"] + 10, 100)
                p["annual_inc"] *= 0.6
            f.write(json.dumps({"input": p}) + "\n")
    print(f"Likha: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="real", choices=["real", "sample"])
    ap.add_argument("--path", default=None)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--shift", action="store_true")
    a = ap.parse_args()
    main(a.source, a.path, a.n, a.shift)
