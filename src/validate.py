"""Day 7a: data validation with Pandera. Bad data should fail LOUDLY, before it reaches the model."""
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema
from src.data import LEAKY, RESOLVED

_NUM = dict(coerce=True)


def _no_leaky_columns(df):
    return not (LEAKY & set(df.columns))


# Ranges are deliberately generous (real Lending Club has odd rows). They catch typos and broken
# files (negative loans, dti = 5000, term = 48), not unusual-but-possible borrowers.
_COLUMNS = {
        "loan_amnt": Column(float, [Check.gt(0), Check.le(100_000)], **_NUM),
        "term_months": Column(int, Check.isin([36, 60]), **_NUM),
        "installment": Column(float, Check.gt(0), **_NUM),
        "annual_inc": Column(float, [Check.ge(0), Check.le(100_000_000)], **_NUM),
        "dti": Column(float, [Check.ge(-1), Check.le(1000)], nullable=True, **_NUM),
        "revol_util": Column(float, [Check.ge(0), Check.le(1000)], nullable=True, **_NUM),
        "revol_bal": Column(float, Check.ge(0), **_NUM),
        "delinq_2yrs": Column(float, Check.ge(0), nullable=True, **_NUM),
        "open_acc": Column(float, Check.ge(0), nullable=True, **_NUM),
        "pub_rec": Column(float, Check.ge(0), nullable=True, **_NUM),
        "total_acc": Column(float, Check.ge(0), nullable=True, **_NUM),
        "fico_range_low": Column(float, Check.in_range(300, 850), **_NUM),
        "fico_range_high": Column(float, Check.in_range(300, 850), **_NUM),
        "emp_years": Column(float, Check.in_range(0, 10), nullable=True, **_NUM),
        "issue_d": Column(pa.DateTime, **_NUM),
        "earliest_cr_line": Column(pa.DateTime, nullable=True, **_NUM),
        "home_ownership": Column(str, nullable=False),
        "verification_status": Column(str, nullable=False),
        "purpose": Column(str, nullable=False),
        "loan_status": Column(str, Check.isin(RESOLVED)),
        "default": Column(int, Check.isin([0, 1]), **_NUM),
}

# Row-level rules: apply to a whole table AND to a single new applicant
_ROW_CHECKS = [
    Check(_no_leaky_columns, error="leaky columns present in data"),
    Check(lambda df: df["fico_range_high"] >= df["fico_range_low"], error="fico_high < fico_low"),
    Check(lambda df: df["earliest_cr_line"].isna() | (df["earliest_cr_line"] <= df["issue_d"]),
          error="credit history starts after the loan was issued"),
]
# Table-level rules: only make sense for a big table (one applicant with unknown dti is 100% "missing")
_TABLE_CHECKS = [
    Check(lambda df: df["revol_util"].isna().mean() <= 0.10, error="more than 10% revol_util missing"),
    Check(lambda df: df["dti"].isna().mean() <= 0.10, error="more than 10% dti missing"),
]

# strict=False: extra columns are fine; the leaky-column check is what guards us
CLEAN_SCHEMA = DataFrameSchema(_COLUMNS, checks=_ROW_CHECKS + _TABLE_CHECKS, strict=False)

# NEW applicant rows (no target / status columns, no table-level rules). The API calls this.
APPLICANT_SCHEMA = DataFrameSchema(
    {k: v for k, v in _COLUMNS.items() if k not in ("default", "loan_status")},
    checks=_ROW_CHECKS, strict=False)


def _fail(schema_errors, what):
    cases = schema_errors.failure_cases
    summary = (cases.groupby(["column", "check"], dropna=False).size()
               .sort_values(ascending=False).head(8).to_string())
    missing = cases.loc[cases["check"] == "column_in_dataframe", "failure_case"].tolist()
    extra = f"\nMissing columns: {missing}" if missing else ""
    raise ValueError(f"{what} failed validation ({len(cases)} problems). Top issues:\n{summary}{extra}") from None


def validate_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Check the cleaned training table. Reports ALL problems at once, not just the first."""
    try:
        CLEAN_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        _fail(e, "Cleaned data")
    return df


def validate_applicant(df: pd.DataFrame) -> pd.DataFrame:
    """Check applicant rows before scoring them."""
    try:
        APPLICANT_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        _fail(e, "Applicant data")
    return df
