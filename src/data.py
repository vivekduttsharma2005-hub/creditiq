import pandas as pd
from pathlib import Path

SAFE_COLS = [
    "loan_amnt",
    "term",
    "installment",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "fico_range_low",
    "fico_range_high",
]

LENDER_COLS = ["int_rate", "grade"]

META_COLS = ["issue_d", "loan_status"]

COLS = SAFE_COLS + LENDER_COLS + META_COLS

LEAKY = {
    "total_pymnt",
    "total_rec_prncp",
    "total_rec_int",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "out_prncp",
    "last_credit_pull_d",
}

RESOLVED = ["Fully Paid", "Charged Off"]

NUMERIC_FEATURES = [
    "loan_amnt",
    "term_months",
    "installment",
    "emp_years",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "fico_range_low",
    "fico_range_high",
]

CATEGORICAL_FEATURES = [
    "home_ownership",
    "verification_status",
    "purpose",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _parse_emp(x):
    if pd.isna(x):
        return float("nan")

    if x.startswith("<"):
        return 0.0

    return float("".join(ch for ch in x if ch.isdigit()))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Keep resolved loans, build target, turn text columns into numbers."""
    df = df.copy()

    df["issue_d"] = pd.to_datetime(
        df["issue_d"],
        format="%b-%Y",
    )

    df["earliest_cr_line"] = pd.to_datetime(
        df["earliest_cr_line"],
        format="%b-%Y",
    )

    df = df[df["loan_status"].isin(RESOLVED)].copy()

    df["default"] = (
        df["loan_status"] == "Charged Off"
    ).astype(int)

    df["term_months"] = (
        df["term"]
        .str.extract(r"(\d+)")[0]
        .astype(int)
    )

    df["emp_years"] = df["emp_length"].map(_parse_emp)

    return df


RAW_PATH = Path(
    "data/raw/lendingclub_download/accepted_2007_to_2018Q4.csv.gz"
)

SAMPLE_PATH = Path(
    "data/sample/sample_accepted.csv"
)


def load_data(
    source="real",
    path=None,
    validate=True,
) -> pd.DataFrame:
    """Load Lending Club data and validate it by default."""

    path = (
        Path(path)
        if path
        else (RAW_PATH if source == "real" else SAMPLE_PATH)
    )

    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")

    if source == "sample":
        print("WARNING: FAKE data. Results are NOT real.")

    raw = pd.read_csv(
        path,
        usecols=COLS,
        low_memory=False,
    )

    df = clean(raw)

    if validate:
        _validate_data(df)

    return df


def _validate_data(df: pd.DataFrame):
    """Basic checks to catch broken or unexpected data early."""

    required = set(COLS) | {
        "default",
        "term_months",
        "emp_years",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Dataset is empty after cleaning.")

    if not df["default"].isin([0, 1]).all():
        raise ValueError("Target 'default' must contain only 0 and 1.")

    if not df["issue_d"].notna().all():
        raise ValueError("Found missing issue dates.")

    if not df["loan_status"].isin(RESOLVED).all():
        raise ValueError(
            "Found unresolved loan statuses after cleaning."
        )

    if (df["loan_amnt"] <= 0).any():
        raise ValueError("loan_amnt must be greater than 0.")
TRAIN_END = "2015-01-01"

VAL_END = "2016-01-01"


def time_split(df: pd.DataFrame):
    """Older loans train, next year validates, latest tests. Mimics real use."""

    train = df[
        df["issue_d"] < TRAIN_END
    ].copy()

    val = df[
        (df["issue_d"] >= TRAIN_END)
        & (df["issue_d"] < VAL_END)
    ].copy()

    test = df[
        df["issue_d"] >= VAL_END
    ].copy()

    return train, val, test