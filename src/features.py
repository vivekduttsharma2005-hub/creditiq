import pandas as pd

from src.data import NUMERIC_FEATURES, CATEGORICAL_FEATURES


MISSING_FLAG_COLS = ["revol_util", "emp_years", "dti"]
CAP_COLS = ["annual_inc", "dti"]

GRADE_MAP = {g: i + 1 for i, g in enumerate("ABCDEFG")}


class FeatureBuilder:

    def __init__(self, use_lender_cols=False, cap_quantile=0.99):
        self.use_lender_cols = use_lender_cols
        self.cap_quantile = cap_quantile

    def _base(self, df):
        num = df[NUMERIC_FEATURES].copy()

        num["credit_history_years"] = (
            (df["issue_d"] - df["earliest_cr_line"]).dt.days / 365.25
        )

        num["fico_avg"] = (
            df["fico_range_low"] + df["fico_range_high"]
        ) / 2

        num = num.drop(
            columns=["fico_range_low", "fico_range_high"]
        )

        if self.use_lender_cols:
            num["int_rate"] = df["int_rate"]
            num["grade_num"] = df["grade"].map(GRADE_MAP)

        return num

    def fit(self, train):
        base = self._base(train)

        self.medians_ = base.median()

        self.caps_ = {
            c: base[c].quantile(self.cap_quantile)
            for c in CAP_COLS
        }

        self.categories_ = {
            c: sorted(train[c].dropna().unique())
            for c in CATEGORICAL_FEATURES
        }

        self.columns_ = list(
            self.transform(train).columns
        )

        return self

    def transform(self, df):
        base = self._base(df)

        flags = pd.DataFrame({
            f"{c}_missing": base[c].isna().astype(int)
            for c in MISSING_FLAG_COLS
        }, index=df.index)

        num = base.fillna(self.medians_)

        for c, cap in self.caps_.items():
            num[c] = num[c].clip(upper=cap)

        out = pd.concat([num, flags], axis=1)

        out["loan_to_income"] = (
            num["loan_amnt"] / (num["annual_inc"] + 1)
        )

        out["installment_to_income"] = (
            num["installment"] * 12 / (num["annual_inc"] + 1)
        )

        out["high_util"] = (
            num["revol_util"] > 80
        ).astype(int)

        out["util_x_dti"] = (
            num["revol_util"] * num["dti"] / 100
        )

        for c, cats in self.categories_.items():
            for cat in cats:
                out[f"{c}_{cat}"] = (
                    df[c] == cat
                ).astype(int)

        if hasattr(self, "columns_"):
            out = out.reindex(
                columns=self.columns_,
                fill_value=0
            )

        return out