"""Simple EDA for Lending Club data."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.data import load_data, time_split


# Where we will save our EDA results
FIG = Path("reports/figures")
TAB = Path("reports/tables")


def main(source):

    # Create folders if they don't exist
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(source)

    print("\n==============================")
    print("1. FIRST 5 ROWS")
    print("==============================")
    print(df.head())

    print("\n==============================")
    print("2. SHAPE OF DATA")
    print("==============================")
    print(df.shape)

    print("\n==============================")
    print("3. DATA INFORMATION")
    print("==============================")
    df.info()

    print("\n==============================")
    print("4. MISSING VALUES")
    print("==============================")

    missing = df.isnull().sum()

    print(missing[missing > 0])

    missing.to_csv(TAB / "missing_counts.csv")

    print("\n==============================")
    print("5. NUMERICAL SUMMARY")
    print("==============================")
    print(df.describe())

    print("\n==============================")
    print("6. TARGET VARIABLE")
    print("==============================")

    print("\nDefault counts:")
    print(df["default"].value_counts())

    print("\nDefault percentage:")
    print(
        df["default"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\n==============================")
    print("7. CATEGORICAL VARIABLES")
    print("==============================")

    print("\nGrade:")
    print(df["grade"].value_counts())

    print("\nPurpose:")
    print(df["purpose"].value_counts())

    print("\nTerm:")
    print(df["term_months"].value_counts())

    # Save simple categorical counts
    df["grade"].value_counts().to_csv(
        TAB / "grade_counts.csv"
    )

    df["purpose"].value_counts().to_csv(
        TAB / "purpose_counts.csv"
    )

    df["term_months"].value_counts().to_csv(
        TAB / "term_counts.csv"
    )

    print("\n==============================")
    print("8. FEATURE vs TARGET")
    print("==============================")

    print("\nDefault rate by Grade:")
    grade_default = (
        df.groupby("grade")["default"]
        .mean()
        .mul(100)
        .round(2)
    )

    print(grade_default)

    print("\nDefault rate by Purpose:")
    purpose_default = (
        df.groupby("purpose")["default"]
        .mean()
        .mul(100)
        .round(2)
    )

    print(purpose_default)

    print("\nDefault rate by Term:")
    term_default = (
        df.groupby("term_months")["default"]
        .mean()
        .mul(100)
        .round(2)
    )

    print(term_default)

    grade_default.to_csv(
        TAB / "default_by_grade.csv"
    )

    purpose_default.to_csv(
        TAB / "default_by_purpose.csv"
    )

    term_default.to_csv(
        TAB / "default_by_term.csv"
    )

    print("\n==============================")
    print("9. NUMERICAL VARIABLES")
    print("==============================")

    print("\nAnnual Income:")
    print(df["annual_inc"].describe())

    print("\nDTI:")
    print(df["dti"].describe())

    print("\nLoan Amount:")
    print(df["loan_amnt"].describe())

    print("\nInterest Rate:")
    print(df["int_rate"].describe())

    print("\n==============================")
    print("10. OUTLIER CHECK")
    print("==============================")

    print(
        "Annual income:",
        "median =", round(df["annual_inc"].median(), 2),
        "| 99% =", round(df["annual_inc"].quantile(0.99), 2),
        "| max =", round(df["annual_inc"].max(), 2)
    )

    print(
        "DTI:",
        "median =", round(df["dti"].median(), 2),
        "| 99% =", round(df["dti"].quantile(0.99), 2),
        "| max =", round(df["dti"].max(), 2)
    )

    print("\n==============================")
    print("11. TIME SPLIT")
    print("==============================")

    train, val, test = time_split(df)

    print("Train:", train.shape)
    print("Validation:", val.shape)
    print("Test:", test.shape)

    print("\n==============================")
    print("12. SIMPLE GRAPHS")
    print("==============================")

    # 1. Default distribution
    plt.figure(figsize=(6, 4))
    df["default"].value_counts().sort_index().plot(kind="bar")
    plt.title("Default Distribution")
    plt.xlabel("Default")
    plt.ylabel("Number of Loans")
    plt.tight_layout()
    plt.savefig(FIG / "default_distribution.png")
    plt.close()

    # 2. Annual income distribution
    plt.figure(figsize=(6, 4))
    df["annual_inc"].hist(bins=30)
    plt.title("Annual Income Distribution")
    plt.xlabel("Annual Income")
    plt.ylabel("Number of Loans")
    plt.tight_layout()
    plt.savefig(FIG / "annual_income_distribution.png")
    plt.close()

    # 3. DTI distribution
    plt.figure(figsize=(6, 4))
    df["dti"].hist(bins=30)
    plt.title("DTI Distribution")
    plt.xlabel("DTI")
    plt.ylabel("Number of Loans")
    plt.tight_layout()
    plt.savefig(FIG / "dti_distribution.png")
    plt.close()

    # 4. Default rate by grade
    plt.figure(figsize=(6, 4))
    grade_default.plot(kind="bar")
    plt.title("Default Rate by Grade")
    plt.xlabel("Grade")
    plt.ylabel("Default Rate (%)")
    plt.tight_layout()
    plt.savefig(FIG / "default_by_grade.png")
    plt.close()

    print("\nEDA completed successfully.")

    print("\nSaved files:")
    print("reports/figures/")
    print("reports/tables/")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default="real",
        choices=["real", "sample"]
    )

    args = parser.parse_args()

    main(args.source)