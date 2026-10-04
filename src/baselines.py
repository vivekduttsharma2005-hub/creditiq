"""Day 3: Baseline models."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_data, time_split
from src.features import FeatureBuilder
from src.evaluate import evaluate


def run(source="real", path=None):

    # 1. Load data
    df = load_data(source, path)

    # 2. Split data into train, validation and test
    train, val, test = time_split(df)

    # We don't use test today
    y_train = train["default"]
    y_val = val["default"]

    print(
        f"Train: {len(train):,} | "
        f"Validation: {len(val):,} | "
        f"Train default rate: {y_train.mean():.3f} | "
        f"Validation default rate: {y_val.mean():.3f}"
    )

    # 3. Base-rate model
    train_default_rate = y_train.mean()

    base_predictions = np.full(
        len(y_val),
        train_default_rate
    )

    results = {
        "base_rate": evaluate(
            y_val,
            base_predictions
        )
    }

    # 4. Logistic regression models
    models = [
        ("logistic_no_lender", False, "balanced"),
        ("logistic_no_lender_unweighted", False, None),
        ("logistic_with_lender", True, "balanced"),
    ]

    # 5. Train each model
    for name, use_lender_cols, class_weight in models:

        # Feature engineering
        feature_builder = FeatureBuilder(
            use_lender_cols=use_lender_cols
        )

        feature_builder.fit(train)

        X_train = feature_builder.transform(train)
        X_val = feature_builder.transform(val)

        # Model
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight=class_weight,
                max_iter=2000
            )
        )

        # Train
        model.fit(X_train, y_train)

        # Predict default probability
        predictions = model.predict_proba(X_val)[:, 1]

        # Evaluate
        results[name] = evaluate(
            y_val,
            predictions
        )

    # 6. Show results
    table = pd.DataFrame(results).T.round(3)

    print("\nVALIDATION RESULTS")
    print(table.to_string())

    # 7. Save results
    output_file = Path(
        "reports"
    ) / f"metrics_baselines_{source}.json"

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file.write_text(
        json.dumps(results, indent=2)
    )

    print(f"\nSaved: {output_file}")

    return table


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default="real",
        choices=["real", "sample"]
    )

    parser.add_argument(
        "--path",
        default=None
    )

    args = parser.parse_args()

    run(
        args.source,
        args.path
    )