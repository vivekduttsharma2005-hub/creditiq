import argparse, json
from pathlib import Path
import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score
from src.data import load_data, time_split
from src.features import FeatureBuilder
from src.evaluate import evaluate



def fit_lgbm(X_tr, y_tr, X_va, y_va, weighted, seed=42):
    """Train LightGBM. Early stopping watches validation PR-AUC and stops when it stops improving."""
    spw = (y_tr == 0).sum() / (y_tr == 1).sum() if weighted else 1.0
    model = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.03, num_leaves=31, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        scale_pos_weight=spw, random_state=seed, verbose=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


def run(source="real", path=None, models_dir="models", reports_dir="reports", track=False, tracking_uri=None):
    models_dir, reports_dir = Path(models_dir), Path(reports_dir)
    models_dir.mkdir(exist_ok=True); reports_dir.mkdir(exist_ok=True)
    df = load_data(source, path)
    train, val, _test = time_split(df)                       # test deliberately unused
    y_tr, y_va = train["default"], val["default"]
    print(f"train {len(train):,} | val {len(val):,} | default rate {y_tr.mean():.3f} / {y_va.mean():.3f}")

    results, fitted, lg_params = {}, {}, {}
    for lender in (False, True):
        tag = "with_lender" if lender else "no_lender"
        fb = FeatureBuilder(use_lender_cols=lender).fit(train)   # learned from train only
        X_tr, X_va = fb.transform(train), fb.transform(val)
        for weighted in (False, True):
            name = f"lgbm_{tag}_{'weighted' if weighted else 'plain'}"
            model = fit_lgbm(X_tr, y_tr, X_va, y_va, weighted)
            m = evaluate(y_va, model.predict_proba(X_va)[:, 1])
            m["train_pr_auc"] = float(average_precision_score(y_tr, model.predict_proba(X_tr)[:, 1]))
            m["trees_used"] = int(model.best_iteration_)
            results[name], fitted[name] = m, (fb, model)
            lg_params[name] = {"source": source, "lender_cols": lender, "weighted": weighted,
                               "n_train": len(train), "n_val": len(val), "n_features": X_tr.shape[1],
                               "train_from": str(train["issue_d"].min().date()),
                               "train_to": str(train["issue_d"].max().date()),
                               **{k: model.get_params()[k] for k in
                                  ("learning_rate", "num_leaves", "min_child_samples", "scale_pos_weight")}}

    # keep the better of plain/weighted for each lender setting (chosen by validation PR-AUC)
    for lender in (False, True):
        tag = "with_lender" if lender else "no_lender"
        best = max((n for n in results if f"_{tag}_" in n), key=lambda n: results[n]["pr_auc"])
        fb, model = fitted[best]
        joblib.dump({"feature_builder": fb, "model": model, "features": fb.columns_,
                     "use_lender_cols": lender, "trained_on": source, "name": best},
                    models_dir / f"{tag}_{source}.joblib")
        results[best]["saved_as_champion"] = True
        print(f"champion ({tag}): {best}")

    if track:                                                # one MLflow record per model
        for name in results:
            log_run(name, lg_params[name], results[name], tracking_uri=tracking_uri,
                    tags={"champion": str(bool(results[name].get("saved_as_champion", False))), "source": source})
        print(f"Logged {len(results)} runs to MLflow (view: mlflow ui --backend-store-uri {tracking_uri or 'sqlite:///mlflow.db'})")

    table = pd.DataFrame(results).T
    show = ["roc_auc", "pr_auc", "ks", "brier", "train_pr_auc", "trees_used"]
    print("\nVALIDATION RESULTS (LightGBM)\n", table[show].astype(float).round(3).to_string())

    base_file = reports_dir / f"metrics_baselines_{source}.json"
    if base_file.exists():
        base = pd.DataFrame(json.loads(base_file.read_text())).T
        print("\nLOGISTIC BASELINES (Day 3), PR-AUC:\n", base[["pr_auc"]].round(3).to_string())

    # which features does the no-lender champion lean on?
    best = max((n for n in results if "_no_lender_" in n), key=lambda n: results[n]["pr_auc"])
    fb, model = fitted[best]
    imp = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=fb.columns_)
    imp = (imp / imp.sum()).sort_values(ascending=False).round(3)
    imp.to_csv(reports_dir / f"feature_importance_{source}.csv", header=["share_of_gain"])
    print("\nTop 8 features (share of gain):\n", imp.head(8).to_string())

    (reports_dir / f"metrics_lgbm_{source}.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved models/*_{source}.joblib and reports/metrics_lgbm_{source}.json")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="real", choices=["real", "sample"])
    ap.add_argument("--path", default=None)
    ap.add_argument("--no-track", action="store_true", help="skip MLflow logging")
    a = ap.parse_args()
    run(a.source, a.path, track=not a.no_track)