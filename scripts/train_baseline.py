#!/usr/bin/env python3
"""Phase 8: train classical ML baselines (Logistic Regression, Random
Forest, SVM) on the feature table from extract_features.py, using
subject-aware (GroupKFold) cross-validation so the same person's clips
never leak across train/test -- required for a valid subject-independent
evaluation (spec Objective 12 / Definition of Done -> Research).

Usage:
    python scripts/train_baseline.py --features data/processed/features/features.csv \
        --output models/production/baseline.joblib
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "ml"))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from app.features.feature_vector import FEATURE_NAMES
from app.models.random_forest import train_random_forest
from app.models.svm import train_svm
from app.core.logging import get_logger

logger = get_logger("train_baseline")


def load_dataset(features_csv: str):
    df = pd.read_csv(features_csv)
    X = df[FEATURE_NAMES].values
    y = (df["label"] == "real").astype(int).values  # 1 = REAL, 0 = FAKE
    groups = df["subjectId"].values
    return X, y, groups, df


def evaluate_group_cv(build_fn, X, y, groups, n_splits=5) -> dict:
    gkf = GroupKFold(n_splits=min(n_splits, len(set(groups))))
    aucs, accs, f1s = [], [], []
    for train_idx, test_idx in gkf.split(X, y, groups):
        model = build_fn(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        pred = (proba > 0.5).astype(int)
        aucs.append(roc_auc_score(y[test_idx], proba))
        accs.append(accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred))
    return {
        "meanAuc": float(np.mean(aucs)), "stdAuc": float(np.std(aucs)),
        "meanAccuracy": float(np.mean(accs)), "meanF1": float(np.mean(f1s)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features/features.csv")
    parser.add_argument("--output", default="models/production/baseline.joblib")
    parser.add_argument("--results", default="experiments/results/baseline_cv_results.json")
    args = parser.parse_args()

    X, y, groups, df = load_dataset(args.features)
    logger.info("Loaded %d samples, %d subjects, %d real / %d fake",
                len(y), len(set(groups)), int(y.sum()), int((1 - y).sum()))

    results = {}

    def build_logreg(Xtr, ytr):
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
        pipe.fit(Xtr, ytr)
        return pipe

    results["logistic_regression"] = evaluate_group_cv(build_logreg, X, y, groups)
    results["random_forest"] = evaluate_group_cv(lambda Xtr, ytr: train_random_forest(Xtr, ytr), X, y, groups)
    results["svm"] = evaluate_group_cv(lambda Xtr, ytr: train_svm(Xtr, ytr), X, y, groups)

    logger.info("Cross-validated results:\n%s", json.dumps(results, indent=2))

    best_name = max(results, key=lambda k: results[k]["meanAuc"])
    logger.info("Best model by mean AUC: %s", best_name)

    # Refit the best model on the FULL dataset for the shipped artifact.
    # Note: this final model's reported CV numbers above (not its
    # in-sample fit) are the honest estimate of its generalisation.
    final_builders = {
        "logistic_regression": build_logreg,
        "random_forest": lambda Xtr, ytr: train_random_forest(Xtr, ytr),
        "svm": lambda Xtr, ytr: train_svm(Xtr, ytr),
    }
    final_model = final_builders[best_name](X, y)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, out_path)
    logger.info("Saved final model (%s) to %s", best_name, out_path)

    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"results": results, "selectedModel": best_name}, f, indent=2)


if __name__ == "__main__":
    main()
