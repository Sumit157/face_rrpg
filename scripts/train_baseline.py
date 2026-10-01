#!/usr/bin/env python3
"""Phase 8: train classical ML baselines (Logistic Regression, Random
Forest, SVM) on the feature table from extract_features.py, using
stratified subject-grouped cross-validation so the same person's clips
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from app.features.feature_vector import FEATURE_NAMES
from app.models.random_forest import train_random_forest
from app.models.svm import train_svm
from app.core.logging import get_logger

logger = get_logger("train_baseline")


def load_dataset(features_csv: str):
    df = pd.read_csv(features_csv)
    required = set(FEATURE_NAMES) | {"label", "subjectId"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Feature table is missing required columns: {', '.join(missing)}")

    labels = df["label"].astype(str).str.strip().str.lower()
    unexpected = sorted(set(labels) - {"real", "fake"})
    if unexpected:
        raise ValueError(f"Labels must be 'real' or 'fake'; found: {unexpected}")
    if labels.nunique() != 2:
        raise ValueError("Training requires examples from both real and fake classes")
    if df["subjectId"].isna().any():
        raise ValueError("Every training row must have a subjectId for subject-independent validation")
    if df[FEATURE_NAMES].isna().any().any() or not np.isfinite(df[FEATURE_NAMES].to_numpy(dtype=float)).all():
        raise ValueError("Feature columns must contain only finite numeric values")
    if df["subjectId"].nunique() < 5:
        raise ValueError("At least five distinct subjects are required for 5-fold validation")

    X = df[FEATURE_NAMES].values
    y = (labels == "real").astype(int).values  # 1 = REAL, 0 = FAKE
    groups = df["subjectId"].values
    return X, y, groups, df


def evaluate_group_cv(build_fn, X, y, groups, n_splits=5) -> dict:
    splitter = StratifiedGroupKFold(
        n_splits=min(n_splits, len(set(groups))), shuffle=True, random_state=42
    )
    aucs, accs, balanced_accs, f1s, real_recalls, fake_recalls = [], [], [], [], [], []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        if len(np.unique(y[train_idx])) != 2 or len(np.unique(y[test_idx])) != 2:
            raise ValueError(
                f"Fold {fold} does not contain both classes in train and validation. "
                "Collect more subjects from each class or repair the subject labels."
            )
        model = build_fn(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        pred = (proba >= 0.5).astype(int)
        aucs.append(roc_auc_score(y[test_idx], proba))
        accs.append(accuracy_score(y[test_idx], pred))
        balanced_accs.append(balanced_accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, average="macro"))
        real_recalls.append(recall_score(y[test_idx], pred, pos_label=1))
        fake_recalls.append(recall_score(y[test_idx], pred, pos_label=0))
    return {
        "meanAuc": float(np.mean(aucs)),
        "stdAuc": float(np.std(aucs)),
        "meanAccuracy": float(np.mean(accs)),
        "meanBalancedAccuracy": float(np.mean(balanced_accs)),
        "meanMacroF1": float(np.mean(f1s)),
        "meanRealRecall": float(np.mean(real_recalls)),
        "meanFakeRecall": float(np.mean(fake_recalls)),
        "folds": len(aucs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features/features.csv")
    parser.add_argument("--output", default="models/production/baseline.joblib")
    parser.add_argument("--results", default="experiments/results/baseline_cv_results.json")
    parser.add_argument(
        "--min-cv-balanced-accuracy", type=float, default=0.80,
        help="Do not promote a model below this subject-held-out balanced accuracy (default: 0.80)",
    )
    parser.add_argument(
        "--min-class-recall", type=float, default=0.75,
        help="Do not promote if either class recall is below this value (default: 0.75)",
    )
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

    best_name = max(
        results,
        key=lambda k: (results[k]["meanBalancedAccuracy"], results[k]["meanAuc"]),
    )
    best_metrics = results[best_name]
    logger.info("Best model by balanced accuracy then AUC: %s", best_name)

    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_payload = {
        "results": results,
        "selectedModel": best_name,
        "selectionMetric": "meanBalancedAccuracy_then_meanAuc",
        "promotionThreshold": args.min_cv_balanced_accuracy,
        "minimumClassRecall": args.min_class_recall,
        "trainingSubjects": int(len(set(groups))),
        "trainingSamples": int(len(y)),
        "status": "promotion_pending",
    }
    results_path.write_text(json.dumps(results_payload, indent=2))

    if best_metrics["meanBalancedAccuracy"] < args.min_cv_balanced_accuracy:
        results_payload["status"] = "promotion_rejected_balanced_accuracy"
        results_path.write_text(json.dumps(results_payload, indent=2))
        raise SystemExit(
            f"Refusing to promote {best_name}: subject-held-out balanced accuracy "
            f"{best_metrics['meanBalancedAccuracy']:.3f} is below the requested "
            f"{args.min_cv_balanced_accuracy:.3f}. Improve the labeled data/features "
            "and rerun; the existing model artifact was left unchanged."
        )
    if min(best_metrics["meanRealRecall"], best_metrics["meanFakeRecall"]) < args.min_class_recall:
        results_payload["status"] = "promotion_rejected_class_recall"
        results_path.write_text(json.dumps(results_payload, indent=2))
        raise SystemExit(
            f"Refusing to promote {best_name}: weakest class recall is "
            f"{min(best_metrics['meanRealRecall'], best_metrics['meanFakeRecall']):.3f}, "
            f"below the requested {args.min_class_recall:.3f}. The existing model "
            "artifact was left unchanged."
        )

    # Refit the best model on the FULL dataset for the shipped artifact.
    # Note: this final model's reported CV numbers above (not its
    # in-sample fit) are the honest estimate of its generalisation.
    final_builders = {
        "logistic_regression": build_logreg,
        "random_forest": lambda Xtr, ytr: train_random_forest(Xtr, ytr),
        "svm": lambda Xtr, ytr: train_svm(Xtr, ytr),
    }
    final_model = final_builders[best_name](X, y)
    final_model.model_version_ = f"{best_name}-realdata-v2"
    final_model.training_protocol_ = "stratified_group_5_fold_cv"
    final_model.training_subject_count_ = int(len(set(groups)))
    final_model.training_sample_count_ = int(len(y))
    final_model.cv_balanced_accuracy_ = best_metrics["meanBalancedAccuracy"]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, out_path)
    logger.info("Saved final model (%s) to %s", best_name, out_path)

    results_payload["status"] = "trained_subject_grouped_cv_independent_test_required"
    results_path.write_text(json.dumps(results_payload, indent=2))


if __name__ == "__main__":
    main()
