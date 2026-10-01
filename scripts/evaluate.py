#!/usr/bin/env python3
"""Phase 8-9 / Milestone 15: evaluate a trained model on a held-out
feature table and produce the standard metrics + a confusion matrix
figure required by the Definition of Done (baseline comparison,
ablation, robustness).

Usage:
    python scripts/evaluate.py --model models/production/baseline.joblib \
        --features data/processed/features/features_test.csv
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "ml"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

from app.features.feature_vector import FEATURE_NAMES
from app.core.logging import get_logger

logger = get_logger("evaluate")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/production/baseline.joblib")
    parser.add_argument("--features", required=True)
    parser.add_argument(
        "--training-features", default="data/processed/features/features.csv",
        help="Training feature table used to reject subject overlap with the test table",
    )
    parser.add_argument("--output-dir", default="experiments/results")
    args = parser.parse_args()

    model = joblib.load(args.model)
    df = pd.read_csv(args.features)
    train_df = pd.read_csv(args.training_features)
    required = set(FEATURE_NAMES) | {"label", "subjectId"}
    for name, frame in (("test", df), ("training", train_df)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} feature table is missing columns: {', '.join(missing)}")

    labels = df["label"].astype(str).str.strip().str.lower()
    unexpected = sorted(set(labels) - {"real", "fake"})
    if unexpected or labels.nunique() != 2:
        raise ValueError("Test table must contain both 'real' and 'fake' labels only")
    if df["subjectId"].isna().any() or train_df["subjectId"].isna().any():
        raise ValueError("Train and test rows must have subjectId values")
    overlap = set(df["subjectId"]) & set(train_df["subjectId"])
    if overlap:
        raise ValueError(
            f"Evaluation is not subject-independent: {len(overlap)} subjectId values "
            "occur in both training and test tables"
        )
    for name, frame in (("test", df),):
        values = frame[FEATURE_NAMES].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} feature table contains missing or non-finite feature values")

    X = df[FEATURE_NAMES].values
    y = (labels == "real").astype(int).values

    class_index = list(model.classes_).index(1)
    proba = model.predict_proba(X)[:, class_index]
    pred = (proba > 0.5).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y, proba)),
        "accuracy": float(accuracy_score(y, pred)),
        "balancedAccuracy": float(balanced_accuracy_score(y, pred)),
        "macroF1": float(f1_score(y, pred, average="macro")),
        "realRecall": float(recall_score(y, pred, pos_label=1)),
        "fakeRecall": float(recall_score(y, pred, pos_label=0)),
        "brierScore": float(brier_score_loss(y, proba)),
        "classificationReport": classification_report(y, pred, target_names=["fake", "real"], output_dict=True),
        "evaluationProtocol": "subject_disjoint_held_out_feature_table",
        "trainingSubjects": int(train_df["subjectId"].nunique()),
        "testSubjects": int(df["subjectId"].nunique()),
        "testSamples": int(len(df)),
        "testClassCounts": {"fake": int((y == 0).sum()), "real": int((y == 1).sum())},
    }
    logger.info("Evaluation metrics:\n%s", json.dumps(metrics, indent=2))

    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["fake", "real"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["fake", "real"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_title("Confusion matrix")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    with open(out_dir / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved results to %s", out_dir)


if __name__ == "__main__":
    main()
