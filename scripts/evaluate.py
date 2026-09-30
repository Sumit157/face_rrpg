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
    roc_auc_score, accuracy_score, f1_score, confusion_matrix, classification_report,
)

from app.features.feature_vector import FEATURE_NAMES
from app.core.logging import get_logger

logger = get_logger("evaluate")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/production/baseline.joblib")
    parser.add_argument("--features", required=True)
    parser.add_argument("--output-dir", default="experiments/results")
    args = parser.parse_args()

    model = joblib.load(args.model)
    df = pd.read_csv(args.features)
    X = df[FEATURE_NAMES].values
    y = (df["label"] == "real").astype(int).values

    proba = model.predict_proba(X)[:, 1]
    pred = (proba > 0.5).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y, proba)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "classificationReport": classification_report(y, pred, target_names=["fake", "real"], output_dict=True),
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
