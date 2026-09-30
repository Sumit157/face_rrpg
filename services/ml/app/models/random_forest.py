"""Random Forest training wrapper (Phase 8 classical ML baseline)."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def train_random_forest(X: np.ndarray, y: np.ndarray, n_estimators: int = 300,
                         max_depth: int | None = 8, random_state: int = 42) -> RandomForestClassifier:
    """y convention: 1 = REAL, 0 = FAKE (must match BaselineClassifier.predict)."""
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
    )
    clf.fit(X, y)
    return clf


def cross_validate(clf, X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict:
    """Subject-independent CV should be done via GroupKFold upstream (see
    scripts/train_baseline.py) -- this plain CV is for quick sanity checks only."""
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    return {"mean_auc": float(scores.mean()), "std_auc": float(scores.std()), "folds": scores.tolist()}
