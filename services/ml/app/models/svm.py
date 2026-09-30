"""SVM training wrapper (secondary classical baseline for comparison)."""
from __future__ import annotations
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def train_svm(X: np.ndarray, y: np.ndarray, C: float = 1.0, kernel: str = "rbf",
              random_state: int = 42) -> Pipeline:
    """Wrapped in a Pipeline so the scaler travels with the model at
    inference time -- SVMs are sensitive to feature scale and our feature
    vector mixes very different units (Hz, correlations, 0-100 scores)."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(C=C, kernel=kernel, probability=True, class_weight="balanced",
                     random_state=random_state)),
    ])
    pipeline.fit(X, y)
    return pipeline
