"""Single entry point the pipeline calls for a model probability, hiding
which concrete model backend (heuristic / sklearn / deep) is active."""
from __future__ import annotations
import numpy as np

from app.core.config import settings
from app.models.baseline import BaselineClassifier, PredictionResult

_classifier: BaselineClassifier | None = None


def get_classifier() -> BaselineClassifier:
    global _classifier
    if _classifier is None:
        _classifier = BaselineClassifier(settings.model_path)
    return _classifier


def predict_proba(feature_vector: np.ndarray) -> float:
    """P(REAL) from the active model (or rule heuristic if untrained)."""
    return get_classifier().predict_proba(feature_vector)


def model_info() -> tuple[str, bool]:
    """(model_version, is_heuristic_fallback)."""
    clf = get_classifier()
    return clf.version, not clf.is_trained


def predict(feature_vector: np.ndarray) -> PredictionResult:
    return get_classifier().predict(feature_vector)
