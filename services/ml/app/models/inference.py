"""Single entry point the pipeline calls for a prediction, hiding which
concrete model backend (heuristic / sklearn / deep) is active."""
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


def predict(feature_vector: np.ndarray) -> PredictionResult:
    return get_classifier().predict(feature_vector)
