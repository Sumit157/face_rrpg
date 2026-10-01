"""Classical ML baseline: loads a scikit-learn classifier over the
feature vector defined in app.features.feature_vector.

Label DECISIONS are no longer made here -- app.pipeline.decision fuses
this model's probability with transparent rule cues and quality gating.
This module only answers "P(REAL) = ?" (trained model when present, a
documented heuristic otherwise).
"""
from __future__ import annotations
from dataclasses import dataclass
import os
import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    label: str  # "REAL" | "FAKE" | "UNCERTAIN" (set by app.pipeline.decision)
    probability_real: float
    probability_fake: float
    model_version: str
    is_heuristic_fallback: bool
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "probabilityReal": round(self.probability_real, 4),
            "probabilityFake": round(self.probability_fake, 4),
            "modelVersion": self.model_version,
            "isHeuristicFallback": self.is_heuristic_fallback,
            "explanation": self.explanation,
        }


class BaselineClassifier:
    """Loads a trained sklearn model if present, otherwise falls back to a
    transparent heuristic so the rest of the system stays exercisable."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None
        self._version = "heuristic-rule-fusion-v1"
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.model_path):
            try:
                import joblib
                self._model = joblib.load(self.model_path)
                self._version = getattr(
                    self._model, "model_version_", os.path.basename(self.model_path)
                )
                logger.info("Loaded trained model from %s (%s)", self.model_path, self._version)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load model at %s: %s", self.model_path, exc)
        else:
            logger.warning(
                "No trained model found at %s -- using the rule-based heuristic. "
                "Train one with scripts/train_synthetic_rf.py.", self.model_path
            )

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    @property
    def version(self) -> str:
        return self._version

    @property
    def validation_status(self) -> str:
        """Describe what the loaded artifact's training history supports.

        A trained estimator is not automatically validated on real videos;
        only a separate labeled, subject-disjoint test can establish that.
        """
        if not self.is_trained:
            return "heuristic_unvalidated"
        if "synthetic" in self._version.lower():
            return "synthetic_unvalidated"
        if getattr(self._model, "external_validation_passed_", False):
            return "externally_validated"
        return "trained_unvalidated"

    def predict_proba(self, feature_vector: np.ndarray) -> float:
        """P(REAL) in [0, 1] from the trained model, or the heuristic when
        no model file exists. Does NOT apply any uncertainty margin -- see
        app.pipeline.decision."""
        if self._model is not None:
            proba = self._model.predict_proba(feature_vector.reshape(1, -1))[0]
            # Convention: class index 1 == REAL, 0 == FAKE (train_synthetic_rf.py
            # and train_baseline.py both fit with y = 1 for real).
            return float(proba[1])
        return self._heuristic_probability(feature_vector)

    def predict(self, feature_vector: np.ndarray) -> PredictionResult:
        """Probability wrapped as a PredictionResult with a plain 0.5
        threshold label. Kept for scripts/notebooks; the live pipeline
        uses app.pipeline.decision for labelling (quality-aware)."""
        p_real = self.predict_proba(feature_vector)
        label = "REAL" if p_real >= 0.5 else "FAKE"
        return PredictionResult(
            label=label,
            probability_real=p_real,
            probability_fake=1.0 - p_real,
            model_version=self._version,
            is_heuristic_fallback=self._model is None,
        )

    @staticmethod
    def _heuristic_probability(feature_vector: np.ndarray) -> float:
        """Rule-based fallback P(REAL): the same transparent cue fusion the
        decision layer uses (app.features.cue_scores), so an untrained
        system still behaves consistently with a trained one."""
        from app.features.cue_scores import rule_score
        from app.features.feature_vector import FEATURE_NAMES

        fv = dict(zip(FEATURE_NAMES, feature_vector))
        score, _ = rule_score(fv)
        return float(np.clip(score, 0.01, 0.99))
