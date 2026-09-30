"""Classical ML baseline: trains/loads a scikit-learn classifier over the
feature vector defined in app.features.feature_vector.

Ships with a small, clearly-labelled *heuristic* fallback model so the
API and frontend are fully exercisable before any real dataset has been
trained on -- this fallback must never be confused with a validated
classifier. See docs/limitations.md.
"""
from __future__ import annotations
from dataclasses import dataclass
import os
import numpy as np

from app.features.feature_vector import FEATURE_NAMES
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    label: str  # "REAL" | "FAKE" | "UNCERTAIN"
    probability_real: float
    probability_fake: float
    model_version: str
    is_heuristic_fallback: bool

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "probabilityReal": round(self.probability_real, 4),
            "probabilityFake": round(self.probability_fake, 4),
            "modelVersion": self.model_version,
            "isHeuristicFallback": self.is_heuristic_fallback,
        }


class BaselineClassifier:
    """Loads a trained sklearn model if present, otherwise falls back to a
    transparent, documented heuristic so the rest of the system is testable."""

    UNCERTAIN_MARGIN = 0.12  # |p_real - 0.5| below this -> UNCERTAIN

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None
        self._version = "untrained-heuristic-v0"
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.model_path):
            try:
                import joblib
                self._model = joblib.load(self.model_path)
                self._version = os.path.basename(self.model_path)
                logger.info("Loaded trained model from %s", self.model_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load model at %s: %s", self.model_path, exc)
        else:
            logger.warning(
                "No trained model found at %s -- using heuristic fallback. "
                "Train one with scripts/train_baseline.py.", self.model_path
            )

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def predict(self, feature_vector: np.ndarray) -> PredictionResult:
        if self._model is not None:
            proba = self._model.predict_proba(feature_vector.reshape(1, -1))[0]
            # Convention: class index 1 == REAL, 0 == FAKE (see train_baseline.py)
            p_real = float(proba[1])
        else:
            p_real = self._heuristic_probability(feature_vector)

        p_fake = 1.0 - p_real
        if abs(p_real - 0.5) < self.UNCERTAIN_MARGIN:
            label = "UNCERTAIN"
        else:
            label = "REAL" if p_real > 0.5 else "FAKE"

        return PredictionResult(
            label=label,
            probability_real=p_real,
            probability_fake=p_fake,
            model_version=self._version,
            is_heuristic_fallback=self._model is None,
        )

    @staticmethod
    def _heuristic_probability(feature_vector: np.ndarray) -> float:
        """A simple, fully transparent weighted combination of the features
        most supported by the literature as informative (signal quality,
        cross-region correlation/frequency agreement, BPM stability). This
        is NOT a validated classifier -- it exists only so the pipeline
        produces a labelled, explainable output before real training data
        is available."""
        fv = dict(zip(FEATURE_NAMES, feature_vector))
        quality = fv["signal_quality_mean"] / 100.0
        corr = np.clip((fv["cross_region_mean_correlation"] + 1) / 2, 0, 1)
        freq_agree = fv["cross_region_frequency_agreement"]
        stability = 1.0 - np.clip(fv["bpm_cv"], 0, 1)
        # Harmonic/phase/amplitude checks adapted from the rPPG deepfake
        # detector notebook (Kate & Shelar, MIT WPU) -- a real pulse has
        # harmonic structure, stays phase-locked across regions, and
        # scales proportionally in amplitude across regions.
        harmonics = np.clip(fv.get("harmonic_ratio", 0.0) / 0.1, 0, 1)  # ~0.1 ratio saturates the score
        coherence = np.clip(fv.get("phase_coherence", 0.0), 0, 1)
        amp_consistency = 1.0 - np.clip(fv.get("amplitude_cv", 0.0), 0, 1)

        score = (
            0.20 * quality + 0.20 * corr + 0.15 * freq_agree + 0.15 * stability
            + 0.10 * harmonics + 0.10 * coherence + 0.10 * amp_consistency
        )
        # Squash toward 0.5 when signal quality is poor -- a noisy signal
        # should push the system toward UNCERTAIN, not a confident verdict.
        score = 0.5 + (score - 0.5) * quality
        return float(np.clip(score, 0.01, 0.99))
