"""Confidence calibration/annotation.

Confidence is deliberately reported alongside, and discounted by, signal
quality -- a model probability of 0.9 computed from a poor-quality
signal is not trustworthy at 0.9.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class CalibratedConfidence:
    raw_probability: float
    signal_quality: float
    adjusted_confidence: float

    def to_dict(self) -> dict:
        return {
            "rawProbability": round(self.raw_probability, 4),
            "signalQuality": round(self.signal_quality, 1),
            "adjustedConfidence": round(self.adjusted_confidence, 4),
        }


def calibrate(probability: float, signal_quality_score: float) -> CalibratedConfidence:
    """Pulls probability toward 0.5 in proportion to how far signal quality
    falls below a 'trustworthy' threshold (70/100). This is a transparent
    heuristic discount, not a statistically fitted calibration curve --
    real calibration (e.g. Platt scaling) should be fitted once labelled
    evaluation data exists (see docs/limitations.md)."""
    quality_factor = min(signal_quality_score / 70.0, 1.0)
    adjusted = 0.5 + (probability - 0.5) * quality_factor
    return CalibratedConfidence(
        raw_probability=probability,
        signal_quality=signal_quality_score,
        adjusted_confidence=adjusted,
    )
