"""Assembles the final flat feature vector used by the classical ML models.

Keeping this in one place means the training scripts and the live
inference pipeline can never silently drift out of sync on feature
ordering.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np

FEATURE_NAMES: List[str] = [
    "bpm_mean",
    "bpm_std",
    "bpm_cv",
    "bpm_max_jump",
    "signal_quality_mean",
    "snr_db_mean",
    "cross_region_mean_correlation",
    "cross_region_frequency_agreement",
    "motion_score",
    "illumination_stability",
    "region_bpm_spread",  # max-min of per-region mean BPM: disagreement across regions
    "harmonic_ratio",       # 2f/3f cardiac harmonic energy vs peak (real pulses aren't a pure tone)
    "phase_coherence",      # cross-region phase stability via Hilbert transform, 0-1
    "amplitude_cv",         # coefficient of variation of peak amplitude across regions
]


@dataclass
class FeatureVector:
    values: Dict[str, float] = field(default_factory=dict)

    def as_array(self) -> np.ndarray:
        return np.array([self.values.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float64)

    def to_dict(self) -> dict:
        return {k: round(v, 6) for k, v in self.values.items()}


def build_feature_vector(
    region_results: Dict[str, "object"],  # RegionHRResult, typed loosely to avoid circular import
    consistency,  # ConsistencyResult
    motion,  # MotionResult
    illumination,  # IlluminationResult
    harmonic_ratio_value: float = 0.0,
    phase_coherence_value: float = 0.0,
    amplitude_cv_value: float = 0.0,
) -> FeatureVector:
    bpm_means = [r.temporal.bpm_mean for r in region_results.values() if r.temporal.bpm_mean > 0]
    bpm_stds = [r.temporal.bpm_std for r in region_results.values()]
    bpm_cvs = [r.temporal.bpm_cv for r in region_results.values()]
    bpm_jumps = [r.temporal.bpm_max_jump for r in region_results.values()]
    qualities = [r.mean_quality_score for r in region_results.values()]

    region_bpm_spread = (max(bpm_means) - min(bpm_means)) if len(bpm_means) >= 2 else 0.0

    values = {
        "bpm_mean": float(np.mean(bpm_means)) if bpm_means else 0.0,
        "bpm_std": float(np.mean(bpm_stds)) if bpm_stds else 0.0,
        "bpm_cv": float(np.mean(bpm_cvs)) if bpm_cvs else 0.0,
        "bpm_max_jump": float(np.mean(bpm_jumps)) if bpm_jumps else 0.0,
        "signal_quality_mean": float(np.mean(qualities)) if qualities else 0.0,
        "snr_db_mean": float(np.mean(
            [q.snr_db for r in region_results.values() for q in r.window_quality]
        )) if region_results else 0.0,
        "cross_region_mean_correlation": consistency.mean_correlation,
        "cross_region_frequency_agreement": consistency.frequency_agreement,
        "motion_score": motion.motion_score,
        "illumination_stability": illumination.stability_score,
        "region_bpm_spread": region_bpm_spread,
        "harmonic_ratio": harmonic_ratio_value,
        "phase_coherence": phase_coherence_value,
        "amplitude_cv": amplitude_cv_value,
    }
    return FeatureVector(values=values)
