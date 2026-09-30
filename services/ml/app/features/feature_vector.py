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
    # --- v2 features (pipeline overhaul) ---
    "pulse_band_ratio",     # spectral power inside 0.7-4 Hz / total (compression-robust)
    "spectral_entropy",     # normalised entropy of the band spectrum; low = sharp pulse peak
    "lr_cheek_correlation", # left cheek vs right cheek pulse-band correlation (key bilateral cue)
    "template_correlation", # cycle-to-template waveform repeatability, 0-1
    "pulse_skewness",       # waveform asymmetry of the band-passed trace
    "kept_window_ratio",    # fraction of windows that passed quality gating
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
    extras: Dict[str, float] | None = None,  # morphology/spectral/quality-gating scalars
    gated_bpm: List[float] | None = None,  # kept-window BPMs across regions (quality gate)
    kept_snr: List[float] | None = None,  # kept-window SNR values
    region_kept_bpm: Dict[str, List[float]] | None = None,  # per-region kept BPMs
) -> FeatureVector:
    """Build the flat feature vector.

    Temporal aggregates use ONLY quality-gated windows when gating info is
    provided (the gate's whole point is that garbage windows must not
    pollute the clip-level features); fall back to raw window series for
    callers that don't gate (tests, notebooks)."""
    extras = extras or {}
    gated = gated_bpm is not None

    if gated:
        bpm_vals = [b for b in (gated_bpm or []) if b > 0]
        mean_bpm = float(np.mean(bpm_vals)) if bpm_vals else 0.0
        std_bpm = float(np.std(bpm_vals)) if bpm_vals else 0.0
        cv_bpm = std_bpm / mean_bpm if mean_bpm > 0 else 0.0
        jumps = np.abs(np.diff(bpm_vals)) if len(bpm_vals) > 1 else np.array([0.0])
        jump_bpm = float(jumps.max()) if len(bpm_vals) else 0.0
        per_region_means = [
            float(np.mean(bl)) if (bl := [b for b in blist if b > 0]) else 0.0
            for blist in (region_kept_bpm or {}).values()
        ]
        per_region_means = [m for m in per_region_means if m > 0]
        region_bpm_spread = (
            max(per_region_means) - min(per_region_means)
            if len(per_region_means) >= 2 else 0.0
        )
    else:
        bpm_means = [r.temporal.bpm_mean for r in region_results.values() if r.temporal.bpm_mean > 0]
        bpm_stds = [r.temporal.bpm_std for r in region_results.values()]
        bpm_cvs = [r.temporal.bpm_cv for r in region_results.values()]
        bpm_jumps = [r.temporal.bpm_max_jump for r in region_results.values()]
        mean_bpm = float(np.mean(bpm_means)) if bpm_means else 0.0
        std_bpm = float(np.mean(bpm_stds)) if bpm_stds else 0.0
        cv_bpm = float(np.mean(bpm_cvs)) if bpm_cvs else 0.0
        jump_bpm = float(np.mean(bpm_jumps)) if bpm_jumps else 0.0
        region_bpm_spread = (
            max(bpm_means) - min(bpm_means) if len(bpm_means) >= 2 else 0.0
        )

    qualities = [r.mean_quality_score for r in region_results.values()]
    if kept_snr:
        snr_mean = float(np.mean(kept_snr))
    else:
        snr_vals = [q.snr_db for r in region_results.values() for q in r.window_quality]
        snr_mean = float(np.mean(snr_vals)) if snr_vals else -20.0

    values = {
        "bpm_mean": mean_bpm,
        "bpm_std": std_bpm,
        "bpm_cv": cv_bpm,
        "bpm_max_jump": jump_bpm,
        "signal_quality_mean": float(np.mean(qualities)) if qualities else 0.0,
        "snr_db_mean": snr_mean,
        "cross_region_mean_correlation": consistency.mean_correlation,
        "cross_region_frequency_agreement": consistency.frequency_agreement,
        "motion_score": motion.motion_score,
        "illumination_stability": illumination.stability_score,
        "region_bpm_spread": region_bpm_spread,
        "harmonic_ratio": extras.get("harmonic_ratio", 0.0),
        "phase_coherence": extras.get("phase_coherence", 0.0),
        "amplitude_cv": extras.get("amplitude_cv", 0.0),
        "pulse_band_ratio": extras.get("pulse_band_ratio", 0.0),
        "spectral_entropy": extras.get("spectral_entropy", 1.0),
        "lr_cheek_correlation": extras.get("lr_cheek_correlation", 0.0),
        "template_correlation": extras.get("template_correlation", 0.5),
        "pulse_skewness": extras.get("pulse_skewness", 0.0),
        "kept_window_ratio": extras.get("kept_window_ratio", 0.0),
        # Kept outside FEATURE_NAMES until the deployed model is retrained;
        # the decision fusion consumes it immediately without schema drift.
        "effective_kept_window_ratio": extras.get("effective_kept_window_ratio", 0.0),
    }
    return FeatureVector(values=values)
