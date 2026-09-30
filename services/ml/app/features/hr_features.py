"""Convenience aggregation of heart-rate-related features for one region."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

from app.features.frequency_domain import estimate_bpm, SpectrumResult
from app.features.time_domain import temporal_stability, TemporalFeatures
from app.rppg.windows import make_windows
from app.rppg.quality import assess_quality, SignalQuality


@dataclass
class RegionHRResult:
    region: str
    window_bpm: List[float]
    window_quality: List[SignalQuality]
    temporal: TemporalFeatures
    full_spectrum: SpectrumResult
    mean_quality_score: float


def analyse_region(region: str, signal: np.ndarray, fps: float,
                    window_seconds: float, step_seconds: float) -> RegionHRResult:
    windows = make_windows(signal, fps, window_seconds, step_seconds)
    window_bpm: List[float] = []
    window_quality: List[SignalQuality] = []

    for w in windows:
        spectrum = estimate_bpm(w.data, fps)
        quality = assess_quality(spectrum.freqs_hz, spectrum.power, spectrum.peak_hz)
        window_bpm.append(spectrum.bpm)
        window_quality.append(quality)

    temporal = temporal_stability(window_bpm)
    full_spectrum = estimate_bpm(signal, fps)
    mean_quality = float(np.mean([q.quality_score for q in window_quality])) if window_quality else 0.0

    return RegionHRResult(
        region=region,
        window_bpm=window_bpm,
        window_quality=window_quality,
        temporal=temporal,
        full_spectrum=full_spectrum,
        mean_quality_score=mean_quality,
    )
