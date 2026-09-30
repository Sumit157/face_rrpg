"""rPPG extraction, feature computation and signal quality for every
requested region, given the raw RGB traces from pipeline.extraction."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

from app.rppg.preprocessing import extract_rppg
from app.features.hr_features import analyse_region, RegionHRResult
from app.features.consistency import cross_region_consistency, ConsistencyResult
from app.features.motion import score_motion, MotionResult
from app.features.illumination import score_illumination, IlluminationResult
from app.features.harmonics import harmonic_ratio
from app.features.phase_coherence import phase_coherence
from app.features.amplitude_consistency import amplitude_cv
from app.core.logging import get_logger

logger = get_logger(__name__)

MIN_SAMPLES_FOR_ANALYSIS = 30  # ~1s at 30fps; below this a region is unusable


@dataclass
class RegionAnalysis:
    region_results: Dict[str, RegionHRResult]
    region_signals: Dict[str, np.ndarray]
    consistency: ConsistencyResult
    motion: MotionResult
    illumination: IlluminationResult
    usable_regions: List[str]
    harmonic_ratio: float = 0.0
    phase_coherence: float = 0.0
    amplitude_cv: float = 0.0


def analyse_regions(
    region_rgb: Dict[str, np.ndarray],
    fps: float,
    method: str,
    window_seconds: float,
    step_seconds: float,
    face_centroids: List[np.ndarray],
) -> RegionAnalysis:
    region_results: Dict[str, RegionHRResult] = {}
    region_signals: Dict[str, np.ndarray] = {}
    usable_regions: List[str] = []

    illumination_traces = []

    for region, rgb in region_rgb.items():
        if len(rgb) < MIN_SAMPLES_FOR_ANALYSIS:
            logger.warning("Region '%s' has too few samples (%d); skipping", region, len(rgb))
            continue
        signal = extract_rppg(rgb, fps, method=method)
        region_signals[region] = signal
        region_results[region] = analyse_region(region, signal, fps, window_seconds, step_seconds)
        usable_regions.append(region)
        illumination_traces.append(rgb)

    region_peak_hz = {
        r: res.full_spectrum.peak_hz for r, res in region_results.items()
    }
    consistency = cross_region_consistency(region_signals, region_peak_hz)
    motion = score_motion(face_centroids)

    if illumination_traces:
        stacked = np.concatenate(illumination_traces, axis=0)
        illumination = score_illumination(stacked)
    else:
        illumination = IlluminationResult(0.0, 0.0, 0.0)

    # Novel features adapted from the rPPG deepfake detector notebook
    # (Kate & Shelar, MIT WPU): harmonic structure and cross-region phase
    # stability, computed on the full-clip signals rather than per-window.
    harm_ratio = 0.0
    coherence = 0.0
    amp_cv = 0.0
    if region_signals:
        primary_region = next(iter(region_signals))
        primary_result = region_results[primary_region]
        harm_ratio = harmonic_ratio(
            primary_result.full_spectrum.freqs_hz,
            primary_result.full_spectrum.power,
            primary_result.full_spectrum.peak_hz,
        )
        coherence = phase_coherence(region_signals)
        amp_cv = amplitude_cv(region_signals, fps)

    return RegionAnalysis(
        region_results=region_results,
        region_signals=region_signals,
        consistency=consistency,
        motion=motion,
        illumination=illumination,
        usable_regions=usable_regions,
        harmonic_ratio=harm_ratio,
        phase_coherence=coherence,
        amplitude_cv=amp_cv,
    )
