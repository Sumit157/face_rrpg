"""rPPG extraction, feature computation and signal quality for every
requested region, given the raw RGB traces from pipeline.extraction.

Pipeline stage: signal extraction -> quality gating -> feature extras.
Runs GREEN/CHROM/POS in parallel when method == "auto", gates per-window
quality, and gathers the morphology/spectral extras the decision layer
needs."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np

from app.features.hr_features import RegionHRResult
from app.features.frequency_domain import estimate_bpm
from app.features.consistency import cross_region_consistency, ConsistencyResult
from app.features.motion import score_motion, MotionResult
from app.features.illumination import score_illumination, IlluminationResult
from app.features.harmonics import harmonic_ratio
from app.features.phase_coherence import phase_coherence
from app.features.amplitude_consistency import amplitude_cv
from app.features.morphology import pulse_template_correlation, pulse_skewness
from app.features.spectral_features import band_power_ratio, spectral_entropy
from app.pipeline.quality_gating import GatingReport, gate_windows, gated_temporal
from app.rppg.multimethod import MethodSelection, analyse_methods, extract_method_signals
from app.core.logging import get_logger

logger = get_logger(__name__)

MIN_SAMPLES_FOR_ANALYSIS = 30  # ~1s at 30fps; below this a region is unusable


@dataclass
class RegionAnalysis:
    region_results: Dict[str, RegionHRResult]
    region_signals: Dict[str, np.ndarray]  # chosen-method signals, region -> (T,)
    consistency: ConsistencyResult
    motion: MotionResult
    illumination: IlluminationResult
    usable_regions: List[str]
    harmonic_ratio: float = 0.0
    phase_coherence: float = 0.0
    amplitude_cv: float = 0.0
    # --- pipeline overhaul additions ---
    method_selection: MethodSelection = field(default_factory=lambda: MethodSelection(selected="pos"))
    gating: GatingReport = field(default_factory=GatingReport)
    extras: Dict[str, float] = field(default_factory=dict)


def analyse_regions(
    region_rgb: Dict[str, np.ndarray],
    fps: float,
    method: str,
    window_seconds: float,
    step_seconds: float,
    face_centroids: List[np.ndarray],
    flow_magnitudes: List[float] | None = None,
) -> RegionAnalysis:
    # 1. Signal extraction: every requested method, every usable region.
    region_signals_by_method: Dict[str, Dict[str, np.ndarray]] = {}
    illumination_traces = []
    for region, rgb in region_rgb.items():
        if len(rgb) < MIN_SAMPLES_FOR_ANALYSIS:
            logger.warning("Region '%s' has too few samples (%d); skipping", region, len(rgb))
            continue
        region_signals_by_method[region] = extract_method_signals(rgb, fps, method)
        illumination_traces.append(rgb)

    # 2. Multi-method analysis + selection (pass-through when method != auto).
    region_results, selection = analyse_methods(
        region_signals_by_method, fps, window_seconds, step_seconds, method
    )
    region_signals = {
        region: signals[selection.selected]
        for region, signals in region_signals_by_method.items()
        if selection.selected in signals and region in region_results
    }
    usable_regions = [r for r in region_signals_by_method if r in region_results]

    region_peak_hz = {r: res.full_spectrum.peak_hz for r, res in region_results.items()}
    consistency = cross_region_consistency(region_signals, region_peak_hz)
    motion = score_motion(face_centroids, flow_magnitudes)

    if illumination_traces:
        stacked = np.concatenate(illumination_traces, axis=0)
        illumination = score_illumination(stacked)
    else:
        illumination = IlluminationResult(0.0, 0.0, 0.0)

    # 3. Quality gating: drop windows that fail the SNR/band-ratio floor.
    gating = gate_windows(region_results)

    # 4. Clip-level extras (harmonic/phase/amplitude + new morphology and
    #    spectral features) computed on the chosen method's signals.
    extras: Dict[str, float] = {}
    if region_signals:
        primary_region = next(iter(region_signals))
        primary_result = region_results[primary_region]
        extras["harmonic_ratio"] = harmonic_ratio(
            primary_result.full_spectrum.freqs_hz,
            primary_result.full_spectrum.power,
            primary_result.full_spectrum.peak_hz,
        )
        extras["phase_coherence"] = phase_coherence(region_signals)
        extras["amplitude_cv"] = amplitude_cv(region_signals, fps)

        band_ratios, entropies, templates, skews = [], [], [], []
        for region_name, sig in region_signals.items():
            if region_name == primary_region:
                spec = primary_result.full_spectrum
            else:
                spec = estimate_bpm(sig, fps)
            band_ratios.append(band_power_ratio(spec.freqs_hz, spec.power))
            entropies.append(spectral_entropy(spec.freqs_hz, spec.power))
            templates.append(pulse_template_correlation(sig, fps))
            skews.append(pulse_skewness(sig))
        extras["pulse_band_ratio"] = float(np.mean(band_ratios))
        extras["spectral_entropy"] = float(np.mean(entropies))
        extras["template_correlation"] = float(np.mean(templates))
        extras["pulse_skewness"] = float(np.mean(skews))

        # Left vs right cheek bilateral correlation (key deepfake cue).
        if "left_cheek" in region_signals and "right_cheek" in region_signals:
            a = region_signals["left_cheek"]
            b = region_signals["right_cheek"]
            n = min(len(a), len(b))
            if n > 2:
                lr = float(np.corrcoef(a[:n], b[:n])[0, 1])
                extras["lr_cheek_correlation"] = 0.0 if np.isnan(lr) else lr

    extras["kept_window_ratio"] = gating.kept_ratio
    extras["effective_kept_window_ratio"] = gating.effective_kept_ratio
    extras["gated_bpm_mean"], extras["gated_bpm_std"], extras["gated_bpm_cv"], \
        extras["gated_bpm_max_jump"] = gated_temporal(gating.kept_bpm)

    logger.info(
        "Analysis: method=%s (candidates: %s), windows kept %d/%d (%.0f%%), "
        "kept SNR %.1f dB",
        selection.selected, selection.summary(), gating.kept_windows,
        gating.total_windows, gating.kept_ratio * 100, gating.kept_snr_mean,
    )

    return RegionAnalysis(
        region_results=region_results,
        region_signals=region_signals,
        consistency=consistency,
        motion=motion,
        illumination=illumination,
        usable_regions=usable_regions,
        harmonic_ratio=extras.get("harmonic_ratio", 0.0),
        phase_coherence=extras.get("phase_coherence", 0.0),
        amplitude_cv=extras.get("amplitude_cv", 0.0),
        method_selection=selection,
        gating=gating,
        extras=extras,
    )
