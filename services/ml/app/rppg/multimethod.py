"""Multi-method rPPG extraction: run GREEN, CHROM and POS in parallel on
every region and pick the method that measures the best pulse SNR.

Rationale (pyVHR-style "method voting"): each classical method has a
different robustness profile -- GREEN is strongest in bright scenes,
POS/CHROM cancel slow illumination change -- so on any *single* video the
best method is not knowable in advance. Running all three and selecting
per analysis (the same set of windows for every region, so cross-region
comparisons stay method-consistent) removes that guesswork instead of
hard-coding one method.

Selection criterion: mean per-window signal-quality score of the
candidate method, averaged across usable regions. The chosen method and
its competitor scores are reported so the decision stays explainable.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from app.core.logging import get_logger
from app.features.hr_features import RegionHRResult, analyse_region
from app.rppg.preprocessing import extract_rppg

logger = get_logger(__name__)

CANDIDATE_METHODS: List[str] = ["green", "chrom", "pos"]


@dataclass
class MethodSelection:
    selected: str
    mean_quality: Dict[str, float] = field(default_factory=dict)  # per candidate
    mean_bpm: Dict[str, float] = field(default_factory=dict)  # quality-gated candidate HR
    frequency_agreement: float = 0.5

    def summary(self) -> str:
        ranked = sorted(self.mean_quality.items(), key=lambda kv: -kv[1])
        parts = [
            f"{m}={q:.1f} (HR {self.mean_bpm.get(m, 0.0):.1f} bpm)"
            for m, q in ranked
        ]
        return f"{', '.join(parts)}; method frequency agreement={self.frequency_agreement:.0%}"


def extract_method_signals(
    rgb_trace: np.ndarray, fps: float, method: str
) -> Dict[str, np.ndarray]:
    """One preprocessed rPPG signal per requested method for one region."""
    methods = CANDIDATE_METHODS if method == "auto" else [method]
    out: Dict[str, np.ndarray] = {}

    def extract_one(m: str) -> tuple[str, np.ndarray | None]:
        try:
            return m, extract_rppg(rgb_trace, fps, method=m)
        except Exception:  # noqa: BLE001 -- one broken method must not kill the analysis
            logger.exception("rPPG method '%s' failed on a region; skipping it", m)
            return m, None

    # The three colour projections are independent for a given ROI. Run them
    # concurrently, while collecting in candidate order for deterministic
    # diagnostics and selecting the best method downstream.
    if len(methods) > 1:
        with ThreadPoolExecutor(max_workers=len(methods), thread_name_prefix="rppg") as pool:
            results = pool.map(extract_one, methods)
            out.update({name: signal for name, signal in results if signal is not None})
    else:
        name, signal = extract_one(methods[0])
        if signal is not None:
            out[name] = signal
    return out


def analyse_methods(
    region_signals: Dict[str, Dict[str, np.ndarray]],
    fps: float,
    window_seconds: float,
    step_seconds: float,
    requested_method: str,
) -> tuple[Dict[str, RegionHRResult], MethodSelection]:
    """Run windowed HR analysis for the chosen method.

    region_signals: region -> method -> preprocessed signal.
    Returns per-region results for ONE method (all regions consistent) plus
    the selection record. When a concrete method was requested there is
    only one candidate and selection is a pass-through.
    """
    candidates = (
        CANDIDATE_METHODS if requested_method == "auto" else [requested_method]
    )
    # Per-candidate per-region analysis -- cheap (Welch PSDs only), and it
    # gives every candidate the exact same windowing so scores compare.
    per_method_results: Dict[str, Dict[str, RegionHRResult]] = {}
    mean_quality: Dict[str, float] = {}
    mean_bpm: Dict[str, float] = {}
    for m in candidates:
        region_res: Dict[str, RegionHRResult] = {}
        scores: List[float] = []
        for region, signals in region_signals.items():
            sig = signals.get(m)
            if sig is None or len(sig) == 0:
                continue
            res = analyse_region(region, sig, fps, window_seconds, step_seconds)
            region_res[region] = res
            scores.append(res.mean_quality_score)
        per_method_results[m] = region_res
        mean_quality[m] = float(np.mean(scores)) if scores else -1.0
        usable_bpm = [
            bpm
            for result in region_res.values()
            for bpm, quality in zip(result.window_bpm, result.window_quality)
            if 40.0 <= bpm <= 180.0 and quality.snr_db >= -8.0
        ]
        mean_bpm[m] = float(np.median(usable_bpm)) if usable_bpm else 0.0

    measured_bpm = [bpm for bpm in mean_bpm.values() if bpm > 0.0]
    # Fraction of measurable method pairs that agree within 12 BPM. Two
    # methods are enough to expose agreement; absent methods remain neutral.
    agreeing_pairs = sum(
        abs(measured_bpm[i] - measured_bpm[j]) <= 12.0
        for i in range(len(measured_bpm))
        for j in range(i + 1, len(measured_bpm))
    )
    pair_count = len(measured_bpm) * (len(measured_bpm) - 1) // 2
    frequency_agreement = agreeing_pairs / pair_count if pair_count else 0.5

    best = max(mean_quality, key=lambda k: mean_quality[k])
    # Guard: if every candidate scored -1.0 (no usable region at all) keep
    # the requested/first method so downstream code still has a signal.
    if mean_quality[best] < 0.0:
        best = candidates[0]
    if best != requested_method:
        logger.info(
            "Auto-selected rPPG method '%s' (mean window quality: %s)",
            best, ", ".join(f"{m}={q:.1f}" for m, q in sorted(mean_quality.items(), key=lambda kv: -kv[1])),
        )
    selection = MethodSelection(
        selected=best,
        mean_quality=mean_quality,
        mean_bpm=mean_bpm,
        frequency_agreement=float(frequency_agreement),
    )
    return per_method_results[best], selection
