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

    def summary(self) -> str:
        ranked = sorted(self.mean_quality.items(), key=lambda kv: -kv[1])
        parts = [f"{m}={q:.1f}" for m, q in ranked]
        return ", ".join(parts)


def extract_method_signals(
    rgb_trace: np.ndarray, fps: float, method: str
) -> Dict[str, np.ndarray]:
    """One preprocessed rPPG signal per requested method for one region."""
    methods = CANDIDATE_METHODS if method == "auto" else [method]
    out: Dict[str, np.ndarray] = {}
    for m in methods:
        try:
            out[m] = extract_rppg(rgb_trace, fps, method=m)
        except Exception:  # noqa: BLE001 -- one broken method must not kill the analysis
            logger.exception("rPPG method '%s' failed on a region; skipping it", m)
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
    selection = MethodSelection(selected=best, mean_quality=mean_quality)
    return per_method_results[best], selection
