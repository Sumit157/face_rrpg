"""Per-window quality gating.

Sliding windows whose spectrum shows no usable pulse (floor SNR, tiny
band ratio, or an implausible peak) are REJECTED rather than averaged
into the clip-level features. This is the single most important fix for
indecisive verdicts: garbage windows used to drag every aggregate toward
"no signal", which both real and fake videos then scored equally on.

Rejections are counted and exposed as a GatingReport so the UI and
evidence layer can say *how much* of the clip survived -- the kept ratio
is also a feature and a key input to the decision's measurement-strength
calculation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from app.features.hr_features import RegionHRResult

# Compression and phone footage frequently sits below the older -6 dB
# cutoff. Keep borderline windows with a small reliability weight.
# Full weight is reached at 0 dB because that is the practical ceiling
# of this pipeline's Welch-based SNR measurement; the old +3 dB target
# gave every kept window on real footage a weight near 0.35.
SNR_FLOOR_DB = -8.0
SNR_FULL_WEIGHT_DB = 0.0
BAND_RATIO_MIN = 0.02
BPM_MIN = 40.0
BPM_MAX = 180.0


@dataclass
class GatingReport:
    total_windows: int = 0
    kept_windows: int = 0
    kept_snr: List[float] = field(default_factory=list)      # SNR of kept windows
    kept_weights: List[float] = field(default_factory=list)  # soft reliability of kept windows
    kept_bpm: List[float] = field(default_factory=list)      # BPM of kept windows (all regions)
    region_kept_bpm: Dict[str, List[float]] = field(default_factory=dict)
    rejected_reasons: Dict[str, int] = field(default_factory=dict)
    # region -> list of per-window kept flags (index-aligned with the
    # region's window list), for the Pipeline tab's window segmentation.
    window_kept: Dict[str, List[bool]] = field(default_factory=dict)

    @property
    def kept_ratio(self) -> float:
        return self.kept_windows / self.total_windows if self.total_windows else 0.0

    @property
    def kept_snr_mean(self) -> float:
        return float(np.mean(self.kept_snr)) if self.kept_snr else -20.0

    @property
    def effective_kept_ratio(self) -> float:
        """Quality-weighted fraction of usable windows.

        A borderline -7 dB window remains useful evidence, but contributes
        much less than a clean +3 dB measurement.
        """
        if not self.total_windows:
            return 0.0
        return float(np.sum(self.kept_weights) / self.total_windows)

    def _reject(self, reason: str) -> None:
        self.rejected_reasons[reason] = self.rejected_reasons.get(reason, 0) + 1

    def to_dict(self) -> dict:
        return {
            "totalWindows": self.total_windows,
            "keptWindows": self.kept_windows,
            "keptRatio": round(self.kept_ratio, 4),
            "effectiveKeptRatio": round(self.effective_kept_ratio, 4),
            "keptSnrMeanDb": round(self.kept_snr_mean, 2),
            "rejectedReasons": dict(self.rejected_reasons),
        }


def gate_windows(region_results: Dict[str, RegionHRResult]) -> GatingReport:
    """Evaluate every (region, window) against the quality floor."""
    report = GatingReport()
    for region, res in region_results.items():
        kept_flags: List[bool] = []
        kept_bpm: List[float] = []
        for i, quality in enumerate(res.window_quality):
            bpm = res.window_bpm[i] if i < len(res.window_bpm) else 0.0
            report.total_windows += 1
            reason = None
            if quality.snr_db < SNR_FLOOR_DB:
                reason = "snr"
            elif quality.band_ratio < BAND_RATIO_MIN:
                reason = "band_ratio"
            elif bpm and not (BPM_MIN <= bpm <= BPM_MAX):
                reason = "bpm_implausible"
            if reason is None:
                kept_flags.append(True)
                report.kept_windows += 1
                report.kept_snr.append(quality.snr_db)
                report.kept_weights.append(float(np.clip(
                    (quality.snr_db - SNR_FLOOR_DB) /
                    (SNR_FULL_WEIGHT_DB - SNR_FLOOR_DB),
                    0.10,
                    1.0,
                )))
                report.kept_bpm.append(bpm)
                kept_bpm.append(bpm)
            else:
                kept_flags.append(False)
                report._reject(reason)
        report.window_kept[region] = kept_flags
        report.region_kept_bpm[region] = kept_bpm
    return report


def gated_temporal(bpm_series: List[float]) -> Tuple[float, float, float, float]:
    """(mean, std, cv, max_jump) over kept-window BPMs; zeros when empty."""
    arr = np.array([b for b in bpm_series if b > 0], dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = float(arr.mean())
    std = float(arr.std())
    cv = std / mean if mean > 0 else 0.0
    jumps = np.abs(np.diff(arr)) if len(arr) > 1 else np.array([0.0])
    return mean, std, cv, float(jumps.max())
