"""Time-domain and temporal-stability features across windows."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class TemporalFeatures:
    bpm_mean: float
    bpm_std: float
    bpm_cv: float  # coefficient of variation: std/mean, unitless stability measure
    bpm_max_jump: float  # largest bpm change between consecutive windows

    def to_dict(self) -> dict:
        return {
            "bpmMean": round(self.bpm_mean, 2),
            "bpmStd": round(self.bpm_std, 2),
            "bpmCv": round(self.bpm_cv, 4),
            "bpmMaxJump": round(self.bpm_max_jump, 2),
        }


def temporal_stability(bpm_series: List[float]) -> TemporalFeatures:
    arr = np.array([b for b in bpm_series if b > 0], dtype=np.float64)
    if len(arr) == 0:
        return TemporalFeatures(0.0, 0.0, 0.0, 0.0)
    mean = float(arr.mean())
    std = float(arr.std())
    cv = std / mean if mean > 0 else 0.0
    jumps = np.abs(np.diff(arr)) if len(arr) > 1 else np.array([0.0])
    return TemporalFeatures(
        bpm_mean=mean, bpm_std=std, bpm_cv=cv, bpm_max_jump=float(jumps.max())
    )
