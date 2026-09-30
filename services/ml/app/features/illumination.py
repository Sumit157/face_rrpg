"""Illumination stability scoring from raw region brightness over time."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class IlluminationResult:
    mean_brightness: float
    brightness_std: float
    stability_score: float  # 0-100, higher = more stable lighting

    def to_dict(self) -> dict:
        return {
            "meanBrightness": round(self.mean_brightness, 2),
            "brightnessStd": round(self.brightness_std, 2),
            "stabilityScore": round(self.stability_score, 1),
        }


def score_illumination(rgb_trace: np.ndarray) -> IlluminationResult:
    """rgb_trace: (T, 3). Overall brightness = mean of channel means."""
    if len(rgb_trace) == 0:
        return IlluminationResult(0.0, 0.0, 0.0)
    brightness = rgb_trace.mean(axis=1)
    mean_b = float(brightness.mean())
    std_b = float(brightness.std())
    cv = std_b / mean_b if mean_b > 0 else 1.0
    # Lower coefficient of variation = more stable lighting.
    stability = float(np.clip(100 * (1 - min(cv / 0.15, 1.0)), 0, 100))
    return IlluminationResult(mean_b, std_b, stability)
