"""Amplitude coefficient of variation across regions: real blood-volume
pulses scale roughly proportionally across forehead/cheeks (perfusion
differs by region but changes together), so their peak spectral
amplitudes should have low relative spread. Adapted from the rPPG
deepfake detector notebook (Kate & Shelar, MIT WPU)."""
from __future__ import annotations
from typing import Dict
import numpy as np

from app.features.frequency_domain import welch_psd
from app.core.config import settings


def amplitude_cv(region_signals: Dict[str, np.ndarray], fps: float) -> float:
    """Coefficient of variation (std/mean) of each region's peak spectral
    amplitude within the cardiac band. Lower = more consistent = more
    physiologically plausible."""
    amps = []
    for signal in region_signals.values():
        freqs, power = welch_psd(signal, fps)
        band = (freqs >= settings.band_low_hz) & (freqs <= settings.band_high_hz)
        if not band.any():
            continue
        amps.append(float(power[band].max()))

    if len(amps) < 2:
        return 0.0
    mean_amp = np.mean(amps)
    if mean_amp < 1e-10:
        return 0.0
    return float(np.std(amps) / mean_amp)
