"""Frequency-domain features that survive video compression.

Compression mostly destroys amplitude detail and adds broadband noise;
the *shape* of the spectrum inside the physiological band is far more
robust. Two complementary measures:
- band_power_ratio: how much of the total spectral energy sits in the
  0.7-4 Hz heart-rate band (a real pulse concentrates energy there).
- spectral_entropy: how concentrated the band spectrum is around a
  single peak (normalised 0-1; low = sharp pulse peak, high = noise).
"""
from __future__ import annotations

import numpy as np

from app.core.config import settings


def band_power_ratio(freqs_hz: np.ndarray, power: np.ndarray) -> float:
    """Fraction of spectral power inside the physiological HR band, 0-1."""
    total = float(np.sum(power))
    if total <= 0:
        return 0.0
    band = (freqs_hz >= settings.band_low_hz) & (freqs_hz <= settings.band_high_hz)
    return float(np.clip(np.sum(power[band]) / total, 0.0, 1.0))


def spectral_entropy(freqs_hz: np.ndarray, power: np.ndarray) -> float:
    """Normalised Shannon entropy of the power inside the HR band.

    0 = all energy in one bin (idealised pulse), 1 = flat spectrum."""
    band = (freqs_hz >= settings.band_low_hz) & (freqs_hz <= settings.band_high_hz)
    p = power[band]
    if len(p) < 2:
        return 1.0
    total = float(np.sum(p))
    if total <= 0:
        return 1.0
    probs = p / total
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log(probs)))
    return float(np.clip(entropy / np.log(len(p)), 0.0, 1.0))
