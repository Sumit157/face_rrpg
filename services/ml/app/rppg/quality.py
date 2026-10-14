"""Signal-quality scoring.

The single most important safety property of this application: a noisy
rPPG trace must never be silently presented as strong evidence. This
module produces a 0-100 signal-quality score plus the SNR value it is
derived from, using the ratio of spectral energy inside a narrow band
around the dominant HR peak versus the rest of the physiological band.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from app.core.config import settings


@dataclass
class SignalQuality:
    snr_db: float
    quality_score: float  # 0-100
    dominant_freq_hz: float
    band_ratio: float  # fraction of band energy within +-0.1Hz of the peak

    def to_dict(self) -> dict:
        return {
            "snrDb": round(self.snr_db, 2),
            "qualityScore": round(self.quality_score, 1),
            "dominantFreqHz": round(self.dominant_freq_hz, 4),
            "bandRatio": round(self.band_ratio, 4),
        }


def compute_snr(freqs: np.ndarray, power: np.ndarray, peak_hz: float,
                 tolerance_hz: float = 0.1) -> tuple[float, float]:
    """Returns (snr_db, band_ratio) for the harmonic-aware SNR around peak_hz."""
    eps = 1e-12
    signal_band = np.abs(freqs - peak_hz) <= tolerance_hz
    # first harmonic too, since it's physiologically part of the same pulse
    harmonic_band = np.abs(freqs - 2 * peak_hz) <= tolerance_hz
    signal_mask = signal_band | harmonic_band

    signal_power = power[signal_mask].sum()
    total_power = power.sum()
    noise_power = max(total_power - signal_power, eps)

    snr_db = 10 * np.log10((signal_power + eps) / noise_power)
    band_ratio = float(signal_power / (total_power + eps))
    return float(snr_db), band_ratio


def score_from_snr(snr_db: float) -> float:
    """Map SNR (dB) to an intuitive 0-100 score via a clipped linear map.

    Calibrated to the SNR this pipeline actually measures (Welch power
    around the dominant peak vs the rest of the band): buried/noise-floor
    signals sit near -9 dB, usable real-world footage near -5..-2 dB, and
    even clean clips rarely exceed +2 dB. The old -10..+12 range made
    every real video score in the 20s and read as 'poor'."""
    lo, hi = -9.0, 2.0
    score = (snr_db - lo) / (hi - lo) * 100
    return float(np.clip(score, 0, 100))


def assess_quality(freqs: np.ndarray, power: np.ndarray, peak_hz: float) -> SignalQuality:
    snr_db, band_ratio = compute_snr(freqs, power, peak_hz)
    score = score_from_snr(snr_db)
    return SignalQuality(
        snr_db=snr_db, quality_score=score, dominant_freq_hz=peak_hz, band_ratio=band_ratio
    )
