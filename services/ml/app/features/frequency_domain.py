"""FFT / PSD spectrum and BPM estimation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import signal as sps

from app.core.config import settings


@dataclass
class SpectrumResult:
    freqs_hz: np.ndarray
    power: np.ndarray
    peak_hz: float
    bpm: float

    def to_dict(self, include_arrays: bool = False) -> dict:
        out = {"peakHz": round(self.peak_hz, 4), "bpm": round(self.bpm, 2)}
        if include_arrays:
            out["freqsHz"] = self.freqs_hz.tolist()
            out["power"] = self.power.tolist()
        return out


def welch_psd(x: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Welch's method gives a smoother, lower-variance PSD estimate than a
    raw periodogram, which matters a lot for short (~10-15s) windows."""
    nperseg = min(len(x), max(64, int(fps * 4)))
    freqs, power = sps.welch(x, fs=fps, nperseg=nperseg)
    return freqs, power


def estimate_bpm(x: np.ndarray, fps: float) -> SpectrumResult:
    freqs, power = welch_psd(x, fps)
    band = (freqs >= settings.band_low_hz) & (freqs <= settings.band_high_hz)
    if not band.any():
        return SpectrumResult(freqs, power, peak_hz=0.0, bpm=0.0)
    band_freqs = freqs[band]
    band_power = power[band]
    peak_idx = int(np.argmax(band_power))
    peak_hz = float(band_freqs[peak_idx])
    bpm = peak_hz * 60.0
    return SpectrumResult(freqs_hz=freqs, power=power, peak_hz=peak_hz, bpm=bpm)
