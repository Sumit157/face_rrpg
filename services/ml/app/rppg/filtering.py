"""Detrending, normalisation and band-pass filtering for rPPG traces."""
from __future__ import annotations
import numpy as np
from scipy import signal as sps

from app.core.config import settings


def detrend(x: np.ndarray) -> np.ndarray:
    """Remove slow drift (illumination change, minor head movement) using a
    smoothness-priors detrending approach (Tarvainen et al., 2002), which
    is gentler than a simple linear detrend and standard in the rPPG
    literature."""
    n = len(x)
    if n < 3:
        return x - np.mean(x)
    lam = 300  # smoothing strength; larger = smoother baseline removed
    identity = np.eye(n)
    d2 = np.diff(identity, n=2, axis=0)
    baseline = np.linalg.solve(
        identity + (lam ** 2) * (d2.T @ d2), x
    )
    return x - baseline


def normalise(x: np.ndarray) -> np.ndarray:
    std = x.std()
    if std < 1e-8:
        return x - x.mean()
    return (x - x.mean()) / std


def bandpass(x: np.ndarray, fps: float, low_hz: float = None, high_hz: float = None,
             order: int = 3) -> np.ndarray:
    """Butterworth band-pass restricted to the physiologically plausible
    heart-rate range so noise outside that band can't masquerade as signal."""
    low_hz = low_hz if low_hz is not None else settings.band_low_hz
    high_hz = high_hz if high_hz is not None else settings.band_high_hz
    nyq = fps / 2.0
    low = max(low_hz / nyq, 1e-4)
    high = min(high_hz / nyq, 0.999)
    if low >= high:
        return x
    b, a = sps.butter(order, [low, high], btype="band")
    padlen = min(3 * max(len(b), len(a)), len(x) - 1)
    if padlen <= 0:
        return x
    return sps.filtfilt(b, a, x)


def preprocess(raw_signal: np.ndarray, fps: float) -> np.ndarray:
    """Full preprocessing chain: detrend -> normalise -> band-pass."""
    x = detrend(raw_signal)
    x = normalise(x)
    x = bandpass(x, fps)
    return x
