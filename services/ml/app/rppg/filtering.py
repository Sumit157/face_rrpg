"""Detrending, normalisation and band-pass filtering for rPPG traces."""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from scipy import linalg as sla
from scipy import signal as sps

from app.core.config import settings

LAM = 300  # smoothing strength; larger = smoother baseline removed


@lru_cache(maxsize=4)
def _detrend_cholesky(n: int, lam: int):
    """Cholesky factor of the smoothness-priors system matrix (I + lam^2 D2' D2).

    The matrix depends only on the signal length, so factorising once and
    reusing it turns the original O(n^3) dense solve per call into an
    O(n^2) back-substitution -- a large win when the same length recurs
    (training over thousands of synthetic clips, or all regions of one
    video which share a sample count)."""
    d2 = np.diff(np.eye(n), n=2, axis=0)
    a = np.eye(n) + (lam ** 2) * (d2.T @ d2)
    return sla.cho_factor(a, lower=True)


def detrend(x: np.ndarray) -> np.ndarray:
    """Remove slow drift (illumination change, minor head movement) using a
    smoothness-priors detrending approach (Tarvainen et al., 2002), which
    is gentler than a simple linear detrend and standard in the rPPG
    literature. Very long clips (n > 4000) take a cheap linear-trend +
    smoothing baseline instead: the exact solver would need an n x n dense
    matrix (gigabytes at the 600s cap) for visually no benefit after the
    band-pass."""
    n = len(x)
    if n < 3:
        return x - x.mean()
    if n > 4000:
        linear = sps.detrend(x)
        window = min(n - 1, max(101, (n // 20) | 1))
        baseline = sps.savgol_filter(linear, window_length=window, polyorder=2)
        return linear - baseline
    c, low = _detrend_cholesky(n, LAM)
    baseline = sla.cho_solve((c, low), x)
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
