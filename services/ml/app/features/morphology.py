"""Waveform-morphology features of the extracted pulse trace.

Real pulsatile blood-volume waves have a characteristic shape (sharp
systolic upstroke, slower decay, optional dicrotic notch) that repeats
with small natural variation. Synthesised or reconstructed signals tend
either to be shapeless noise (template correlation ~0) or an unnaturally
perfect oscillator (near-identical cycles, zero skewness). Both extremes
are informative; neither is used on its own.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from app.features.frequency_domain import estimate_bpm

MIN_CYCLES = 4
TEMPLATE_POINTS = 32
MAX_CYCLES = 40


def pulse_template_correlation(signal: np.ndarray, fps: float) -> float:
    """Mean correlation between individual pulse cycles and their median
    template, in [0, 1]. Returns 0.5 (neutral) when the dominant peak is
    unusable or too few cycles exist to judge."""
    n = len(signal)
    if n < int(fps * 6) or signal.std() < 1e-8:
        return 0.5
    spectrum = estimate_bpm(signal, fps)
    if spectrum.peak_hz <= 0:
        return 0.5
    period = fps / spectrum.peak_hz
    if period < 4 or period * MIN_CYCLES > n:
        return 0.5

    # Cut the signal into cycles using upward crossings of the Hilbert
    # phase so the recording's phase offset doesn't bias the segmentation.
    try:
        from scipy.signal import hilbert

        analytic = hilbert(signal)
    except Exception:  # noqa: BLE001
        return 0.5
    phase = np.angle(analytic)
    unwrapped = np.unwrap(phase)
    # Cycle boundaries = where the unwrapped phase enters a new 2*pi turn.
    cycle_turn = np.floor(unwrapped / (2.0 * np.pi)).astype(np.int64)
    starts = np.where(np.diff(cycle_turn) != 0)[0] + 1
    if len(starts) < MIN_CYCLES + 1:
        return 0.5

    cycles = []
    for a, b in zip(starts[:-1], starts[1:]):
        seg = signal[a:b]
        if len(seg) < 8:
            continue
        resampled = np.interp(
            np.linspace(0, 1, TEMPLATE_POINTS), np.linspace(0, 1, len(seg)), seg
        )
        std = resampled.std()
        if std < 1e-8:
            continue
        cycles.append((resampled - resampled.mean()) / std)
        if len(cycles) >= MAX_CYCLES:
            break
    if len(cycles) < MIN_CYCLES:
        return 0.5
    cycles_arr = np.stack(cycles)
    template = np.median(cycles_arr, axis=0)
    t_std = template.std()
    if t_std < 1e-8:
        return 0.0
    template = (template - template.mean()) / t_std
    corrs = [float(np.mean(cycle * template)) for cycle in cycles_arr]
    return float(np.clip(np.mean(corrs), 0.0, 1.0))


def pulse_skewness(signal: np.ndarray) -> float:
    """Skewness of the band-passed pulse waveform. A pulse-like wave
    (sharp peaks, long troughs) is right-skewed; measurement noise is
    symmetric (~0). Returned unclipped (the classifier sees the raw value)."""
    if len(signal) < 8 or signal.std() < 1e-8:
        return 0.0
    return float(stats.skew(signal))
