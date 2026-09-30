"""Harmonic ratio: checks whether energy also appears at 2x and 3x the
dominant cardiac frequency, as a real pulse waveform does (it isn't a
pure sinusoid). Adapted from the rPPG deepfake detector notebook
(Kate & Shelar, MIT WPU) -- a synthesised/interpolated signal is more
likely to look like a clean single tone with no harmonic structure.

CAVEAT (found while integrating this into the tested pipeline): on pure
noise with no true dominant peak, this ratio can come out *high* rather
than low, because "peak energy" is itself just a noise spike and
"energy near 2x/3x that spike" is comparably noisy -- i.e. the metric is
only meaningful when there IS a real dominant peak. Guard its use with
signal_quality / dominance, not on its own (see
app/explainability/evidence.py, which only surfaces harmonic-based
evidence once signal quality clears QUALITY_MARGINAL).
"""
from __future__ import annotations
import numpy as np


def harmonic_ratio(freqs_hz: np.ndarray, power: np.ndarray, peak_hz: float,
                    tolerance_hz: float = 0.2) -> float:
    """power/freqs_hz should be the *full* spectrum (not band-limited),
    since 2x/3x the peak can fall outside the primary cardiac band."""
    eps = 1e-10

    def energy_near(target_hz: float) -> float:
        idx = np.where(np.abs(freqs_hz - target_hz) < tolerance_hz)[0]
        return float(np.max(power[idx])) if len(idx) else 0.0

    peak_energy = energy_near(peak_hz)
    if peak_energy < eps:
        return 0.0
    return (energy_near(2 * peak_hz) + energy_near(3 * peak_hz)) / (peak_energy + eps)
