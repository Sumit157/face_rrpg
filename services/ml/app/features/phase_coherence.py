"""Phase coherence: cross-region phase stability via the Hilbert
transform. Real blood flow reaches forehead and both cheeks at nearly
the same phase of the cardiac cycle (small, fixed pulse-transit delays);
if per-region phases drift apart over the clip, that's an unstable,
likely-non-physiological signal. Adapted from the rPPG deepfake detector
notebook (Kate & Shelar, MIT WPU) -- listed there as a novel feature
beyond the base paper's method, and it is: it's a distinct measurement
from the cross_region_consistency correlation already in
app/features/consistency.py (phase relationship over time, vs.
amplitude correlation)."""
from __future__ import annotations
from typing import Dict
import numpy as np
from scipy.signal import hilbert


def instantaneous_phase(signal: np.ndarray) -> np.ndarray:
    return np.unwrap(np.angle(hilbert(signal)))


def phase_coherence(region_signals: Dict[str, np.ndarray]) -> float:
    """Returns a 0-1 score: 1.0 = perfectly phase-locked across regions,
    lower = phases drift apart over the clip. Needs >=2 regions."""
    regions = list(region_signals.keys())
    if len(regions) < 2:
        return 1.0  # nothing to compare; don't penalise a single-region analysis

    phases = {r: instantaneous_phase(region_signals[r]) for r in regions}
    min_len = min(len(p) for p in phases.values())
    diffs = []
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            a, b = phases[regions[i]][:min_len], phases[regions[j]][:min_len]
            diffs.append(np.std(a - b))

    mean_phase_drift = float(np.mean(diffs))
    # exp(-drift / 2π): drift of 0 -> 1.0, drift of ~2π (a full cycle of
    # relative slip) -> ~0.37, matching the notebook's calibration.
    return float(np.exp(-mean_phase_drift / (2 * np.pi)))
