"""POS method (Wang et al., 2017, "Algorithmic Principles of Remote PPG").

Projects RGB onto the plane orthogonal to the skin-tone (illumination)
vector in a temporally-normalised RGB space, which tends to generalise
better across skin tones and lighting than CHROM.
"""
from __future__ import annotations
import numpy as np


def pos_signal(rgb_trace: np.ndarray, fps: float, window_seconds: float = 1.6) -> np.ndarray:
    """rgb_trace: shape (T, 3) with columns [R, G, B]. Returns shape (T,).

    POS is defined over a sliding temporal window (~1.6s per the original
    paper) whose outputs are overlap-added into the full-length signal.
    """
    if rgb_trace.ndim != 2 or rgb_trace.shape[1] != 3:
        raise ValueError("rgb_trace must have shape (T, 3)")

    n = rgb_trace.shape[0]
    win = max(3, int(round(window_seconds * fps)))
    signal = np.zeros(n, dtype=np.float64)
    eps = 1e-8

    projection = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float64)

    for start in range(0, n - win + 1):
        end = start + win
        segment = rgb_trace[start:end]  # (win, 3)
        mean = segment.mean(axis=0) + eps
        normed = segment / mean  # temporal normalisation (Cn in the paper)

        s = normed @ projection.T  # (win, 2) -> [S1, S2]
        s1, s2 = s[:, 0], s[:, 1]
        alpha = s1.std() / (s2.std() + eps)
        h = s1 + alpha * s2

        h = h - h.mean()
        signal[start:end] += h

    return signal


METHODS = {"green": None, "chrom": None, "pos": None}  # populated by preprocessing.py
