"""CHROM method (de Haan & Jeanne, 2013).

Projects RGB onto two chrominance axes designed to cancel specular
reflection and motion-induced intensity changes that swamp naive
green-channel extraction, then combines them using their standard
deviation ratio to further suppress residual motion.
"""
from __future__ import annotations
import numpy as np


def chrom_signal(rgb_trace: np.ndarray) -> np.ndarray:
    """rgb_trace: shape (T, 3) with columns [R, G, B]. Returns shape (T,)."""
    if rgb_trace.ndim != 2 or rgb_trace.shape[1] != 3:
        raise ValueError("rgb_trace must have shape (T, 3)")

    r, g, b = rgb_trace[:, 0], rgb_trace[:, 1], rgb_trace[:, 2]
    # Normalise each channel by its temporal mean to remove DC / illumination level.
    eps = 1e-8
    rn = r / (r.mean() + eps)
    gn = g / (g.mean() + eps)
    bn = b / (b.mean() + eps)

    x = 3 * rn - 2 * gn
    y = 1.5 * rn + gn - 1.5 * bn

    std_x = x.std() + eps
    std_y = y.std() + eps
    alpha = std_x / std_y

    signal = x - alpha * y
    return signal
