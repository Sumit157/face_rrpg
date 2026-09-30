"""Green-channel rPPG (Verkruysse et al., 2008).

The simplest, most widely replicated rPPG method: the green channel has
the strongest plethysmographic response because haemoglobin absorbs
green light most strongly. Used here as the baseline method every other
method is compared against.
"""
from __future__ import annotations
import numpy as np


def green_signal(rgb_trace: np.ndarray) -> np.ndarray:
    """rgb_trace: shape (T, 3) with columns [R, G, B]. Returns shape (T,)."""
    if rgb_trace.ndim != 2 or rgb_trace.shape[1] != 3:
        raise ValueError("rgb_trace must have shape (T, 3)")
    return rgb_trace[:, 1].astype(np.float64)
