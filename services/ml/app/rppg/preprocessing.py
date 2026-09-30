"""Ties together RGB trace -> chosen method -> preprocessed rPPG signal."""
from __future__ import annotations
import numpy as np

from app.rppg.green import green_signal
from app.rppg.chrom import chrom_signal
from app.rppg.pos import pos_signal
from app.rppg.filtering import preprocess

METHOD_FUNCS = {
    "green": lambda rgb, fps: green_signal(rgb),
    "chrom": lambda rgb, fps: chrom_signal(rgb),
    "pos": lambda rgb, fps: pos_signal(rgb, fps),
}


def extract_rppg(rgb_trace: np.ndarray, fps: float, method: str = "pos") -> np.ndarray:
    """rgb_trace: (T, 3) raw mean RGB per frame. Returns preprocessed rPPG (T,)."""
    method = method.lower()
    if method not in METHOD_FUNCS:
        raise ValueError(f"Unknown rPPG method '{method}'. Use one of {list(METHOD_FUNCS)}")
    raw = METHOD_FUNCS[method](rgb_trace, fps)
    return preprocess(raw, fps)
