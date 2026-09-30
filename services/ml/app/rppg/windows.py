"""Sliding-window segmentation over a preprocessed signal."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class Window:
    index: int
    start_sample: int
    end_sample: int
    start_seconds: float
    end_seconds: float
    data: np.ndarray


def make_windows(x: np.ndarray, fps: float, window_seconds: float,
                  step_seconds: float) -> List[Window]:
    win_len = int(round(window_seconds * fps))
    step_len = max(1, int(round(step_seconds * fps)))
    windows: List[Window] = []
    if win_len <= 0 or win_len > len(x):
        if len(x) > 0:
            windows.append(Window(0, 0, len(x), 0.0, len(x) / fps, x))
        return windows

    idx = 0
    start = 0
    while start + win_len <= len(x):
        end = start + win_len
        windows.append(
            Window(
                index=idx,
                start_sample=start,
                end_sample=end,
                start_seconds=start / fps,
                end_seconds=end / fps,
                data=x[start:end],
            )
        )
        idx += 1
        start += step_len
    return windows
