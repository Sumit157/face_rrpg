"""Frame-by-frame video reading, decoupled from any particular consumer."""
from __future__ import annotations
from typing import Iterator, Tuple
import cv2
import numpy as np


class VideoReader:
    """Iterates BGR frames from a video file with a simple, explicit API.

    Using a context-managed reader (rather than passing raw cv2.VideoCapture
    objects around) keeps resource cleanup predictable across the pipeline.
    """

    def __init__(self, video_path: str, max_frames: int | None = None):
        self.video_path = video_path
        self.max_frames = max_frames
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self) -> "VideoReader":
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._cap is not None:
            self._cap.release()

    @property
    def fps(self) -> float:
        assert self._cap is not None
        return float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0

    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        assert self._cap is not None, "Use VideoReader as a context manager"
        idx = 0
        while True:
            if self.max_frames is not None and idx >= self.max_frames:
                break
            ok, frame = self._cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
