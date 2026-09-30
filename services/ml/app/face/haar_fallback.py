"""Haar cascade face detector — fallback for when MediaPipe's Face
Landmarker is unavailable (e.g. its .task model hasn't been downloaded
yet) or fails to detect a face on a given frame.

Unlike the MediaPipe Tasks model, Haar cascade XML files ship bundled
inside opencv-python itself (cv2.data.haarcascades) — no network
download required. This makes it useful both as a genuine runtime
fallback (per the notebook this was adapted from) and as the only face
detector that can be exercised in network-restricted environments.

Trade-off: Haar gives only a face bounding box, not 468 fine-grained
landmarks, so ROI placement here is a cruder box-quadrant split rather
than the polygon masks in app/face/roi.py. Good enough for a fallback;
not a replacement for MediaPipe when it's available.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class HaarFaceBox:
    x: int
    y: int
    width: int
    height: int


class HaarFaceDetector:
    """Thin wrapper around OpenCV's bundled Haar cascade. No external
    model file, no network access required -- always available."""

    def __init__(self, cascade_name: str = "haarcascade_frontalface_default.xml"):
        cascade_path = cv2.data.haarcascades + cascade_name
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade from {cascade_path}")

    def detect(self, frame_bgr: np.ndarray) -> Optional[HaarFaceBox]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        detections = self._cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=2, minSize=(30, 30)
        )
        if len(detections) == 0:
            return None
        # Largest detected face by area, in case of multiple faces.
        x, y, w, h = max(detections, key=lambda d: d[2] * d[3])
        return HaarFaceBox(x=int(x), y=int(y), width=int(w), height=int(h))


def box_to_region_rects(box: HaarFaceBox) -> dict[str, Tuple[int, int, int, int]]:
    """Splits a face bounding box into forehead/left-cheek/right-cheek
    rectangles using the same empirically-chosen fractions as the
    MediaPipe-based ROIs, so both detectors produce comparable regions.
    Returns each region as (x0, y0, x1, y1) in pixel coordinates."""
    x0, y0, fw, fh = box.x, box.y, box.width, box.height
    return {
        "forehead": (
            x0 + fw // 4, y0 + int(fh * 0.04),
            x0 + 3 * fw // 4, y0 + int(fh * 0.22),
        ),
        "left_cheek": (
            x0 + int(fw * 0.08), y0 + int(fh * 0.42),
            x0 + int(fw * 0.38), y0 + int(fh * 0.68),
        ),
        "right_cheek": (
            x0 + int(fw * 0.62), y0 + int(fh * 0.42),
            x0 + int(fw * 0.92), y0 + int(fh * 0.68),
        ),
    }


def mean_rgb_in_rect(frame_bgr: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray | None:
    """Mirrors app.face.roi.mean_rgb_in_mask's signature/semantics but for
    a plain rectangle instead of a polygon mask, so callers can treat
    Haar and MediaPipe ROI sources interchangeably."""
    xa, ya, xb, yb = rect
    h, w = frame_bgr.shape[:2]
    xa, ya = max(0, xa), max(0, ya)
    xb, yb = min(w, xb), min(h, yb)
    if xb <= xa or yb <= ya:
        return None
    region = frame_bgr[ya:yb, xa:xb]
    b, g, r = cv2.split(region)
    return np.array([r.mean(), g.mean(), b.mean()], dtype=np.float64)
