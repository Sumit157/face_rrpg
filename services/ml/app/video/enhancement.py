"""Frame enhancement for low-light video. Adapted from the rPPG
deepfake detector notebook (Kate & Shelar, MIT WPU): apply CLAHE
(contrast-limited adaptive histogram equalisation) to the luminance
channel only when a frame is dark enough that face detection is likely
to struggle, so well-lit frames aren't needlessly altered (which could
distort the very rPPG signal we're trying to measure)."""
from __future__ import annotations
import cv2
import numpy as np

DARK_FRAME_THRESHOLD = 90  # mean grayscale value below which CLAHE is applied


def enhance_if_dark(frame_bgr: np.ndarray, threshold: int = DARK_FRAME_THRESHOLD) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if gray.mean() >= threshold:
        return frame_bgr
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
