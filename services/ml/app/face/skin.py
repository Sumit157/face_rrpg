"""YCbCr skin-pixel segmentation.

ROI polygons (landmark groups or Haar sub-rects) still contain eyes,
brows, nostrils, lips and hair edges. Restricting the mean colour to
pixels that actually look like skin in YCbCr space removes that
contamination, which measurably cleans the rPPG trace on low-resolution
video where a few bad pixels matter a lot.

The classic 8-bit YCbCr rule (Kovac et al. / widely used in face-skin
literature): 77 <= Cb <= 127 and 133 <= Cr <= 173, plus a luminance
sanity range so pure-black/white pixels (shadows, speculars) are dropped.
"""
from __future__ import annotations
from typing import Tuple

import cv2
import numpy as np

CB_RANGE = (77, 127)
CR_RANGE = (133, 173)
Y_RANGE = (35, 235)
MIN_SKIN_FRACTION = 0.08  # below this the threshold is probably wrong for this ROI
MIN_SKIN_PIXELS = 12


def skin_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Boolean (H, W) mask of skin-like pixels."""
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    inside = (
        (cb >= CB_RANGE[0]) & (cb <= CB_RANGE[1])
        & (cr >= CR_RANGE[0]) & (cr <= CR_RANGE[1])
        & (y >= Y_RANGE[0]) & (y <= Y_RANGE[1])
    )
    return inside


def mean_skin_rgb(frame_bgr: np.ndarray, roi_mask: np.ndarray) -> np.ndarray | None:
    """Mean R, G, B over skin pixels inside the ROI mask.

    Falls back to all ROI pixels when the skin threshold rejects nearly
    everything (e.g. unusual skin tones under coloured lighting) so the
    pipeline degrades to the unsegmented behaviour instead of losing the
    sample entirely. Returns None only if the ROI itself is empty."""
    roi = roi_mask.astype(bool)
    if not roi.any():
        return None
    skin = skin_mask(frame_bgr) & roi
    roi_count = int(roi.sum())
    if (
        int(skin.sum()) < MIN_SKIN_PIXELS
        or int(skin.sum()) < MIN_SKIN_FRACTION * roi_count
    ):
        skin = roi
    b = frame_bgr[:, :, 0][skin]
    g = frame_bgr[:, :, 1][skin]
    r = frame_bgr[:, :, 2][skin]
    return np.array([r.mean(), g.mean(), b.mean()], dtype=np.float64)


def mean_skin_rgb_rect(
    frame_bgr: np.ndarray, rect: Tuple[int, int, int, int]
) -> np.ndarray | None:
    """Skin-filtered mean over an (x, y, w, h) rectangle (Haar path)."""
    x, y, w, h = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
    if w <= 0 or h <= 0:
        return None
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    x0, y0 = max(0, x), max(0, y)
    x1 = min(frame_bgr.shape[1], x + w)
    y1 = min(frame_bgr.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    mask[y0:y1, x0:x1] = 255
    return mean_skin_rgb(frame_bgr, mask)
