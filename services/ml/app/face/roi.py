"""Region-of-interest extraction: forehead, left cheek, right cheek.

Landmark indices below are standard MediaPipe FaceMesh indices chosen to
sit on skin (avoiding eyes, brows, mouth, nostrils) so the extracted RGB
signal isn't contaminated by non-skin pixels or fast eye/mouth motion.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import cv2

from app.face.detector import FaceLandmarks

# Landmark index groups that outline a small polygon per region.
REGION_LANDMARKS: Dict[str, List[int]] = {
    "forehead": [10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109],
    "left_cheek": [50, 187, 207, 205, 36, 142, 100, 118],
    "right_cheek": [280, 411, 427, 425, 266, 371, 329, 347],
}

RegionMask = np.ndarray


@dataclass
class ROISet:
    """Extracted skin-pixel masks for one frame, one region per key."""
    masks: Dict[str, RegionMask]


def _polygon_mask(frame_shape: Tuple[int, int], landmarks: FaceLandmarks,
                   indices: List[int]) -> RegionMask:
    h, w = frame_shape
    pts = np.array([landmarks.px(i) for i in indices], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(pts) >= 3:
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)
    return mask


def extract_rois(frame_shape: Tuple[int, int], landmarks: FaceLandmarks,
                  regions: List[str]) -> ROISet:
    masks = {}
    for region in regions:
        indices = REGION_LANDMARKS.get(region)
        if indices is None:
            continue
        masks[region] = _polygon_mask(frame_shape, landmarks, indices)
    return ROISet(masks=masks)


def mean_rgb_in_mask(frame_bgr: np.ndarray, mask: RegionMask) -> np.ndarray | None:
    """Mean R, G, B over masked pixels. Returns None if the mask is empty."""
    if mask.sum() == 0:
        return None
    b, g, r = cv2.split(frame_bgr)
    count = mask.astype(bool)
    if not count.any():
        return None
    mean_b = b[count].mean()
    mean_g = g[count].mean()
    mean_r = r[count].mean()
    return np.array([mean_r, mean_g, mean_b], dtype=np.float64)
