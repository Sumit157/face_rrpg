"""Motion scoring from face landmark displacement between frames.

Used both as a robustness/warning signal (large motion degrades rPPG
reliability regardless of authenticity) and as an auxiliary feature.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class MotionResult:
    mean_displacement: float
    max_displacement: float
    motion_score: float  # 0-100, higher = more motion

    def to_dict(self) -> dict:
        return {
            "meanDisplacement": round(self.mean_displacement, 5),
            "maxDisplacement": round(self.max_displacement, 5),
            "motionScore": round(self.motion_score, 1),
        }


def score_motion(landmark_centroids: List[np.ndarray]) -> MotionResult:
    """landmark_centroids: list of (x, y) normalised face-centroid positions,
    one per frame (e.g. mean of all FaceMesh points)."""
    if len(landmark_centroids) < 2:
        return MotionResult(0.0, 0.0, 0.0)
    pts = np.array(landmark_centroids)
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    mean_disp = float(diffs.mean())
    max_disp = float(diffs.max())
    # Empirical scaling: normalised-coordinate frame-to-frame displacement
    # above ~0.02 (2% of frame size) is visible, deliberate head motion.
    motion_score = float(np.clip((mean_disp / 0.02) * 100, 0, 100))
    return MotionResult(mean_disp, max_disp, motion_score)
