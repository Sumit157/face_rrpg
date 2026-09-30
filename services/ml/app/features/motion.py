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


def score_motion(
    landmark_centroids: List[np.ndarray],
    flow_magnitudes: List[float] | None = None,
) -> MotionResult:
    """landmark_centroids: list of (x, y) normalised face-centroid positions,
    one per frame (e.g. mean of all FaceMesh points).

    flow_magnitudes (optional): mean optical-flow speed inside the face
    box (normalised frame units per frame) from the extraction pass.
    When present it is blended with the landmark-displacement score --
    optical flow catches translation/rotation that centroid displacement
    alone underestimates (and works on the Haar path where there are no
    landmarks to jitter with)."""
    if len(landmark_centroids) >= 2:
        pts = np.array(landmark_centroids)
        diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        mean_disp = float(diffs.mean())
        max_disp = float(diffs.max())
        # Empirical scaling: normalised-coordinate frame-to-frame displacement
        # above ~0.02 (2% of frame size) is visible, deliberate head motion.
        centroid_score = float(np.clip((mean_disp / 0.02) * 100, 0, 100))
    else:
        mean_disp, max_disp, centroid_score = 0.0, 0.0, 0.0

    if flow_magnitudes:
        flow = np.asarray(flow_magnitudes, dtype=np.float64)
        # Same 0.02 reference: mean flow of 2% frame size per frame = 100.
        flow_score = float(np.clip((float(flow.mean()) / 0.02) * 100, 0, 100))
        motion_score = 0.6 * centroid_score + 0.4 * flow_score
    else:
        motion_score = centroid_score
    return MotionResult(mean_disp, max_disp, float(np.clip(motion_score, 0, 100)))
