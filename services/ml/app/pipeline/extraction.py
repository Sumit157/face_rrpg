"""Video -> per-region raw RGB traces + face/motion tracking, in one pass.

Reading the video only once and extracting all regions together keeps
this efficient and keeps the per-frame face-detection cost paid only
once regardless of how many regions/methods are requested downstream.

Hardening added by the pipeline overhaul:
- YCbCr skin segmentation inside each ROI (app.face.skin) so eyes/lips/
  hair edges don't pollute the mean colour.
- Background illumination subtraction: per-frame corner-patch luminance
  is recorded and its common-mode deviation removed from the region RGB
  after the pass (POS/CHROM then see an even cleaner trace).
- Optical-flow motion magnitude inside the face box, blended into the
  motion score (catches translation that centroid displacement misses).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np
import cv2

from app.video.reader import VideoReader
from app.face.detector import FaceDetector
from app.face.tracker import TrackingReport
from app.face.roi import REGION_LANDMARKS, extract_rois, mean_rgb_in_mask
from app.face.skin import mean_skin_rgb, mean_skin_rgb_rect
from app.face.haar_fallback import HaarFaceDetector, box_to_region_rects, mean_rgb_in_rect
from app.video.enhancement import enhance_if_dark
from app.explainability.visuals import FrameTraceCollector
from app.core.logging import get_logger

logger = get_logger(__name__)

MAX_ANALYSED_SECONDS = 600
FLOW_WIDTH = 320  # downscale for Farneback optical flow (speed)


@dataclass
class ExtractionResult:
    fps: float
    region_rgb: Dict[str, np.ndarray]  # region -> (T, 3)
    tracking: TrackingReport
    face_centroids: List[np.ndarray]
    frames_processed: int
    used_haar_fallback: bool = False
    haar_frame_count: int = 0
    # Normalised per-frame detection records (~5 Hz) for the UI overlay.
    frame_trace: List[dict] = field(default_factory=list)
    frame_stride: int = 1
    flow_magnitudes: List[float] = field(default_factory=list)  # normalised units/frame


def _try_init_mediapipe() -> FaceDetector | None:
    """MediaPipe's Face Landmarker needs a downloaded .task model file
    (see app/face/detector.py). If it isn't present, fall back to Haar
    for the whole run rather than failing outright."""
    try:
        return FaceDetector(max_num_faces=1)
    except FileNotFoundError as exc:
        logger.warning(
            "MediaPipe Face Landmarker unavailable (%s). Falling back to "
            "Haar cascade detection for this run -- expect coarser ROIs "
            "and reduced accuracy.", exc,
        )
        return None


def _norm_bbox(points: np.ndarray) -> List[float]:
    """Bounding box of landmark points as [x, y, w, h] normalised to 0-1."""
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    return [round(float(x0), 4), round(float(y0), 4),
            round(float(x1 - x0), 4), round(float(y1 - y0), 4)]


def _norm_roi_polygons(
    points: np.ndarray, regions: List[str]
) -> Dict[str, List[List[float]]]:
    """Per-region ROI outlines (MediaPipe landmark groups) normalised to 0-1."""
    out: Dict[str, List[List[float]]] = {}
    for region in regions:
        indices = REGION_LANDMARKS.get(region)
        if not indices:
            continue
        out[region] = [
            [round(float(points[i, 0]), 4), round(float(points[i, 1]), 4)]
            for i in indices
            if i < len(points)
        ]
    return out


def _haar_roi_polygons(
    box, regions: List[str], frame_w: int, frame_h: int
) -> Dict[str, List[List[float]]]:
    """Haar fallback equivalent: region rectangles as 4-point polygons."""
    rects = box_to_region_rects(box)
    out: Dict[str, List[List[float]]] = {}
    for region in regions:
        rect = rects.get(region)
        if rect is None:
            continue
        xa, ya, xb, yb = rect
        out[region] = [
            [round(xa / frame_w, 4), round(ya / frame_h, 4)],
            [round(xb / frame_w, 4), round(ya / frame_h, 4)],
            [round(xb / frame_w, 4), round(yb / frame_h, 4)],
            [round(xa / frame_w, 4), round(yb / frame_h, 4)],
        ]
    return out


class _FlowMeter:
    """Farneback optical flow on a downscaled frame, masked to the face box."""

    def __init__(self) -> None:
        self.prev_gray: np.ndarray | None = None

    def update(self, frame_bgr: np.ndarray, box) -> float | None:
        h, w = frame_bgr.shape[:2]
        scale = FLOW_WIDTH / w
        small = cv2.resize(frame_bgr, (FLOW_WIDTH, max(1, int(h * scale))))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mag: float | None = None
        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                pyr_scale=0.5, levels=2, winsize=15,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
            )
            speed = np.linalg.norm(flow, axis=2)
            if box is not None:
                x0 = max(0, int(box[0] * scale))
                y0 = max(0, int(box[1] * scale))
                x1 = min(speed.shape[1], int((box[0] + box[2]) * scale))
                y1 = min(speed.shape[0], int((box[1] + box[3]) * scale))
                region = speed[y0:y1, x0:x1]
                if region.size == 0:
                    region = speed
            else:
                region = speed
            # Normalised to frame width so the scale matches centroid displacement.
            mag = float(region.mean()) / FLOW_WIDTH
        self.prev_gray = gray
        return mag


def _corner_luma(frame_bgr: np.ndarray) -> float:
    """Mean luminance of four corner patches -- a background/illumination proxy."""
    h, w = frame_bgr.shape[:2]
    p = max(8, min(h, w) // 10)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners = [
        gray[:p, :p], gray[:p, -p:], gray[-p:, :p], gray[-p:, -p:],
    ]
    return float(np.mean([c.mean() for c in corners]))


def extract_region_traces(video_path: str, regions: List[str]) -> ExtractionResult:
    tracking = TrackingReport()
    region_samples: Dict[str, List[tuple[int, np.ndarray]]] = {r: [] for r in regions}
    bg_luma_by_idx: Dict[int, float] = {}
    centroids: List[np.ndarray] = []
    flow_magnitudes: List[float] = []
    haar_frame_count = 0
    collector: FrameTraceCollector | None = None
    flow_meter = _FlowMeter()

    mediapipe_detector = _try_init_mediapipe()
    haar_detector = HaarFaceDetector() if mediapipe_detector is None else None
    used_haar_fallback = mediapipe_detector is None

    try:
        with VideoReader(video_path) as reader:
            fps = reader.fps
            max_frames = int(MAX_ANALYSED_SECONDS * fps)
            collector = FrameTraceCollector(fps)
            for idx, frame in reader.frames():
                if idx >= max_frames:
                    break
                h, w = frame.shape[:2]
                timestamp_s = idx / fps if fps else 0.0
                bg_luma_by_idx[idx] = _corner_luma(frame)
                # CLAHE is only ever used to HELP DETECTION on dark frames --
                # it's a nonlinear local-contrast operation, so running it on
                # the frame we then measure a few-unit pulsatile brightness
                # change from would corrupt the very signal we're extracting.
                # The RGB samples below always come from the original frame.
                detection_frame = enhance_if_dark(frame)
                enhanced = detection_frame is not frame

                if mediapipe_detector is not None:
                    landmarks = mediapipe_detector.detect(detection_frame)
                    if landmarks is not None:
                        tracking.update(True)
                        centroids.append(landmarks.points.mean(axis=0))
                        bbox = _norm_bbox(landmarks.points)
                        mag = flow_meter.update(frame, bbox)
                        if mag is not None:
                            flow_magnitudes.append(mag)
                        rois = extract_rois((h, w), landmarks, regions)
                        for region, mask in rois.masks.items():
                            rgb = mean_skin_rgb(frame, mask)  # original frame, skin pixels
                            if rgb is not None:
                                region_samples[region].append((idx, rgb))
                        collector.add(
                            idx, timestamp_s, True, "mediapipe",
                            bbox=bbox,
                            roi=_norm_roi_polygons(landmarks.points, regions),
                            enhanced=enhanced,
                        )
                        continue
                    # MediaPipe missed this frame -- try Haar before giving up.
                    if haar_detector is None:
                        haar_detector = HaarFaceDetector()

                box = haar_detector.detect(detection_frame)
                if box is None:
                    tracking.update(False)
                    flow_meter.update(frame, None)
                    collector.add(
                        idx, timestamp_s, False, "none",
                        bbox=None, roi=None, enhanced=enhanced,
                    )
                    continue

                tracking.update(True)
                haar_frame_count += 1
                centroids.append(
                    np.array([(box.x + box.width / 2) / w, (box.y + box.height / 2) / h])
                )
                mag = flow_meter.update(
                    frame, (box.x, box.y, box.width, box.height)
                )
                if mag is not None:
                    flow_magnitudes.append(mag)
                rects = box_to_region_rects(box)
                for region in regions:
                    rect = rects.get(region)
                    if rect is None:
                        continue
                    xa, ya, xb, yb = rect
                    rgb = mean_skin_rgb_rect(
                        frame, (xa, ya, xb - xa, yb - ya)
                    )  # original frame, skin pixels
                    if rgb is not None:
                        region_samples[region].append((idx, rgb))
                collector.add(
                    idx, timestamp_s, True, "haar",
                    bbox=[round(box.x / w, 4), round(box.y / h, 4),
                          round(box.width / w, 4), round(box.height / h, 4)],
                    roi=_haar_roi_polygons(box, regions, w, h),
                    enhanced=enhanced,
                )
    finally:
        if mediapipe_detector is not None:
            mediapipe_detector.close()

    # --- Background illumination subtraction ---------------------------------
    # Common-mode (additive) luminance drift from the scene -- clouds, auto
    # exposure, someone moving off-camera -- is identical across regions and
    # channels, so removing the corner-patch deviation leaves the pulsatile
    # chromaticity signal untouched while POS/CHROM see a steadier trace.
    bg_median = float(np.median(list(bg_luma_by_idx.values()))) if bg_luma_by_idx else None
    region_rgb: Dict[str, np.ndarray] = {}
    for region, samples in region_samples.items():
        if not samples:
            region_rgb[region] = np.zeros((0, 3))
            continue
        idxs = [i for i, _ in samples]
        arr = np.array([rgb for _, rgb in samples], dtype=np.float64)
        if bg_median is not None:
            bg = np.array([bg_luma_by_idx.get(i, bg_median) for i in idxs])
            drift = bg - bg_median
            arr = arr - drift[:, None]
        region_rgb[region] = arr

    logger.info(
        "Extraction complete: %d frames, detection_rate=%.2f, haar_fallback_frames=%d, "
        "flow_samples=%d",
        tracking.total_frames, tracking.detection_rate, haar_frame_count,
        len(flow_magnitudes),
    )

    return ExtractionResult(
        fps=fps,
        region_rgb=region_rgb,
        tracking=tracking,
        face_centroids=centroids,
        frames_processed=tracking.total_frames,
        used_haar_fallback=used_haar_fallback,
        haar_frame_count=haar_frame_count,
        frame_trace=collector.entries if collector else [],
        frame_stride=collector.stride if collector else 1,
        flow_magnitudes=flow_magnitudes,
    )
