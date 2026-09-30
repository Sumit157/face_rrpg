"""Video -> per-region raw RGB traces + face/motion tracking, in one pass.

Reading the video only once and extracting all regions together keeps
this efficient and keeps the per-frame face-detection cost paid only
once regardless of how many regions/methods are requested downstream.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np

from app.video.reader import VideoReader
from app.face.detector import FaceDetector
from app.face.tracker import TrackingReport
from app.face.roi import REGION_LANDMARKS, extract_rois, mean_rgb_in_mask
from app.face.haar_fallback import HaarFaceDetector, box_to_region_rects, mean_rgb_in_rect
from app.video.enhancement import enhance_if_dark
from app.explainability.visuals import FrameTraceCollector
from app.core.logging import get_logger

logger = get_logger(__name__)

MAX_ANALYSED_SECONDS = 600


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


def extract_region_traces(video_path: str, regions: List[str]) -> ExtractionResult:
    tracking = TrackingReport()
    region_samples: Dict[str, List[np.ndarray]] = {r: [] for r in regions}
    centroids: List[np.ndarray] = []
    haar_frame_count = 0
    collector: FrameTraceCollector | None = None

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
                        rois = extract_rois((h, w), landmarks, regions)
                        for region, mask in rois.masks.items():
                            rgb = mean_rgb_in_mask(frame, mask)  # original frame
                            if rgb is not None:
                                region_samples[region].append(rgb)
                        collector.add(
                            idx, timestamp_s, True, "mediapipe",
                            bbox=_norm_bbox(landmarks.points),
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
                rects = box_to_region_rects(box)
                for region in regions:
                    rect = rects.get(region)
                    if rect is None:
                        continue
                    rgb = mean_rgb_in_rect(frame, rect)  # original frame
                    if rgb is not None:
                        region_samples[region].append(rgb)
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

    region_rgb = {
        region: (np.array(samples) if samples else np.zeros((0, 3)))
        for region, samples in region_samples.items()
    }

    logger.info(
        "Extraction complete: %d frames, detection_rate=%.2f, haar_fallback_frames=%d",
        tracking.total_frames, tracking.detection_rate, haar_frame_count,
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
    )
