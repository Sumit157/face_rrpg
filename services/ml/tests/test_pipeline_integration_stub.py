"""Integration test for the full video -> ROI -> rPPG -> features ->
prediction pipeline.

The real face detector requires a MediaPipe Tasks model file
(models_assets/face_landmarker.task) fetched over the network at build
time (see scripts/download_models.sh). To keep this test runnable
offline in CI, it monkeypatches FaceDetector with a stub that reports a
fixed face position -- everything downstream (ROI masks, RGB extraction,
rPPG, features, classifier, explainability) runs unmodified against real
pixel data with an embedded synthetic heartbeat.
"""
import numpy as np
import cv2
import pytest

import app.face.detector as detector_module
import app.pipeline.extraction as extraction_module
from app.pipeline.end_to_end import run_analysis

TRUE_BPM = 68.0


class _StubLandmarks:
    def __init__(self, w, h, points):
        self.frame_width = w
        self.frame_height = h
        self.points = points

    def px(self, index):
        x, y = self.points[index]
        return int(x * self.frame_width), int(y * self.frame_height)


class _StubFaceDetector:
    """Reports all 468 landmark points jittered around the frame centre,
    which is where the synthetic test video places its pulsing face."""

    def __init__(self, *args, **kwargs):
        self._rng = np.random.default_rng(0)

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        n = 468
        offsets = self._rng.uniform(-0.06, 0.06, size=(n, 2))
        points = np.column_stack([0.5 + offsets[:, 0], 0.5 + offsets[:, 1] * 0.6]).astype(np.float32)
        return _StubLandmarks(w, h, points)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


@pytest.fixture(autouse=True)
def _patch_face_detector(monkeypatch):
    monkeypatch.setattr(detector_module, "FaceDetector", _StubFaceDetector)
    monkeypatch.setattr(extraction_module, "FaceDetector", _StubFaceDetector)


@pytest.fixture()
def synthetic_face_video(tmp_path):
    path = str(tmp_path / "synthetic_face.mp4")
    fps, duration_s, w, h = 30, 15, 320, 240
    f_hr = TRUE_BPM / 60.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for i in range(int(fps * duration_s)):
        t = i / fps
        pulse = 6 * np.sin(2 * np.pi * f_hr * t)
        color = np.clip(np.array([120, 150, 200]) + pulse, 0, 255).astype(np.uint8)
        frame = np.full((h, w, 3), 40, dtype=np.uint8)
        cv2.ellipse(frame, (w // 2, h // 2), (60, 80), 0, 0, 360, color.tolist(), -1)
        writer.write(frame)
    writer.release()
    return path


def test_full_pipeline_on_synthetic_face_video(synthetic_face_video):
    events = []
    report = run_analysis(
        video_path=synthetic_face_video,
        method="pos",
        regions=["forehead", "left_cheek", "right_cheek"],
        window_seconds=5,
        step_seconds=2,
        on_progress=lambda stage, pct, msg: events.append((stage, pct, msg)),
    )
    d = report.to_dict()

    assert events[-1][0] == "completed"
    assert d["faceQuality"]["detectionRate"] == 1.0
    assert d["prediction"]["label"] in ("REAL", "FAKE", "UNCERTAIN")
    assert abs(d["bpm"]["overall"]["mean"] - TRUE_BPM) < 12
    assert 0 <= d["confidence"]["adjustedConfidence"] <= 1
    assert isinstance(d["evidence"], list)
    assert isinstance(d["warnings"], list)


def test_pipeline_reports_failure_on_video_with_no_frames(tmp_path):
    bad_path = str(tmp_path / "not_a_video.mp4")
    with open(bad_path, "wb") as f:
        f.write(b"not a real video file")

    report = run_analysis(video_path=bad_path)
    d = report.to_dict()
    assert d["prediction"]["label"] == "FAILED"
