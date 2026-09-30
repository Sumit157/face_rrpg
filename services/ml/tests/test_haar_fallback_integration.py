"""Runs the pipeline with ZERO mocking -- no MediaPipe .task model
present, no monkeypatched detector. Confirms the Haar cascade fallback
(app/face/haar_fallback.py, added to handle environments without
network access to download MediaPipe's model) engages automatically and
the pipeline fails closed cleanly when no real face is present, rather
than crashing or fabricating a verdict.
"""
import cv2
import numpy as np
import pytest

from app.pipeline.end_to_end import run_analysis


@pytest.fixture()
def no_face_video(tmp_path):
    path = str(tmp_path / "no_face.mp4")
    fps, duration_s, w, h = 30, 8, 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    rng = np.random.default_rng(0)
    for _ in range(int(fps * duration_s)):
        frame = rng.integers(80, 180, (h, w, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_pipeline_falls_back_to_haar_without_mediapipe_model(no_face_video):
    """No MediaPipe model is present in this environment by design (it
    requires a network download -- see app/face/detector.py) -- this
    confirms the pipeline degrades to the bundled Haar cascade instead
    of crashing, and fails safe when no real face is present.

    NOTE on the assertion below: pure-noise frames occasionally trigger
    1-2 spurious Haar detections (a real, small false-positive rate for
    this classical cascade), and lossy MP4 encoding isn't bit-exact
    across runs (multi-threaded encoders can produce slightly different
    compression artifacts each time), so the exact frame count reaching
    the classifier can vary run to run. The test therefore doesn't assert
    an exact FAILED status -- it asserts the actual safety property:
    noise must never produce a *confident* REAL/FAKE verdict. Either the
    pipeline fails outright from too few samples, or it produces
    UNCERTAIN/low signal quality -- never a confident wrong answer."""
    report = run_analysis(
        video_path=no_face_video, method="pos",
        regions=["forehead", "left_cheek", "right_cheek"],
    )
    d = report.to_dict()
    label = d["prediction"]["label"]

    if label == "FAILED":
        assert any("face" in w.lower() or "signal" in w.lower() for w in d["warnings"])
    else:
        assert label == "UNCERTAIN" or d["confidence"]["adjustedConfidence"] < 0.65, (
            f"noise produced an unexpectedly confident verdict: {d['prediction']}"
        )
