"""Visuals trace artifact (storage/visuals/{analysis_id}.json).

Verifies the Pipeline-tab data: stage timeline, normalised frame trace,
window segmentation -- written on success AND on failure so the UI can
show how far an analysis got.
"""
import json

import cv2
import numpy as np
import pytest

import app.face.detector as detector_module
import app.pipeline.extraction as extraction_module
from app.core.config import settings
from app.pipeline.end_to_end import run_analysis

import test_pipeline_integration_stub as stubs

TRUE_BPM = 68.0


@pytest.fixture(autouse=True)
def _patch_face_detector(monkeypatch):
    monkeypatch.setattr(detector_module, "FaceDetector", stubs._StubFaceDetector)
    monkeypatch.setattr(extraction_module, "FaceDetector", stubs._StubFaceDetector)


@pytest.fixture()
def visuals_storage(tmp_path, monkeypatch):
    storage = tmp_path / "storage"
    monkeypatch.setattr(settings, "storage_dir", str(storage))
    return storage


@pytest.fixture()
def synthetic_face_video(tmp_path):
    path = str(tmp_path / "synthetic_face.mp4")
    fps, duration_s, w, h = 30, 15, 320, 240
    f_hr = TRUE_BPM / 60.0
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(int(fps * duration_s)):
        pulse = 6 * np.sin(2 * np.pi * f_hr * (i / fps))
        color = np.clip(np.array([120, 150, 200]) + pulse, 0, 255).astype(np.uint8)
        frame = np.full((h, w, 3), 40, dtype=np.uint8)
        cv2.ellipse(frame, (w // 2, h // 2), (60, 80), 0, 0, 360, color.tolist(), -1)
        writer.write(frame)
    writer.release()
    return path


def test_visuals_written_on_success(synthetic_face_video, visuals_storage):
    report = run_analysis(
        video_path=synthetic_face_video,
        method="pos",
        regions=["forehead", "left_cheek", "right_cheek"],
        window_seconds=5,
        step_seconds=2,
        analysis_id="visuals-test-ok",
    )
    path = visuals_storage / "visuals" / "visuals-test-ok.json"
    assert path.exists(), "visuals trace must be written on success"
    data = json.loads(path.read_text(encoding="utf-8"))

    # Stage timeline: ordered, non-negative durations, covers the run.
    stages = data["stages"]
    assert [s["stage"] for s in stages][0] == "validating"
    assert stages[-1]["stage"] == "completed"
    assert all(s["durationMs"] >= 0 for s in stages)
    assert all(
        stages[i]["endS"] <= stages[i + 1]["startS"] + 1e-6 for i in range(len(stages) - 1)
    )

    # Frame trace: normalised coordinates, honest miss records.
    frames = data["frames"]
    assert frames, "frame trace must not be empty"
    assert data["extraction"]["framesProcessed"] == 15 * 30
    assert data["extraction"]["detector"] == "mediapipe"
    for entry in frames:
        assert 0.0 <= entry["t"] <= 15.0
        if entry["detected"]:
            assert entry["method"] == "mediapipe"
            assert entry["bbox"] is not None and len(entry["bbox"]) == 4
            assert set(entry["roi"]) <= {"forehead", "left_cheek", "right_cheek"}
            for poly in entry["roi"].values():
                for x, y in poly:
                    assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0

    # Window segmentation: boundaries within clip, BPMs present.
    assert {w["region"] for w in data["windows"]} == {
        "forehead", "left_cheek", "right_cheek"
    }
    for region_windows in data["windows"]:
        assert region_windows["windows"]
        first = region_windows["windows"][0]
        assert first["startS"] == 0.0 and first["endS"] == 5.0
        assert first["bpm"] > 0
        assert region_windows["windows"][-1]["endS"] <= 15.0

    assert data["result"]["method"] == "pos"
    assert report.to_dict()["prediction"]["label"] in ("REAL", "FAKE", "UNCERTAIN")


def test_visuals_written_on_failure(tmp_path, visuals_storage):
    bad_path = str(tmp_path / "not_a_video.mp4")
    with open(bad_path, "wb") as f:
        f.write(b"not a real video file")

    report = run_analysis(video_path=bad_path, analysis_id="visuals-test-fail")
    assert report.to_dict()["prediction"]["label"] == "FAILED"

    path = visuals_storage / "visuals" / "visuals-test-fail.json"
    assert path.exists(), "failure runs must still leave a trace for the UI"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["stages"][0]["stage"] == "validating"
    assert data["stages"][-1]["stage"] == "failed"
    assert data["frames"] == []
    assert data["windows"] == []
