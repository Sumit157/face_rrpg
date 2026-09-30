"""Face landmark detection using MediaPipe's Face Landmarker (Tasks API).

MediaPipe >= 0.10 ships face detection/landmarking as a downloadable
"Tasks" model rather than bundling it in the package, so a `.task` model
file must be present on disk (see scripts/download_models.sh /
Dockerfile, which fetch it at build time). This gives 478 3D landmarks
per face, which is more than enough to derive stable forehead / cheek
ROIs.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import os
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mediapipe is required for face detection. Install with `pip install mediapipe`."
    ) from exc

DEFAULT_MODEL_PATH = os.getenv(
    "FACE_LANDMARKER_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "models_assets", "face_landmarker.task"),
)
MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


@dataclass
class FaceLandmarks:
    """Normalised (0-1) landmark coordinates for a single detected face."""
    points: np.ndarray  # shape (N, 2), x/y normalised to frame width/height
    frame_width: int
    frame_height: int

    def px(self, index: int) -> tuple[int, int]:
        x, y = self.points[index]
        return int(x * self.frame_width), int(y * self.frame_height)


class FaceDetector:
    """Resource-managed wrapper around MediaPipe FaceLandmarker (VIDEO mode).

    Raises a clear, actionable error if the model asset hasn't been
    downloaded yet, instead of a confusing low-level failure.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, max_num_faces: int = 1,
                 min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Face landmarker model not found at '{model_path}'.\n"
                f"Download it with:\n"
                f"  mkdir -p models_assets && curl -L -o {model_path} {MODEL_DOWNLOAD_URL}\n"
                f"or run scripts/download_models.sh (requires network access)."
            )
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            min_face_presence_confidence=min_detection_confidence,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def detect(self, frame_bgr: np.ndarray) -> Optional[FaceLandmarks]:
        h, w = frame_bgr.shape[:2]
        rgb = frame_bgr[:, :, ::-1].copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._timestamp_ms += 33  # nominal ~30fps step; caller doesn't need real PTS
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not result.face_landmarks:
            return None
        landmarks = result.face_landmarks[0]
        points = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
        return FaceLandmarks(points=points, frame_width=w, frame_height=h)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
