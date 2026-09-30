"""Video metadata extraction using OpenCV."""
from __future__ import annotations
from dataclasses import dataclass
import cv2


@dataclass
class VideoMetadata:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    codec: str

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3),
            "frameCount": self.frame_count,
            "durationSeconds": round(self.duration_seconds, 3),
            "codec": self.codec,
        }


def read_metadata(video_path: str) -> VideoMetadata:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(
            [chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]
        ).strip()
        duration = frame_count / fps if fps > 0 else 0.0
        return VideoMetadata(
            path=video_path,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration,
            codec=codec or "unknown",
        )
    finally:
        cap.release()
