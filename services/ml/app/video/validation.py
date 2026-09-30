"""Input validation for uploaded videos.

Deliberately conservative: reject anything that isn't a plausible,
decodable, reasonably-sized facial video before it reaches the rest of
the pipeline.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List

from app.core.config import settings
from app.video.metadata import read_metadata, VideoMetadata

MIN_DURATION_SECONDS = 3.0
MIN_FPS = 5.0
MAX_DURATION_SECONDS = 600.0


@dataclass
class ValidationResult:
    valid: bool
    metadata: VideoMetadata | None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def validate_video(video_path: str) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []

    ext = os.path.splitext(video_path)[1].lower()
    if ext not in settings.allowed_extensions:
        errors.append(f"Unsupported file extension '{ext}'")

    if not os.path.exists(video_path):
        errors.append("File does not exist")
        return ValidationResult(valid=False, metadata=None, errors=errors)

    size_mb = os.path.getsize(video_path) / (1024 * 1024)
    if size_mb > settings.max_video_size_mb:
        errors.append(
            f"File too large ({size_mb:.1f} MB > {settings.max_video_size_mb} MB)"
        )

    metadata = None
    try:
        metadata = read_metadata(video_path)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Could not decode video: {exc}")

    if metadata is not None:
        if metadata.fps < MIN_FPS:
            warnings.append(
                f"Low frame rate ({metadata.fps:.1f} fps) reduces rPPG reliability"
            )
        if metadata.duration_seconds < MIN_DURATION_SECONDS:
            errors.append(
                f"Video too short ({metadata.duration_seconds:.1f}s < "
                f"{MIN_DURATION_SECONDS}s minimum for a usable window)"
            )
        if metadata.duration_seconds > MAX_DURATION_SECONDS:
            warnings.append("Video is long; only the first 10 minutes will be analysed")
        if metadata.width == 0 or metadata.height == 0:
            errors.append("Video has no readable frames")

    return ValidationResult(
        valid=len(errors) == 0, metadata=metadata, errors=errors, warnings=warnings
    )
