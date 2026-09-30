"""Central configuration for the ML service.

Values are overridable via environment variables so the same code works
locally, in Docker, and in notebooks/scripts.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

try:
    # Optional: load services/ml/.env (or the repo-root .env) when
    # python-dotenv is installed. All values remain overridable by real
    # environment variables either way.
    from dotenv import load_dotenv

    load_dotenv()
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))
except ImportError:
    pass


def _find_repo_root() -> Path:
    """Walk up from this file until pnpm-workspace.yaml is found, so
    relative STORAGE_DIR/MODEL_PATH values resolve identically whether the
    service is started from the repo root, services/ml, or Docker."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pnpm-workspace.yaml").exists():
            return parent
    return Path.cwd()


REPO_ROOT = _find_repo_root()


def _resolve(value: str) -> str:
    p = Path(value)
    return str(p if p.is_absolute() else (REPO_ROOT / p).resolve())


@dataclass
class Settings:
    app_name: str = "rppg-ml-service"
    host: str = os.getenv("ML_HOST", "0.0.0.0")
    port: int = int(os.getenv("ML_PORT", "8000"))

    # rPPG defaults
    default_method: str = os.getenv("RPPG_DEFAULT_METHOD", "pos")  # green | chrom | pos
    default_regions: List[str] = field(
        default_factory=lambda: ["forehead", "left_cheek", "right_cheek"]
    )
    default_window_seconds: float = float(os.getenv("RPPG_WINDOW_SECONDS", "12"))
    default_step_seconds: float = float(os.getenv("RPPG_STEP_SECONDS", "3"))

    # physiologically plausible heart-rate band (Hz) -> 42-240 bpm
    band_low_hz: float = 0.7
    band_high_hz: float = 4.0

    # storage -- resolved against the repo root, never the process cwd
    storage_dir: str = _resolve(os.getenv("STORAGE_DIR", "./storage"))
    max_video_size_mb: int = int(os.getenv("MAX_VIDEO_SIZE_MB", "500"))
    allowed_extensions: List[str] = field(
        default_factory=lambda: [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    )

    # model
    model_path: str = _resolve(os.getenv("MODEL_PATH", "./models/production/baseline.joblib"))


settings = Settings()
