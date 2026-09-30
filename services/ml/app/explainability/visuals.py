"""Persist a per-analysis visualization trace for the UI's "Pipeline" tab.

storage/visuals/{analysis_id}.json answers the questions the signal
artifacts don't: which frames were actually read, when a face was (or
wasn't) detected, where the face box and per-region ROI polygons sat in
each sampled frame, how long each pipeline stage took, and how the clip
was split into sliding windows.

Kept out of the result payload (like the signal artifacts) -- loaded on
demand by the Node API's GET /api/analyses/:id/visuals. Coordinates are
normalised to [0, 1] so the frontend can draw them directly onto the
video element regardless of playback size.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)

# ~5 Hz frame sampling for the trace; the pipeline itself still reads
# every frame -- this only controls how many entries the UI overlay draws.
TARGET_FRAME_HZ = 5.0
MAX_FRAME_ENTRIES = 600


class FrameTraceCollector:
    """Collects normalised per-frame detection records at ~TARGET_FRAME_HZ.

    If a long video would exceed MAX_FRAME_ENTRIES the stride doubles and
    existing entries are thinned, keeping the file bounded while staying
    evenly spread across the clip.
    """

    def __init__(self, fps: float):
        self.stride = max(1, int(round(fps / TARGET_FRAME_HZ))) if fps and fps > 0 else 6
        self.entries: List[dict] = []

    def add(
        self,
        frame_index: int,
        timestamp_s: float,
        detected: bool,
        method: str,
        bbox: Optional[List[float]],
        roi: Optional[Dict[str, List[List[float]]]],
        enhanced: bool = False,
    ) -> None:
        if frame_index % self.stride != 0:
            return
        if len(self.entries) >= MAX_FRAME_ENTRIES:
            self._thin()
        self.entries.append(
            {
                "f": frame_index,
                "t": round(timestamp_s, 3),
                "detected": detected,
                "method": method,
                "bbox": bbox,
                "roi": roi,
                "enhanced": enhanced,
            }
        )

    def _thin(self) -> None:
        del self.entries[::2]
        self.stride *= 2


def save_visuals(
    analysis_id: str,
    storage_dir: str,
    *,
    stages: List[dict],
    extraction: Optional[dict] = None,
    frames: Optional[List[dict]] = None,
    windows: Optional[List[dict]] = None,
    result: Optional[dict] = None,
) -> Optional[str]:
    """Write storage/visuals/{analysis_id}.json and return its path.

    Returns None only when there is genuinely nothing to show (no stage
    timeline and no frame records -- e.g. the caller never started).
    Always called on both success and failure paths so the UI can show
    how far an analysis got before it broke.
    """
    if not stages and not frames:
        return None
    out_dir = Path(storage_dir) / "visuals"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysisId": analysis_id,
        "stages": stages,
        "extraction": extraction or {},
        "frames": frames or [],
        "windows": windows or [],
        "result": result or {},
    }
    path = out_dir / f"{analysis_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    logger.info(
        "Saved visuals trace for %s -> %s (%d frame entries, %d stages)",
        analysis_id, path, len(frames or []), len(stages),
    )
    return str(path)
