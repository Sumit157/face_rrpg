"""In-memory progress registry.

/v1/analyse is a single synchronous HTTP call, but the Node API needs live
stage updates while that call is running. The FastAPI route registers an
`on_progress` callback that writes here, and the Node worker polls
GET /v1/analyse/{id}/progress every few hundred milliseconds.

Kept in-process and unbounded-but-trimmed: entries are cleared on start of
each analysis and pruned when older than a day (research service, one host,
no need for Redis).
"""
from __future__ import annotations

import threading
import time
from typing import Dict

_lock = threading.Lock()
_store: Dict[str, dict] = {}
_MAX_AGE_SECONDS = 24 * 3600


def start(analysis_id: str) -> None:
    now = time.time()
    with _lock:
        # Opportunistic prune so the registry can't grow forever.
        stale = [k for k, v in _store.items() if now - v["updatedAt"] > _MAX_AGE_SECONDS]
        for k in stale:
            del _store[k]
        _store[analysis_id] = {
            "analysisId": analysis_id,
            "stage": "queued",
            "progress": 0,
            "message": "Queued",
            "done": False,
            "updatedAt": now,
            "startedAt": now,
            # Stage transitions with timestamps relative to start (seconds)
            # -- lets the UI draw live stage durations while running.
            "history": [{"stage": "queued", "at": 0.0}],
        }


def update(analysis_id: str, stage: str, progress: int, message: str) -> None:
    with _lock:
        entry = _store.setdefault(analysis_id, {"analysisId": analysis_id})
        now = time.time()
        if entry.get("stage") != stage:
            started_at = entry.get("startedAt", now)
            history = entry.setdefault("history", [])
            history.append({"stage": stage, "at": round(now - started_at, 3)})
        entry.update(
            {
                "stage": stage,
                "progress": progress,
                "message": message,
                "done": stage in ("completed", "failed"),
                "updatedAt": now,
            }
        )


def get(analysis_id: str) -> dict | None:
    with _lock:
        entry = _store.get(analysis_id)
        return dict(entry) if entry else None


def clear(analysis_id: str) -> None:
    with _lock:
        _store.pop(analysis_id, None)
