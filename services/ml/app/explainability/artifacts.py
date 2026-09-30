"""Persist large per-region signal arrays (waveform + spectrum) as JSON
files instead of embedding them in the analysis result.

The Node API's GET /api/analyses/:id/signal reads these files back for the
Signal Explorer UI -- keeping multi-hundred-kilobyte arrays out of the
primary result payload (spec: large arrays are saved as files/artifacts).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from app.core.logging import get_logger
from app.pipeline.analysis import RegionAnalysis

logger = get_logger(__name__)

# Cap waveform length so a long video can't produce a multi-MB artifact;
# indices are evenly strided, preserving shape.
MAX_WAVEFORM_POINTS = 4000


def _waveform_payload(signal: np.ndarray, fps: float) -> dict:
    n = len(signal)
    if n > MAX_WAVEFORM_POINTS:
        stride = int(np.ceil(n / MAX_WAVEFORM_POINTS))
        signal = signal[::stride]
        timestamps = np.arange(0, len(signal), dtype=np.float64) * (stride / fps)
    else:
        timestamps = np.arange(n, dtype=np.float64) / fps
    return {
        "timestamps": np.round(timestamps, 3).tolist(),
        "waveform": np.round(signal.astype(np.float64), 5).tolist(),
    }


def save_signal_artifacts(
    analysis_id: str,
    region_analysis: RegionAnalysis,
    fps: float,
    storage_dir: str,
) -> Dict[str, str]:
    """Write storage/signals/{analysis_id}.json and return region -> path."""
    if not region_analysis.region_signals:
        return {}

    out_dir = Path(storage_dir) / "signals"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, dict] = {}
    for region, signal in region_analysis.region_signals.items():
        entry = _waveform_payload(signal, fps)
        result = region_analysis.region_results.get(region)
        if result is not None:
            spectrum = result.full_spectrum
            entry.update(
                {
                    "freqsHz": np.round(spectrum.freqs_hz, 4).tolist(),
                    "power": np.round(spectrum.power, 6).tolist(),
                    "peakHz": round(spectrum.peak_hz, 4),
                    "bpm": round(spectrum.bpm, 2),
                }
            )
        payload[region] = entry

    path = out_dir / f"{analysis_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    logger.info("Saved signal artifacts for %s -> %s", analysis_id, path)
    return {region: str(path) for region in payload}


def load_signal_artifacts(storage_dir: str) -> Dict[str, dict]:
    """Read all storage/signals/*.json files (used by tests / notebooks)."""
    out_dir = Path(storage_dir) / "signals"
    if not out_dir.exists():
        return {}
    loaded: Dict[str, dict] = {}
    for path in out_dir.glob("*.json"):
        try:
            loaded[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Unreadable artifact file: %s", path)
    return loaded
