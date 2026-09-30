#!/usr/bin/env python3
"""Phase 7 dataset pipeline: extract per-region RGB traces for every
video under data/raw/{real,fake} and cache them under data/processed.

Usage:
    python scripts/preprocess_videos.py --input data/raw --output data/processed
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "ml"))

import numpy as np
from app.pipeline.extraction import extract_region_traces
from app.core.logging import get_logger

logger = get_logger("preprocess_videos")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
REGIONS = ["forehead", "left_cheek", "right_cheek"]


def iter_videos(root: Path):
    for label_dir, label in [(root / "real", "real"), (root / "fake", "fake")]:
        if not label_dir.exists():
            continue
        for path in label_dir.rglob("*"):
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                yield path, label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--output", default="data/processed")
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output) / "signals"
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    for video_path, label in iter_videos(input_root):
        subject_id = video_path.stem
        out_path = output_root / f"{label}__{subject_id}.npz"
        logger.info("Processing %s (%s)", video_path, label)
        try:
            result = extract_region_traces(str(video_path), REGIONS)
        except FileNotFoundError as exc:
            logger.error("Skipping %s: %s", video_path, exc)
            continue

        np.savez_compressed(
            out_path,
            fps=result.fps,
            detection_rate=result.tracking.detection_rate,
            **{f"region_{r}": arr for r, arr in result.region_rgb.items()},
        )
        manifest.append({
            "video": str(video_path),
            "label": label,
            "subjectId": subject_id,
            "signalFile": str(out_path),
            "detectionRate": result.tracking.detection_rate,
        })

    manifest_path = Path(args.output) / "metadata" if False else Path("data/metadata")
    manifest_path.mkdir(parents=True, exist_ok=True)
    with open(manifest_path / "preprocess_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest with %d entries", len(manifest))


if __name__ == "__main__":
    main()
