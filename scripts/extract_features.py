#!/usr/bin/env python3
"""Phase 7/8: turn the cached per-region RGB traces from
preprocess_videos.py into a flat feature table ready for training.

Usage:
    python scripts/extract_features.py --manifest data/metadata/preprocess_manifest.json \
        --method pos --output data/processed/features/features.csv
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "ml"))

import numpy as np
import pandas as pd

from app.rppg.preprocessing import extract_rppg
from app.features.hr_features import analyse_region
from app.features.consistency import cross_region_consistency
from app.features.motion import score_motion
from app.features.illumination import score_illumination
from app.features.feature_vector import build_feature_vector, FEATURE_NAMES
from app.core.logging import get_logger

logger = get_logger("extract_features")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/metadata/preprocess_manifest.json")
    parser.add_argument("--method", default="pos", choices=["green", "chrom", "pos"])
    parser.add_argument("--window-seconds", type=float, default=12.0)
    parser.add_argument("--step-seconds", type=float, default=3.0)
    parser.add_argument("--output", default="data/processed/features/features.csv")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    rows = []
    for entry in manifest:
        data = np.load(entry["signalFile"])
        fps = float(data["fps"])
        region_keys = [k for k in data.files if k.startswith("region_")]

        region_signals = {}
        for key in region_keys:
            region = key.replace("region_", "")
            rgb = data[key]
            if len(rgb) < 30:
                continue
            region_signals[region] = extract_rppg(rgb, fps, method=args.method)

        if not region_signals:
            logger.warning("No usable regions for %s, skipping", entry["video"])
            continue

        region_results = {
            r: analyse_region(r, sig, fps, args.window_seconds, args.step_seconds)
            for r, sig in region_signals.items()
        }
        peak_hz = {r: res.full_spectrum.peak_hz for r, res in region_results.items()}
        consistency = cross_region_consistency(region_signals, peak_hz)

        # Motion/illumination need raw RGB and centroids; centroids weren't
        # cached by preprocess_videos.py in this minimal version, so motion
        # defaults to zero here -- extend preprocess_videos.py to cache
        # centroids if motion robustness experiments need it per-sample.
        from app.features.motion import MotionResult
        motion = MotionResult(0.0, 0.0, 0.0)
        illumination = score_illumination(np.concatenate(
            [data[k] for k in region_keys], axis=0
        ))

        fv = build_feature_vector(region_results, consistency, motion, illumination)
        row = dict(zip(FEATURE_NAMES, fv.as_array()))
        row["label"] = entry["label"]
        row["subjectId"] = entry["subjectId"]
        row["video"] = entry["video"]
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d feature rows to %s", len(df), out_path)


if __name__ == "__main__":
    main()
