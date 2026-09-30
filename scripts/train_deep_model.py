#!/usr/bin/env python3
"""Milestone 13: train the 1D CNN / TCN over multi-region rPPG sequences.

This is a Phase-2 research extension (see spec section 24). Requires
PyTorch and a windowed-sequence dataset (not the flat feature table used
by train_baseline.py) -- build that dataset from data/processed/signals
before running this. Left as a scaffold: fill in DataLoader construction
once a labelled dataset with enough subjects is available, since a deep
model is not justifiable on a handful of clips.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "ml"))

from app.models.cnn1d import build_model as build_cnn1d
from app.models.tcn import build_model as build_tcn
from app.core.logging import get_logger

logger = get_logger("train_deep_model")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["cnn1d", "tcn"], default="cnn1d")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    logger.warning(
        "train_deep_model.py is a scaffold: implement dataset loading "
        "(windowed multi-region rPPG sequences + labels) before training. "
        "See docs/methodology.md for the intended experiment design."
    )
    model = build_cnn1d() if args.arch == "cnn1d" else build_tcn()
    logger.info("Instantiated %s model: %s", args.arch, model.__class__.__name__)


if __name__ == "__main__":
    main()
