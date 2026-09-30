#!/usr/bin/env python3
"""Generate synthetic test videos for exercising the full pipeline
without needing real footage.

Uses a static portrait photo (default: matplotlib's bundled
grace_hopper.jpg) and modulates its colour channels with a
heartbeat-like pulse so the rPPG pipeline has a real face to detect and
a physiologically plausible signal to recover.

Sensitivity-test variants:
  --bpm 72        clean physiological pulse (baseline, should read REAL)
  --pulse-amp 0   no pulse at all (flat noise only)
  --erratic       BPM jumps randomly every 2 s (unstable pulse)
  --desync 1.5    forehead/cheeks get phase-shifted pulses (spatially
                  inconsistent blood-flow timing)

Usage (from repo root, ML venv active):
    python scripts/make_test_video.py [--image PATH] [--out PATH] \
        [--seconds 15] [--bpm 72] [--pulse-amp 6] [--erratic] \
        [--desync SECONDS]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = (
    REPO_ROOT
    / "services"
    / "ml"
    / "venv"
    / "Lib"
    / "site-packages"
    / "matplotlib"
    / "mpl-data"
    / "sample_data"
    / "grace_hopper.jpg"
)
DEFAULT_OUT = REPO_ROOT / "tests" / "fixtures" / "synthetic_face_video.mp4"

# Face crop window (centre-weighted square) - keeps the face large enough
# for MediaPipe/Haar detection at 640x640.
CROP_SIZE = 520


def face_square(img: np.ndarray) -> tuple[int, int, int]:
    """Return (x, y, side) of a centre-weighted square around the face."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    )
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    h, w = img.shape[:2]
    side = min(CROP_SIZE, h, w)
    if len(faces):
        # Largest detected face
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        cx, cy = x + fw // 2, y + fh // 2
        side = min(max(int(fh * 2.2), 240), h, w)
    else:
        cx, cy = w // 2, int(h * 0.42)
    x0 = max(0, min(cx - side // 2, w - side))
    y0 = max(0, min(cy - side // 2, h - side))
    return x0, y0, side


def build_phase(args: argparse.Namespace) -> np.ndarray:
    """Per-frame pulse phase (radians), integrated over time.

    Constant BPM by default; --erratic re-samples the BPM every 2 s.
    """
    n = int(args.seconds * args.fps)
    if args.bpm <= 0 or args.pulse_amp <= 0:
        return np.zeros(n)
    if args.erratic:
        rng = np.random.default_rng(7)
        bpm_per_frame = np.repeat(rng.uniform(45.0, 150.0, size=max(1, n // (2 * int(args.fps)))), 2 * int(args.fps))
        bpm_per_frame = np.resize(bpm_per_frame, n)
    else:
        bpm_per_frame = np.full(n, float(args.bpm))
    phase_step = 2 * np.pi * bpm_per_frame / 60.0 / args.fps
    return np.cumsum(phase_step)


def region_masks(size: int) -> np.ndarray:
    """Band index per pixel: 0 = forehead (top third), 1 = left cheek,
    2 = right cheek -- approximate ROI geometry after the face crop."""
    yy, xx = np.mgrid[0:size, 0:size]
    masks = np.zeros((size, size), dtype=np.int8)
    masks[yy < size // 3] = 0
    masks[(yy >= size // 3) & (xx < size // 2)] = 1
    masks[(yy >= size // 3) & (xx >= size // 2)] = 2
    return masks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="input portrait photo")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output .mp4 path")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--bpm", type=float, default=72.0, help="0 = no pulse")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--pulse-amp",
        type=float,
        default=6.0,
        help="green-channel modulation depth (0 = flat/no pulse)",
    )
    parser.add_argument(
        "--erratic", action="store_true", help="BPM jumps randomly every 2 seconds"
    )
    parser.add_argument(
        "--desync",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="phase offset between facial regions (spatial desync)",
    )
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"Could not read image: {args.image}", file=sys.stderr)
        return 1

    x0, y0, side = face_square(img)
    crop = img[y0 : y0 + side, x0 : y0 + side]
    frame = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_AREA).astype(np.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (640, 640)
    )
    if not writer.isOpened():
        print("Could not open VideoWriter", file=sys.stderr)
        return 1

    n_frames = int(args.seconds * args.fps)
    phase = build_phase(args)
    masks = region_masks(640)
    # Per-region phase offsets: forehead 0, cheeks shifted by --desync.
    offset_frames = [0.0, args.desync * args.fps, 2.0 * args.desync * args.fps]
    idx = [np.clip(np.arange(n_frames) - off, 0, n_frames - 1).astype(int) for off in offset_frames]

    amp_g = args.pulse_amp
    rng = np.random.default_rng(42)

    for i in range(n_frames):
        # Blood-volume pulse: mostly green-channel modulation (as in real
        # PPG), tiny overall luminance wobble, mild sensor noise.
        # Each region band gets its own pulse phase, so regions can disagree.
        pulses = np.array(
            [np.sin(phase[j]) for j in (idx[0][i], idx[1][i], idx[2][i])],
            dtype=np.float32,
        )
        band_pulse = pulses[masks]  # (640, 640) pulse value per pixel
        noise = rng.normal(0, 1.6, frame.shape).astype(np.float32)
        out = frame + noise
        out[:, :, 1] += band_pulse * amp_g
        out[:, :, 0] += band_pulse * (amp_g * 1.5 / 6.0)
        out[:, :, 2] += band_pulse * (amp_g * 2.5 / 6.0)
        writer.write(np.clip(out, 0, 255).astype(np.uint8))

    writer.release()
    kind = (
        f"{args.bpm:g} bpm, amp {args.pulse_amp:g}"
        + (", erratic" if args.erratic else "")
        + (f", desync {args.desync:g}s" if args.desync else "")
    )
    print(f"Wrote {out_path} ({n_frames} frames; {kind})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
