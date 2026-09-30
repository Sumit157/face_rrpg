#!/usr/bin/env python3
"""Train the production RandomForest on a SYNTHETIC rPPG coherence corpus.

Why synthetic: no labelled real/fake video corpus ships with this repo
(and labelling one honestly is a research project of its own). What we
CAN generate rigorously is the underlying SIGNAL-LEVEL distinction the
detector is built on:

  REAL  -> a single latent cardiac pulse shared by all facial regions
          (physiological delays < 0.15 s, natural RR variability, PPG
          waveform with harmonics) plus realistic motion, illumination
          drift, noise and compression quantisation.

  FAKE  -> variants that break that structure: fully independent region
           noise, a shared pulse but regions desynchronised beyond
           physiological delay, a "too perfect" harmonic-free sine with
           zero variability (robotic), or a real pulse buried below the
           measurability floor.

Every sample is pushed through the SAME pipeline as production
(analyse_regions with method="auto" + quality gating + the shared
feature builder), so training and inference can never drift apart.
The class label is driven by pulse COHERENCE structure, while quality,
motion and illumination distributions are deliberately overlapping --
so the model cannot cheat by mapping "bad video" to "fake".

IMPORTANT: model_version_ is recorded as rf-synthetic-v1. This model
learns physiological coherence, not dataset identity; evaluate it on
real videos before trusting it anywhere important (docs/limitations.md).

Usage:
    python scripts/train_synthetic_rf.py --samples 4000 --output models/production/baseline.joblib
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "ml"))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from app.features.feature_vector import FEATURE_NAMES, build_feature_vector
from app.pipeline.analysis import analyse_regions

FPS = 30.0
DURATION_S = 16.0
N_FRAMES = int(FPS * DURATION_S)
REGIONS = ["forehead", "left_cheek", "right_cheek"]
MODEL_VERSION = "rf-synthetic-v1"

# Deepfake archetypes: no pulse, spatially desynced pulse, buried pulse,
# and physiologically impossible HR jumps (coherent everywhere but the
# heart rate teleports 45 <-> 150 bpm every ~2 s).
# ("sine" -- a perfectly-coherent synthetic pulse -- was removed: a
# rendered video WITH a coherent, stable pulse must classify REAL per the
# spec ("genuine pulse across regions -> Real").)
FAKE_KINDS = ["independent", "desync", "masked", "erratic"]
FAKE_WEIGHTS = [0.40, 0.30, 0.15, 0.15]

# Recording-quality tiers. Both classes draw from the SAME tier
# distribution (except "masked" fakes, forced to hard) so the model
# cannot use quality as a class shortcut -- but the corpus must still
# cover the full real-world quality range (studio clips up to clean
# measured SNR, hard clips down to the -6 dB floor), otherwise good
# real videos land out-of-distribution and get rejected.
TIERS = ("studio", "normal", "hard")
TIER_WEIGHTS = (0.35, 0.45, 0.20)
TIER_SPECS = {
    #  motion  illum  pulse  snr range (target, feeds noise sigma)
    "studio": {"motion": 0.30, "illum": 0.30, "pulse": 1.7, "snr": (4.0, 14.0)},
    "normal": {"motion": 1.00, "illum": 1.00, "pulse": 1.0, "snr": (-2.0, 8.0)},
    "hard":   {"motion": 1.70, "illum": 1.70, "pulse": 0.7, "snr": (-6.0, 2.0)},
}


# --------------------------------------------------------------------------
# Signal simulation
# --------------------------------------------------------------------------

def _ppg_at(t: np.ndarray, beat_times: np.ndarray) -> np.ndarray:
    """PPG-shaped pulse: sharp systolic peak + dicrotic notch per beat."""
    out = np.zeros_like(t)
    sigma = 0.05
    for bt in beat_times:
        d = t - bt
        if d.min() > 1.0 or d.max() < -1.0:
            continue
        sys_peak = np.exp(-0.5 * (d / sigma) ** 2)
        dic = 0.32 * np.exp(-0.5 * ((d - 0.30) / 0.09) ** 2)
        out += sys_peak + dic
    return out


def _beat_times(rng: np.random.Generator, with_variability: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(beat_times, instantaneous_hr) with realistic RR jitter/drift."""
    hr0 = rng.uniform(58, 92)
    drift_amp = rng.uniform(1.0, 5.0)
    phase = rng.uniform(0, 2 * np.pi)
    times = []
    t = rng.uniform(0.0, 0.4)
    while t < DURATION_S + 0.5:
        times.append(t)
        hr = hr0 + drift_amp * np.sin(2 * np.pi * t / DURATION_S + phase)
        rr = 60.0 / hr
        if with_variability:
            rr += rng.normal(0.0, 0.035)  # sinus arrhythmia-ish RR jitter
            rr = max(rr, 0.35)
        t += rr
    return np.array(times), hr0


def _illumination(t: np.ndarray, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """Common-mode multiplicative illumination change (all regions/channels)."""
    slow = scale * rng.uniform(0.02, 0.10) * np.sin(2 * np.pi * rng.uniform(0.5, 1.5) * t / DURATION_S + rng.uniform(0, 6.28))
    wiggle = scale * rng.uniform(0.01, 0.04) * np.sin(2 * np.pi * t / DURATION_S * rng.uniform(2, 5) + rng.uniform(0, 6.28))
    return 1.0 + slow + wiggle


def _motion_steps(rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """Additive baseline steps (head movement) with exponential decay."""
    baseline = np.zeros(N_FRAMES)
    for _ in range(rng.integers(0, 4)):
        start = rng.integers(0, N_FRAMES - 10)
        amp = scale * rng.uniform(0.5, 4.0) * rng.choice([-1, 1])
        tau = rng.uniform(0.2, 0.6) * FPS
        length = min(N_FRAMES - start, int(3 * tau))
        baseline[start:start + length] += amp * np.exp(-np.arange(length) / tau)
    return baseline


def _base_rgb(rng: np.random.Generator) -> np.ndarray:
    return np.array([
        rng.uniform(115, 165),  # R
        rng.uniform(90, 130),   # G
        rng.uniform(75, 115),   # B
    ])


def _target_noise(rng: np.random.Generator, snr_db: float, signal_std: float) -> float:
    return float(signal_std / (10 ** (snr_db / 20.0)) + 1e-9)


def _erratic_beats(rng: np.random.Generator) -> np.ndarray:
    """Beat times with HR re-sampled every ~2 s over 45-150 bpm: coherent
    across regions (same beats everywhere) but physiologically impossible."""
    beats: list[float] = []
    tcur = 0.0
    while tcur < DURATION_S:
        block_end = tcur + rng.uniform(1.5, 2.5)
        step = 60.0 / rng.uniform(45.0, 150.0)
        next_beat = tcur + rng.uniform(0.4, 1.0) * step
        while next_beat < block_end:
            beats.append(next_beat)
            next_beat += step
        tcur = block_end
    return np.asarray(beats, dtype=np.float64)


def _sample_snr(rng: np.random.Generator, kind: str, tier: str) -> float:
    if kind == "masked":
        return float(rng.uniform(-9, -3))  # deliberately buried
    lo, hi = TIER_SPECS[tier]["snr"]
    return float(rng.uniform(lo, hi))


def _make_rgb_trace(
    pulse: np.ndarray | None,
    region_delay_s: float,
    gain: float,
    rng: np.random.Generator,
    noise_sigma: float,
    common_noise: np.ndarray,
    base_rgb: np.ndarray,
    illum: np.ndarray,
    motion_common: np.ndarray,
    motion_scale: float = 1.0,
    pulse_scale: float = 1.0,
) -> np.ndarray:
    """Combine pulse + illumination + motion + noise into a (T, 3) RGB trace.

    Sensor noise is split common/regional (70% / 30% of variance): on a
    real recording the pixel-averaged traces of one face share systemic
    noise, so cross-region correlations legitimately run high. Fakes keep
    their independent per-region walk / desync as the discriminating cue."""
    # 70% of head motion is common to all regions (the head really moves);
    # the remaining 30% is regional differential (rotation / occlusion).
    motion = 0.7 * motion_common + 0.3 * _motion_steps(rng, motion_scale)

    if pulse is None:
        # No pulse: slow random walk is all that's structured in the trace.
        walk = np.cumsum(rng.normal(0, 1.0, N_FRAMES))
        walk -= walk.mean()
        walk *= rng.uniform(0.5, 1.5) * pulse_scale / (walk.std() + 1e-9)
        rgb = np.empty((N_FRAMES, 3), dtype=np.float64)
        for c, k in enumerate((1.0, 1.2, 1.5)):
            rgb[:, c] = base_rgb[c] + walk * k + motion
    else:
        shift = int(round(region_delay_s * FPS))
        modulation = np.roll(pulse, shift) if shift else pulse
        modulation = modulation * gain * pulse_scale
        # Green carries the strongest pulsatile change; ~1.2% channel modulation.
        channel_scale = np.array([0.5, 1.0, 0.3])
        rgb = np.empty((N_FRAMES, 3), dtype=np.float64)
        for c in range(3):
            rgb[:, c] = base_rgb[c] * (1.0 + 0.012 * modulation * channel_scale[c]) + motion

    rgb *= illum[:, None]
    rgb += common_noise
    rgb += rng.normal(0.0, noise_sigma * np.sqrt(0.3), rgb.shape)
    np.clip(rgb, 0, None, out=rgb)
    return np.round(rgb)  # quantisation: 8-bit camera output


def simulate_clip(
    is_real: bool, rng: np.random.Generator, fake_kind: str | None = None
) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    """Returns (region_rgb, face_centroids) for one synthetic clip."""
    t = np.arange(N_FRAMES) / FPS
    # Same tier distribution for both classes (masked fakes forced hard)
    # so quality alone cannot predict the label.
    tier = str(rng.choice(TIERS, p=TIER_WEIGHTS))
    if fake_kind == "masked":
        tier = "hard"
    spec = TIER_SPECS[tier]
    illum = _illumination(t, rng, scale=spec["illum"])
    motion_common = _motion_steps(rng, scale=spec["motion"])
    base = _base_rgb(rng)
    snr = _sample_snr(rng, "real" if is_real else "fake", tier)
    kind = fake_kind

    if is_real or kind == "masked":
        beat_times, _ = _beat_times(rng, with_variability=True)
        pulse = _ppg_at(t, beat_times)
        if kind == "masked":
            pulse = pulse * rng.uniform(0.10, 0.25)  # buried below the floor
    elif kind == "desync":
        beat_times, _ = _beat_times(rng, with_variability=True)
        pulse = _ppg_at(t, beat_times)
    elif kind == "erratic":
        pulse = _ppg_at(t, _erratic_beats(rng))
    else:  # independent
        pulse = None

    # Clip-level sensor noise: 70% of variance is COMMON to all regions
    # (systemic / pixel-averaged), 30% regional -- drawn once per clip so
    # genuinely real footage shows the high cross-region correlations the
    # production pipeline measures on real recordings (fixture: ~0.99).
    if pulse is not None:
        rep_std = float(np.std(pulse)) * float(np.mean(base)) * 0.012 * spec["pulse"]
    else:  # independent: random-walk "signal"
        rep_std = 1.5
    noise_sigma = _target_noise(rng, snr, max(rep_std, 1e-6))
    common_noise = rng.normal(0.0, noise_sigma * np.sqrt(0.7), (N_FRAMES, 3))

    region_rgb: dict[str, np.ndarray] = {}
    for i, region in enumerate(REGIONS):
        if kind == "desync":
            delay = rng.uniform(0.3, 1.5)  # beyond the physiological 0.15 s max
        elif is_real or kind == "erratic":
            delay = rng.uniform(0.0, 0.06)  # coherent regions (real or erratic)
        elif pulse is None:
            delay = 0.0
        else:
            delay = rng.uniform(0.0, 0.12)
        gain = rng.uniform(0.88, 1.12) if (is_real or kind == "erratic") else rng.uniform(0.7, 1.4)
        region_rgb[region] = _make_rgb_trace(
            pulse, delay, gain, rng, noise_sigma, common_noise,
            base * rng.uniform(0.92, 1.08, 3), illum,
            motion_common, motion_scale=spec["motion"], pulse_scale=spec["pulse"],
        )

    # Face-centroid random walk -> motion_score, same distribution both classes.
    speed = rng.choice([0.003, 0.006, 0.012, 0.020])
    steps = rng.normal(0, speed, size=(N_FRAMES, 2))
    centroids = 0.5 + np.cumsum(steps, axis=0) - 0.5
    return region_rgb, [c for c in centroids]


def features_for_clip(
    region_rgb: dict[str, np.ndarray], centroids: list[np.ndarray]
) -> np.ndarray:
    """Run the production analysis path on one clip -> flat feature vector."""
    analysis = analyse_regions(
        region_rgb, FPS, method="auto",
        window_seconds=12.0, step_seconds=3.0,
        face_centroids=centroids,
    )
    fv = build_feature_vector(
        analysis.region_results,
        analysis.consistency,
        analysis.motion,
        analysis.illumination,
        extras=analysis.extras,
        gated_bpm=analysis.gating.kept_bpm,
        kept_snr=analysis.gating.kept_snr,
        region_kept_bpm=analysis.gating.region_kept_bpm,
    )
    return fv.as_array()


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def generate_dataset(n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, len(FEATURE_NAMES)), dtype=np.float64)
    y = np.zeros(n_samples, dtype=np.int64)
    t0 = time.time()
    for i in range(n_samples):
        is_real = bool(rng.random() < 0.5)
        kind = None if is_real else str(rng.choice(FAKE_KINDS, p=FAKE_WEIGHTS))
        region_rgb, centroids = simulate_clip(is_real, rng, fake_kind=kind)
        try:
            X[i] = features_for_clip(region_rgb, centroids)
        except Exception as exc:  # noqa: BLE001 -- resimulate on rare numerical failures
            print(f"sample {i} failed ({exc}); resimulating", file=sys.stderr)
            region_rgb, centroids = simulate_clip(is_real, rng, fake_kind=kind)
            X[i] = features_for_clip(region_rgb, centroids)
        y[i] = 1 if is_real else 0
        if (i + 1) % 250 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i + 1}/{n_samples} samples ({rate:.1f}/s)", flush=True)
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="models/production/baseline.joblib")
    parser.add_argument("--results", default="experiments/results/synthetic_rf_results.json")
    args = parser.parse_args()

    print(f"Generating {args.samples} synthetic clips through the production pipeline...")
    X, y = generate_dataset(args.samples, args.seed)
    print(f"Dataset: {X.shape[0]} samples, {int(y.sum())} real / {int((1 - y).sum())} fake")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed
    )
    clf = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    clf.model_version_ = MODEL_VERSION  # read by BaselineClassifier._load

    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "auc": float(roc_auc_score(y_test, proba)),
        "f1": float(f1_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
    }
    # Class-conditional accuracy: we care that weak videos don't all land
    # on one side (that would recreate the UNCERTAIN problem as a bias).
    real_mask = y_test == 1
    fake_mask = ~real_mask
    metrics["real_recall"] = float(pred[real_mask].mean()) if real_mask.any() else 0.0
    metrics["fake_recall"] = float(1 - pred[fake_mask].mean()) if fake_mask.any() else 0.0
    importances = {
        name: round(float(imp), 4)
        for name, imp in sorted(
            zip(FEATURE_NAMES, clf.feature_importances_), key=lambda kv: -kv[1]
        )
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(clf, out_path)

    results = {
        "modelVersion": MODEL_VERSION,
        "samples": args.samples,
        "seed": args.seed,
        "features": FEATURE_NAMES,
        "metrics": metrics,
        "featureImportances": importances,
    }
    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))

    print(json.dumps({"metrics": metrics, "output": str(out_path)}, indent=2))
    print("Top features:", json.dumps(dict(list(importances.items())[:8]), indent=2))
    if metrics["auc"] < 0.9:
        print("WARNING: holdout AUC < 0.9 -- investigate before shipping.", file=sys.stderr)


if __name__ == "__main__":
    main()
