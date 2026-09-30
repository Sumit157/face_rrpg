"""Verifies the rPPG math recovers a known synthetic heart rate and that
signal-quality / consistency / classifier / explainability behave
sensibly. Uses synthetic RGB traces so it runs with no video/model
assets and no network access -- see test_pipeline_integration_stub.py
for the version exercising real video I/O and ROI extraction.
"""
import numpy as np
import pytest

from app.rppg.green import green_signal
from app.rppg.chrom import chrom_signal
from app.rppg.pos import pos_signal
from app.rppg.filtering import preprocess
from app.features.frequency_domain import estimate_bpm
from app.rppg.quality import assess_quality
from app.features.hr_features import analyse_region
from app.features.consistency import cross_region_consistency
from app.features.motion import score_motion
from app.features.illumination import score_illumination
from app.features.feature_vector import build_feature_vector
from app.models.baseline import BaselineClassifier
from app.explainability.evidence import build_evidence_and_warnings
from app.explainability.confidence import calibrate

FPS = 30.0
TRUE_BPM = 72.0


def make_synthetic_rgb(fps, duration_s, true_bpm, noise_level=0.5, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1 / fps)
    f_hr = true_bpm / 60.0
    pulse = 1.2 * np.sin(2 * np.pi * f_hr * t) + 0.3 * np.sin(2 * np.pi * 2 * f_hr * t)
    slow_drift = 2.0 * np.sin(2 * np.pi * 0.05 * t)
    noise = rng.normal(0, noise_level, len(t))
    green = 150 + 1.0 * pulse + slow_drift + noise
    red = 150 + 0.5 * pulse + slow_drift + noise * 1.1
    blue = 150 + 0.3 * pulse + slow_drift + noise * 1.2
    return np.stack([red, green, blue], axis=1)


@pytest.mark.parametrize("method_name,fn", [
    ("green", lambda rgb, fps: green_signal(rgb)),
    ("chrom", lambda rgb, fps: chrom_signal(rgb)),
    ("pos", lambda rgb, fps: pos_signal(rgb, fps)),
])
def test_method_recovers_known_bpm(method_name, fn):
    rgb = make_synthetic_rgb(FPS, 20, TRUE_BPM, noise_level=0.3)
    processed = preprocess(fn(rgb, FPS), FPS)
    spectrum = estimate_bpm(processed, FPS)
    assert abs(spectrum.bpm - TRUE_BPM) < 8, f"{method_name} BPM error too high"


def test_noisier_signal_still_estimates_bpm():
    rgb = make_synthetic_rgb(FPS, 20, TRUE_BPM, noise_level=6.0)
    processed = preprocess(pos_signal(rgb, FPS), FPS)
    spectrum = estimate_bpm(processed, FPS)
    quality = assess_quality(spectrum.freqs_hz, spectrum.power, spectrum.peak_hz)
    assert spectrum.bpm > 0
    assert 0 <= quality.quality_score <= 100


def test_synchronised_regions_show_high_consistency():
    region_rgb = {
        "forehead": make_synthetic_rgb(FPS, 20, TRUE_BPM, noise_level=0.4, seed=1),
        "left_cheek": make_synthetic_rgb(FPS, 20, TRUE_BPM, noise_level=0.6, seed=2),
        "right_cheek": make_synthetic_rgb(FPS, 20, TRUE_BPM, noise_level=0.6, seed=3),
    }
    region_signals = {r: preprocess(pos_signal(rgb, FPS), FPS) for r, rgb in region_rgb.items()}
    region_results = {
        r: analyse_region(r, sig, FPS, window_seconds=8, step_seconds=2)
        for r, sig in region_signals.items()
    }
    peak_hz = {r: res.full_spectrum.peak_hz for r, res in region_results.items()}
    consistency = cross_region_consistency(region_signals, peak_hz)
    assert consistency.mean_correlation > 0.3

    motion = score_motion([np.array([0.5, 0.5])] * 200)
    illum = score_illumination(region_rgb["forehead"])
    fv = build_feature_vector(region_results, consistency, motion, illum)

    clf = BaselineClassifier(model_path="/nonexistent/path.joblib")
    pred = clf.predict(fv.as_array())
    assert pred.label in ("REAL", "FAKE", "UNCERTAIN")
    assert pred.is_heuristic_fallback is True

    conf = calibrate(pred.probability_real, fv.values["signal_quality_mean"])
    assert 0 <= conf.adjusted_confidence <= 1

    evidence, warnings = build_evidence_and_warnings(fv.values, pred.label)
    assert isinstance(evidence, list) and isinstance(warnings, list)
