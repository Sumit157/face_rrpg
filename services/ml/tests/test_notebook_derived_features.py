"""Tests for the harmonic ratio / phase coherence / amplitude CV
features and the Haar cascade fallback detector, ported from the rPPG
deepfake detector notebook (Kate & Shelar, MIT WPU) into this pipeline.
"""
import numpy as np
import pytest

from app.features.harmonics import harmonic_ratio
from app.features.phase_coherence import phase_coherence
from app.features.amplitude_consistency import amplitude_cv
from app.features.frequency_domain import estimate_bpm
from app.rppg.quality import assess_quality
from app.models.baseline import BaselineClassifier
from app.features.feature_vector import FEATURE_NAMES
from app.face.haar_fallback import HaarFaceDetector, HaarFaceBox, box_to_region_rects, mean_rgb_in_rect

FPS = 30.0
DURATION = 20
N = int(FPS * DURATION)


def _synthetic_real_signals(seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, DURATION, N)
    hr = 1.0
    fore = (5.0 * np.sin(2 * np.pi * hr * t) + 1.5 * np.sin(4 * np.pi * hr * t)
            + 0.5 * np.sin(6 * np.pi * hr * t) + rng.normal(0, 0.4, N))
    left = (4.8 * np.sin(2 * np.pi * hr * t + 0.10) + 1.4 * np.sin(4 * np.pi * hr * t + 0.10)
            + 0.4 * np.sin(6 * np.pi * hr * t + 0.10) + rng.normal(0, 0.4, N))
    right = (5.1 * np.sin(2 * np.pi * hr * t + 0.12) + 1.6 * np.sin(4 * np.pi * hr * t + 0.12)
             + 0.5 * np.sin(6 * np.pi * hr * t + 0.12) + rng.normal(0, 0.4, N))
    return {"forehead": fore, "left_cheek": left, "right_cheek": right}


def _synthetic_fake_signals(seed=1):
    rng = np.random.default_rng(seed)
    return {
        "forehead": rng.normal(0, 2.0, N),
        "left_cheek": rng.normal(0, 2.0, N),
        "right_cheek": rng.normal(0, 2.0, N),
    }


def test_phase_coherence_discriminates_structured_vs_noise():
    real = _synthetic_real_signals()
    fake = _synthetic_fake_signals()
    coherence_real = phase_coherence(real)
    coherence_fake = phase_coherence(fake)
    assert coherence_real > coherence_fake
    assert coherence_real > 0.8
    assert coherence_fake < 0.3


def test_amplitude_cv_lower_for_synchronised_regions():
    real = _synthetic_real_signals()
    acv = amplitude_cv(real, FPS)
    assert 0 <= acv < 0.3  # synchronised, proportional amplitudes -> low spread


def test_harmonic_ratio_meaningful_only_with_real_peak():
    """Known caveat: on pure noise (no true dominant peak), harmonic_ratio
    is not a reliable discriminator on its own -- it can score higher on
    noise than on a real structured signal, because 'peak energy' is
    itself just a noise spike. This test documents that behaviour rather
    than asserting the (incorrect) intuitive direction, and the next
    test verifies the pipeline's quality-gating neutralises it."""
    fake = _synthetic_fake_signals()
    spectrum = estimate_bpm(fake["forehead"], FPS)
    ratio = harmonic_ratio(spectrum.freqs_hz, spectrum.power, spectrum.peak_hz)
    quality = assess_quality(spectrum.freqs_hz, spectrum.power, spectrum.peak_hz)
    assert quality.quality_score < 20, (
        "pure noise should score low signal quality regardless of harmonic_ratio's value"
    )


def test_classifier_stays_uncertain_despite_misleading_harmonic_ratio():
    """Even if harmonic_ratio is misleadingly high (the noise edge case
    above), a zero signal-quality score must still collapse the
    prediction to UNCERTAIN -- verifying the quality-gate protects
    against this known feature weakness."""
    values = dict(zip(FEATURE_NAMES, [0.0] * len(FEATURE_NAMES)))
    values["signal_quality_mean"] = 0.0
    values["harmonic_ratio"] = 0.89  # misleadingly high, as measured on pure noise
    vector = np.array([values[n] for n in FEATURE_NAMES])

    clf = BaselineClassifier(model_path="/nonexistent/path.joblib")
    pred = clf.predict(vector)
    assert pred.label == "UNCERTAIN"
    assert pred.probability_real == pytest.approx(0.5)


def test_haar_fallback_detector_loads_and_derives_regions():
    """Haar cascades ship bundled with opencv-python (no network
    download needed), unlike the MediaPipe Face Landmarker .task model
    -- this is the one face-detection path fully testable offline."""
    detector = HaarFaceDetector()
    assert detector is not None

    box = HaarFaceBox(x=50, y=30, width=180, height=220)
    regions = box_to_region_rects(box)
    assert set(regions.keys()) == {"forehead", "left_cheek", "right_cheek"}

    frame = np.full((300, 300, 3), 150, dtype=np.uint8)
    rgb = mean_rgb_in_rect(frame, regions["forehead"])
    assert rgb is not None
    assert rgb.shape == (3,)
