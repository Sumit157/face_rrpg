"""Decision-fusion behaviour: decisive Real/Fake on judgeable clips,
UNCERTAIN only on extremely poor videos, explanations always present.
"""
import pytest

from app.pipeline.decision import decide, measurement_strength

# In-distribution clean-real feature set (values stay inside the
# synthetic real-class p5-p95 ranges so the trained RF agrees).
CLEAN_REAL = {
    "bpm_mean": 90.0,
    "bpm_std": 5.59,
    "bpm_cv": 0.0603,
    "bpm_max_jump": 15.0,
    "signal_quality_mean": 30.0,
    "snr_db_mean": -2.5,
    "kept_snr_mean": -2.5,
    "cross_region_mean_correlation": 0.7,
    "cross_region_frequency_agreement": 1.0,
    "motion_score": 40.0,
    "illumination_stability": 75.0,
    "region_bpm_spread": 7.5,
    "harmonic_ratio": 0.63,
    "phase_coherence": 0.42,
    "amplitude_cv": 0.139,
    "pulse_band_ratio": 0.98,
    "spectral_entropy": 0.86,
    "lr_cheek_correlation": 0.7,
    "template_correlation": 0.83,
    "pulse_skewness": 0.55,
    "kept_window_ratio": 1.0,
}

# Measurable signal, but regions share no pulse -- the deepfake signature.
NO_PULSE = {
    **CLEAN_REAL,
    "signal_quality_mean": 28.0,
    "cross_region_mean_correlation": 0.05,
    "cross_region_frequency_agreement": 0.0,
    "phase_coherence": 0.15,
    "lr_cheek_correlation": -0.05,
    "template_correlation": 0.45,
    "pulse_skewness": 0.05,
    "harmonic_ratio": 0.05,
    "spectral_entropy": 0.91,
    "amplitude_cv": 0.45,
    "region_bpm_spread": 30.0,
    "bpm_cv": 0.0,
    "bpm_std": 0.0,
    "bpm_max_jump": 0.0,
}


def test_clean_real_features_decisively_real():
    d = decide(dict(CLEAN_REAL), detection_rate=0.95)
    assert d.label == "REAL"
    assert d.confidence >= 0.70
    assert d.probability_real > 0.5
    assert d.explanation
    assert "Real" in d.explanation


def test_incoherent_measurable_features_decisively_fake():
    d = decide(dict(NO_PULSE), detection_rate=0.95)
    assert d.label == "FAKE"
    assert d.confidence >= 0.65
    assert d.probability_real < 0.5
    assert "Fake" in d.explanation
    assert "coherent" in d.explanation or "correlation" in d.explanation


def test_all_zero_features_uncertain_at_half_probability():
    zeros = {k: 0.0 for k in CLEAN_REAL}
    d = decide(zeros, detection_rate=1.0)
    assert d.label == "UNCERTAIN"
    assert d.probability_real == pytest.approx(0.5)
    assert d.uncertain_reason


def test_extreme_poor_snr_motion_lighting_uncertain():
    extreme = {
        **CLEAN_REAL,
        "signal_quality_mean": 20.0,
        "snr_db_mean": -8.0,
        "kept_snr_mean": -8.0,
        "kept_window_ratio": 0.1,
        "motion_score": 80.0,
        "illumination_stability": 30.0,
    }
    d = decide(extreme, detection_rate=0.5)
    assert d.label == "UNCERTAIN"


def test_low_quality_but_measurable_still_forces_a_decision():
    # Poor-ish clip (the WhatsApp scenario): not EXTREME, so no uncertainty.
    weak = {
        **CLEAN_REAL,
        "signal_quality_mean": 22.0,
        "kept_snr_mean": -4.5,
        "snr_db_mean": -4.5,
        "kept_window_ratio": 0.6,
        "cross_region_frequency_agreement": 1.0,
        "phase_coherence": 0.42,
        "lr_cheek_correlation": 0.5,
        "motion_score": 50.0,
        "illumination_stability": 40.0,
    }
    d = decide(weak, detection_rate=0.8)
    assert d.label in ("REAL", "FAKE")
    assert 0.0 < d.confidence <= 1.0
    assert d.explanation


def test_no_face_detected_is_uncertain():
    d = decide(dict(CLEAN_REAL), detection_rate=0.1)
    assert d.label == "UNCERTAIN"


def test_measurement_strength_bounds():
    assert measurement_strength({**CLEAN_REAL}, 0.95) >= 0.79
    assert measurement_strength({k: 0.0 for k in CLEAN_REAL}, 0.1) < 0.15


def test_confidence_never_overconfident_when_strength_low():
    # Same strong cues, but nearly unusable clip (kept 10%, quality 25,
    # detection 0.5) -> still decidable but clearly discounted.
    degraded = {
        **CLEAN_REAL,
        "signal_quality_mean": 25.0,
        "kept_window_ratio": 0.1,
        "illumination_stability": 35.0,
        "motion_score": 55.0,
        "kept_snr_mean": -3.0,
        "snr_db_mean": -3.0,
    }
    d = decide(degraded, detection_rate=0.5)
    assert d.label in ("REAL", "FAKE")  # not extreme -> forced decision
    assert d.confidence < 0.85
    assert 0 < d.measurement_strength < 0.6
