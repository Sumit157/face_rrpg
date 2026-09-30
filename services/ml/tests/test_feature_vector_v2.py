"""Feature-vector v2: extended feature set + quality-gated aggregation."""
import numpy as np

from app.features.feature_vector import FEATURE_NAMES, build_feature_vector


class _StubTemporal:
    bpm_mean = 70.0
    bpm_std = 1.0
    bpm_cv = 0.01
    bpm_max_jump = 2.0


class _StubRegionResult:
    temporal = _StubTemporal()
    mean_quality_score = 55.0
    window_quality = []


class _StubConsistency:
    mean_correlation = 0.5
    frequency_agreement = 1.0


class _StubMotion:
    motion_score = 30.0


class _StubIllum:
    stability_score = 70.0


def test_feature_names_cover_v2_features():
    assert len(FEATURE_NAMES) == 20
    for name in (
        "pulse_band_ratio",
        "spectral_entropy",
        "lr_cheek_correlation",
        "template_correlation",
        "pulse_skewness",
        "kept_window_ratio",
    ):
        assert name in FEATURE_NAMES


def test_gated_aggregation_uses_only_kept_windows():
    results = {"forehead": _StubRegionResult(), "left_cheek": _StubRegionResult()}
    extras = {
        "kept_window_ratio": 0.75,
        "pulse_band_ratio": 0.6,
        "spectral_entropy": 0.4,
        "lr_cheek_correlation": 0.55,
        "template_correlation": 0.7,
        "pulse_skewness": 0.8,
        "phase_coherence": 0.8,
        "harmonic_ratio": 0.2,
        "amplitude_cv": 0.2,
    }
    fv = build_feature_vector(
        results,
        _StubConsistency(),
        _StubMotion(),
        _StubIllum(),
        extras=extras,
        gated_bpm=[70.0, 71.0, 73.0],
        kept_snr=[5.0, 6.0],
        region_kept_bpm={"forehead": [70.0, 71.0], "left_cheek": [73.0]},
    )
    values = fv.to_dict()
    arr = fv.as_array()
    assert len(arr) == len(FEATURE_NAMES)
    assert not np.isnan(arr).any()
    assert abs(values["bpm_mean"] - float(np.mean([70.0, 71.0, 73.0]))) < 1e-5
    assert values["snr_db_mean"] == 5.5  # kept windows only
    assert values["kept_window_ratio"] == 0.75
    assert values["lr_cheek_correlation"] == 0.55
    # spread across regions: max(70.5, 73) - min -> 73 - 70.5 = 2.5
    assert abs(values["region_bpm_spread"] - 2.5) < 1e-9


def test_ungated_path_still_works_for_tests():
    results = {"forehead": _StubRegionResult()}
    fv = build_feature_vector(results, _StubConsistency(), _StubMotion(), _StubIllum())
    values = fv.to_dict()
    assert values["bpm_mean"] == 70.0
    assert values["kept_window_ratio"] == 0.0  # no gating info -> default 0
    assert values["template_correlation"] == 0.5  # neutral default
