"""Quality gating: garbage windows must be rejected and counted, not
averaged into the clip-level features."""
from app.features.hr_features import RegionHRResult
from app.features.time_domain import TemporalFeatures
from app.rppg.quality import SignalQuality
from app.pipeline.quality_gating import (
    SNR_FLOOR_DB,
    GatingReport,
    gate_windows,
    gated_temporal,
)


def _res(region: str, snrs: float, bpms: float, band_ratios=None) -> RegionHRResult:
    n = len(snrs) if isinstance(snrs, (list, tuple)) else 1
    snrs = [snrs] if not isinstance(snrs, (list, tuple)) else list(snrs)
    bpms = [bpms] if not isinstance(bpms, (list, tuple)) else list(bpms)
    band_ratios = band_ratios or [0.2] * len(snrs)
    return RegionHRResult(
        region=region,
        window_bpm=list(bpms),
        window_quality=[
            SignalQuality(snr_db=s, quality_score=50.0, dominant_freq_hz=b / 60.0,
                          band_ratio=r)
            for s, b, r in zip(snrs, bpms, band_ratios)
        ],
        temporal=TemporalFeatures(0, 0, 0, 0),
        full_spectrum=None,
        mean_quality_score=50.0,
    )


def test_low_snr_windows_are_rejected():
    results = {
        "forehead": _res("forehead", [-10.0, 2.0, 8.0], [70, 72, 71]),
        "left_cheek": _res("left_cheek", [5.0, -12.0], [68, 90]),
    }
    report = gate_windows(results)
    assert report.total_windows == 5
    assert report.kept_windows == 3
    assert report.kept_ratio == 0.6
    assert report.rejected_reasons.get("snr") == 2
    assert all(s > SNR_FLOOR_DB for s in report.kept_snr)
    assert report.window_kept["forehead"] == [False, True, True]
    assert report.region_kept_bpm["forehead"] == [72, 71]
    assert report.to_dict()["keptRatio"] == 0.6


def test_implausible_bpm_and_tiny_band_ratio_rejected():
    results = {
        "forehead": _res(
            "forehead",
            [5.0, 5.0, 5.0],
            [15.0, 200.0, 72.0],
            band_ratios=[0.2, 0.2, 0.001],
        ),
    }
    report = gate_windows(results)
    # 15 bpm and 200 bpm out of range -> bpm_implausible; 72 bpm kept but
    # band_ratio 0.001 < min -> band_ratio rejection.
    assert report.kept_windows == 0
    assert report.kept_ratio == 0.0
    assert set(report.rejected_reasons) == {"bpm_implausible", "band_ratio"}


def test_gated_temporal_empty_is_zero():
    assert gated_temporal([]) == (0.0, 0.0, 0.0, 0.0)
    mean, std, cv, jump = gated_temporal([70.0, 72.0, 71.0])
    assert abs(mean - 71.0) < 1e-9
    assert cv < 0.02
    assert jump == 2.0


def test_empty_gating_report_is_safe():
    report = gate_windows({})
    assert report.total_windows == 0
    assert report.kept_ratio == 0.0
    assert report.kept_snr_mean == -20.0
