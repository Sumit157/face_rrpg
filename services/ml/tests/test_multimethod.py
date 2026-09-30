"""Multi-method rPPG: parallel GREEN/CHROM/POS extraction and best-SNR
selection when method == "auto"."""
import numpy as np

from app.rppg.multimethod import CANDIDATE_METHODS, analyse_methods, extract_method_signals

FPS = 30.0


def _rgb(n=480, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FPS
    pulse = 3.0 * np.sin(2 * np.pi * 1.2 * t)
    return np.stack([
        150 + 0.5 * pulse + rng.normal(0, 0.5, n),
        150 + 1.0 * pulse + rng.normal(0, 0.5, n),
        150 + 0.3 * pulse + rng.normal(0, 0.5, n),
    ], axis=1)


def test_extract_auto_returns_all_candidate_methods():
    signals = extract_method_signals(_rgb(), FPS, "auto")
    assert set(signals.keys()) == set(CANDIDATE_METHODS)
    assert all(len(s) == 480 for s in signals.values())


def test_extract_explicit_method_returns_only_that_method():
    signals = extract_method_signals(_rgb(), FPS, "pos")
    assert set(signals.keys()) == {"pos"}


def test_analyse_methods_selects_the_clean_pulse_method():
    rng = np.random.default_rng(1)
    t = np.arange(480) / FPS
    clean = np.sin(2 * np.pi * 1.3 * t) + rng.normal(0, 0.05, 480)
    noisy = rng.normal(0, 2.0, 480)
    region_signals = {
        "forehead": {"green": clean, "chrom": noisy, "pos": noisy},
        "left_cheek": {"green": clean, "chrom": noisy, "pos": noisy},
    }
    results, selection = analyse_methods(
        region_signals, FPS, window_seconds=8, step_seconds=2, requested_method="auto"
    )
    assert selection.selected == "green"
    assert selection.mean_quality["green"] > selection.mean_quality["chrom"]
    # All regions returned use the SAME selected method.
    assert set(results) == {"forehead", "left_cheek"}


def test_explicit_method_selection_is_passthrough():
    noisy = np.random.default_rng(2).normal(0, 1.0, 480)
    region_signals = {"forehead": {"pos": noisy}}
    results, selection = analyse_methods(
        region_signals, FPS, window_seconds=8, step_seconds=2, requested_method="pos"
    )
    assert selection.selected == "pos"
    assert set(results) == {"forehead"}
