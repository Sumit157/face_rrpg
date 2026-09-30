"""Transparent rule-based cue scores shared by the decision fusion and
the heuristic fallback classifier.

Every cue maps measured features to [0, 1] where 1 = "strongly supports
REAL (physiologically coherent pulse)" and 0 = "supports FAKE". The
weights below are documented in docs/architecture.md; they encode the
literature-backed position that *cross-region coherence of a
band-limited pulsatile signal* is the primary authenticity cue, with
signal strength, temporal stability, waveform morphology and spectral
concentration as supporting cues.

No single cue decides anything -- the fusion in app.pipeline.decision
combines these with the trained classifier and only then labels the clip.
"""
from __future__ import annotations
from typing import Dict

import numpy as np

# Final rule-score weights (sum to 1.0).
WEIGHTS = {
    "coherence": 0.30,
    "snr": 0.20,
    "stability": 0.15,
    "morphology": 0.20,
    "spectral": 0.15,
}


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def coherence_cue(f: Dict[str, float]) -> float:
    """Cross-region synchrony of the pulse: mean correlation, dominant-
    frequency agreement and Hilbert phase stability of one combined."""
    corr = _clip01((f.get("cross_region_mean_correlation", 0.0) - 0.05) / 0.5)
    freq = _clip01(f.get("cross_region_frequency_agreement", 0.0))
    phase = _clip01((f.get("phase_coherence", 0.0) - 0.25) / 0.5)
    return float(np.mean([corr, freq, phase]))


def snr_cue(f: Dict[str, float]) -> float:
    """Kept-window SNR mapped from -8 dB (noise floor) to +6 dB (clean)."""
    snr = f.get("kept_snr_mean", f.get("snr_db_mean", -20.0))
    return _clip01((snr + 8.0) / 14.0)


def stability_cue(f: Dict[str, float]) -> float:
    """Physiologically plausible heart-rate behaviour: low relative
    variability across windows, close agreement between regions, and no
    impossible window-to-window HR swings (a real heart cannot jump
    45 -> 150 bpm between consecutive windows)."""
    cv = 1.0 - _clip01(f.get("bpm_cv", 0.0) / 0.25)
    spread = 1.0 - _clip01(f.get("region_bpm_spread", 0.0) / 40.0)
    jump = 1.0 - _clip01(f.get("bpm_max_jump", 0.0) / 90.0)
    return float(np.mean([cv, spread, jump]))


def morphology_cue(f: Dict[str, float]) -> float:
    """Pulse-template repeatability and asymmetric (pulse-like) waveform."""
    template = _clip01((f.get("template_correlation", 0.5) - 0.3) / 0.6)
    skew = _clip01(f.get("pulse_skewness", 0.0) / 1.0)
    return float(np.mean([template, skew]))


def spectral_cue(f: Dict[str, float]) -> float:
    """Energy concentrated in the HR band with cardiac harmonics."""
    band = _clip01(f.get("pulse_band_ratio", 0.0) / 0.6)
    harmonic = _clip01(f.get("harmonic_ratio", 0.0) / 0.3)
    entropy = 1.0 - _clip01((f.get("spectral_entropy", 1.0) - 0.4) / 0.5)
    return float(np.mean([band, harmonic, entropy]))


def cue_scores(f: Dict[str, float]) -> Dict[str, float]:
    return {
        "coherence": coherence_cue(f),
        "snr": snr_cue(f),
        "stability": stability_cue(f),
        "morphology": morphology_cue(f),
        "spectral": spectral_cue(f),
    }


def rule_score(f: Dict[str, float]) -> tuple[float, Dict[str, float]]:
    """Weighted fusion of the rule cues -> P(REAL) in (0, 1)."""
    scores = cue_scores(f)
    total = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)
    # Stretch: cue midpoints (~0.5) are deliberately mapped near 0.5 so an
    # uninformative feature set scores ~0.5 rather than looking fake-ish.
    return float(np.clip(total, 0.0, 1.0)), scores
