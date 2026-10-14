"""Transparent rPPG cue scores used by the model-free decision layer.

Every cue maps measured features to [0, 1] where 1 = "strongly supports
REAL (physiologically coherent pulse)" and 0 = "supports FAKE". The
weights below are documented in docs/architecture.md; they encode the
literature-backed position that *cross-region coherence of a
band-limited pulsatile signal* is the primary authenticity cue, with
signal strength, temporal stability, waveform morphology and spectral
concentration as supporting cues.

No single cue decides anything. The rPPG decision combines independently
observable signal, region, temporal, morphology, and method-consensus cues.
"""
from __future__ import annotations
from typing import Dict

import numpy as np

# Final rule-score weights (sum to 1.0). Cross-region coherence dominates
# because it is the one cue a synthetic face cannot fake without also
# faking a physiologically consistent pulse everywhere at once; signal
# strength and spectral concentration are weakly discriminative on
# real-world footage and are weighted down accordingly.
WEIGHTS = {
    "coherence": 0.42,
    "method_agreement": 0.08,
    "snr": 0.05,
    "stability": 0.18,
    "morphology": 0.20,
    "spectral": 0.07,
}

# Physiology-implication penalty: when HR behaviour is outside
# physiological limits (see implausible_hr), the rule score is pushed
# down by up to HR_PENALTY. Signal-shape evidence cannot rescue a heart
# rate that jumps 45 -> 150 bpm between consecutive 6s windows.
HR_PENALTY = 0.60


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def coherence_cue(f: Dict[str, float]) -> float:
    """Cross-region synchrony of the pulse: mean correlation, dominant-
    frequency agreement and Hilbert phase stability of one combined."""
    corr = _clip01((f.get("cross_region_mean_correlation", 0.0) - 0.05) / 0.5)
    freq = _clip01(f.get("cross_region_frequency_agreement", 0.0))
    # Measured phase coherence on real footage sits near 0.5 (median for
    # real recordings ~0.52); map that to the upper half rather than
    # reading it as barely-coherent.
    phase = _clip01((f.get("phase_coherence", 0.0) - 0.10) / 0.5)
    return float(np.mean([corr, freq, phase]))


def snr_cue(f: Dict[str, float]) -> float:
    """Kept-window SNR mapped from -8 dB (noise floor) to +6 dB (clean)."""
    snr = f.get("kept_snr_mean", f.get("snr_db_mean", -20.0))
    return _clip01((snr + 8.0) / 14.0)


def stability_cue(f: Dict[str, float]) -> float:
    """Physiologically plausible heart-rate behaviour: close agreement
    between regions and no impossible window-to-window HR swings (a real
    heart cannot jump 45 -> 150 bpm between consecutive windows).

    BPM variability (cv) is deliberately NOT scored here: measured HRV
    does not separate real from fake recordings (both sit near 0.03).
    Implausible variability instead triggers the explicit physiology
    penalty in rule_score.
    """
    # A zero BPM means no temporal estimate. Do not interpret missing
    # measurements as a perfectly stable heart rate.
    if f.get("bpm_mean", 0.0) <= 0.0:
        return 0.5
    spread = 1.0 - _clip01(f.get("region_bpm_spread", 0.0) / 40.0)
    jump = 1.0 - _clip01(f.get("bpm_max_jump", 0.0) / 90.0)
    return float(np.mean([spread, jump]))


def morphology_cue(f: Dict[str, float]) -> float:
    """Pulse-template repeatability and asymmetric (pulse-like) waveform.

    Skew is normalised by 0.5 because real recordings measure a median
    pulse skewness near 0.5; the old divisor of 1.0 read healthy pulses
    as only half-present."""
    morphology_available = f.get(
        "morphology_available",
        float("template_correlation" in f and "pulse_skewness" in f),
    )
    if f.get("bpm_mean", 0.0) <= 0.0 or not morphology_available:
        return 0.5
    template = _clip01((f.get("template_correlation", 0.5) - 0.3) / 0.6)
    skew = _clip01(f.get("pulse_skewness", 0.0) / 0.5)
    return float(np.mean([template, skew]))


def spectral_cue(f: Dict[str, float]) -> float:
    """Energy concentrated in the HR band with cardiac harmonics."""
    spectral_available = f.get(
        "spectral_features_available",
        float("pulse_band_ratio" in f and "spectral_entropy" in f),
    )
    if not spectral_available:
        return 0.5
    band = _clip01(f.get("pulse_band_ratio", 0.0) / 0.6)
    harmonic = _clip01(f.get("harmonic_ratio", 0.0) / 0.3)
    entropy = 1.0 - _clip01((f.get("spectral_entropy", 1.0) - 0.4) / 0.5)
    return float(np.mean([band, harmonic, entropy]))


def method_agreement_cue(f: Dict[str, float]) -> float:
    """Agreement in dominant pulse frequency among GREEN/POS/CHROM.

    This is supportive evidence only: all methods observe the same pixels,
    so agreement is not treated as three statistically independent votes.
    """
    if f.get("available_method_count", 0.0) < 2.0:
        return 0.5  # unobserved, neutral rather than a negative vote
    return _clip01(f.get("method_frequency_agreement", 0.5))


def implausible_hr(f: Dict[str, float]) -> float:
    """0..1 measure of non-physiological heart-rate behaviour.

    A real heart does not jump 45 -> 150 bpm between consecutive 6s
    windows, nor sustain cv > 0.12 at rest. Deepfakes built from
    independently regenerated pulse traces typically do both, even when
    every other cue (correlation, morphology, spectrum) looks real."""
    if f.get("bpm_mean", 0.0) <= 0.0:
        return 0.0
    cv_term = _clip01((f.get("bpm_cv", 0.0) - 0.12) / 0.15)
    jump_term = _clip01((f.get("bpm_max_jump", 0.0) - 30.0) / 60.0)
    return float(max(cv_term, jump_term))


def cue_scores(f: Dict[str, float]) -> Dict[str, float]:
    return {
        "coherence": coherence_cue(f),
        "method_agreement": method_agreement_cue(f),
        "snr": snr_cue(f),
        "stability": stability_cue(f),
        "morphology": morphology_cue(f),
        "spectral": spectral_cue(f),
    }


def rule_score(f: Dict[str, float]) -> tuple[float, Dict[str, float]]:
    """Weighted fusion of the rule cues -> P(REAL) in (0, 1).

    Beyond 0.5 of implausible HR behaviour the physiology penalty scales
    with how extreme the violation is; clip at 0 so a pathological HR
    record cannot push the score negative."""
    scores = cue_scores(f)
    total = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)
    total -= HR_PENALTY * max(0.0, implausible_hr(f) - 0.5)
    # Stretch: cue midpoints (~0.5) are deliberately mapped near 0.5 so an
    # uninformative feature set scores ~0.5 rather than looking fake-ish.
    return float(np.clip(total, 0.0, 1.0)), scores
