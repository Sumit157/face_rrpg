"""Generates human-readable evidence/warning strings from the computed
features. This is the layer that keeps the system honest: it must never
describe a low-quality signal as strong evidence, and it must always
separate 'what we measured' from 'what the model predicted'.
"""
from __future__ import annotations
from typing import Dict, List


QUALITY_GOOD = 65
QUALITY_MARGINAL = 40


def build_evidence_and_warnings(
    feature_values: Dict[str, float],
    prediction_label: str,
) -> tuple[List[str], List[str]]:
    evidence: List[str] = []
    warnings: List[str] = []

    quality = feature_values.get("signal_quality_mean", 0.0)
    corr = feature_values.get("cross_region_mean_correlation", 0.0)
    freq_agree = feature_values.get("cross_region_frequency_agreement", 0.0)
    bpm_cv = feature_values.get("bpm_cv", 0.0)
    motion = feature_values.get("motion_score", 0.0)
    illumination = feature_values.get("illumination_stability", 100.0)
    bpm_spread = feature_values.get("region_bpm_spread", 0.0)
    harmonics = feature_values.get("harmonic_ratio", 0.0)
    phase_coh = feature_values.get("phase_coherence", 0.0)
    amp_cv = feature_values.get("amplitude_cv", 0.0)

    # --- Signal quality gate: always stated first, always honest ---
    if quality < QUALITY_MARGINAL:
        warnings.append(
            "Signal quality is low; rPPG evidence below is unreliable and the "
            "prediction should not be treated as strong evidence."
        )
    elif quality < QUALITY_GOOD:
        warnings.append("Signal quality is moderate; treat evidence with caution.")
    else:
        evidence.append("Strong, stable pulsatile signal (good signal-to-noise ratio).")

    # --- Cross-region consistency ---
    if quality >= QUALITY_MARGINAL:
        if corr >= 0.5:
            evidence.append("Strong region agreement across forehead and cheeks.")
        elif corr <= 0.15:
            warnings.append("Weak correlation between facial regions.")

        if freq_agree >= 0.66:
            evidence.append("Consistent dominant frequency across regions.")
        elif freq_agree <= 0.34:
            warnings.append("Regions disagree on dominant heart-rate frequency.")

    # --- Temporal stability ---
    if quality >= QUALITY_MARGINAL:
        if bpm_cv < 0.08:
            evidence.append("Stable BPM across analysis windows.")
        elif bpm_cv > 0.25:
            warnings.append("BPM estimate fluctuates substantially between windows.")

    if bpm_spread > 15:
        warnings.append("Large BPM disagreement between facial regions.")

    # --- Harmonic structure / phase coherence / amplitude consistency ---
    if quality >= QUALITY_MARGINAL:
        if harmonics >= 0.08:
            evidence.append("Cardiac harmonics present (2nd/3rd harmonic energy detected).")
        elif harmonics < 0.01:
            warnings.append("No harmonic structure detected; pulse waveform looks unnaturally simple.")

        if phase_coh >= 0.7:
            evidence.append("Regions stay phase-locked over time (stable pulse-transit timing).")
        elif phase_coh < 0.4:
            warnings.append("Phase relationship between regions drifts over the clip.")

        if amp_cv > 0.6:
            warnings.append("Pulse amplitude varies inconsistently across facial regions.")

    # --- Confounds that affect reliability regardless of authenticity ---
    if motion > 60:
        warnings.append("Moderate-to-high head motion detected; may reduce reliability.")
    if illumination < 50:
        warnings.append("Unstable lighting detected during the video.")

    if prediction_label == "UNCERTAIN":
        warnings.append(
            "The physiological evidence does not clearly favour REAL or FAKE; "
            "this should be treated as inconclusive, not as a negative result."
        )

    return evidence, warnings
