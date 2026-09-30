"""Generates human-readable evidence/warning strings from the computed
features. This is the layer that keeps the system honest: it must never
describe a low-quality signal as strong evidence, and it must always
separate 'what we measured' from 'what the model predicted'.

Evidence = measurements that SUPPORT the stated verdict.
Warnings = confounds/caveats that affect reliability.
"""
from __future__ import annotations
from typing import Dict, List

from app.pipeline.decision import Decision


QUALITY_GOOD = 65
QUALITY_MARGINAL = 40


def build_evidence_and_warnings(
    feature_values: Dict[str, float],
    decision: Decision,
) -> tuple[List[str], List[str]]:
    evidence: List[str] = []
    warnings: List[str] = []

    quality = feature_values.get("signal_quality_mean", 0.0)
    corr = feature_values.get("cross_region_mean_correlation", 0.0)
    lr_corr = feature_values.get("lr_cheek_correlation", 0.0)
    freq_agree = feature_values.get("cross_region_frequency_agreement", 0.0)
    bpm_cv = feature_values.get("bpm_cv", 0.0)
    motion = feature_values.get("motion_score", 0.0)
    illumination = feature_values.get("illumination_stability", 100.0)
    bpm_spread = feature_values.get("region_bpm_spread", 0.0)
    harmonics = feature_values.get("harmonic_ratio", 0.0)
    phase_coh = feature_values.get("phase_coherence", 0.0)
    amp_cv = feature_values.get("amplitude_cv", 0.0)
    kept_ratio = feature_values.get("kept_window_ratio", 1.0)
    template = feature_values.get("template_correlation", 0.5)
    label = decision.label

    # --- Signal quality gate: always stated first, always honest ---
    if quality < QUALITY_MARGINAL:
        warnings.append(
            "Signal quality is low; rPPG evidence below is less reliable and "
            "confidence has been discounted accordingly."
        )
    elif quality < QUALITY_GOOD:
        warnings.append("Signal quality is moderate; treat evidence with caution.")
    else:
        evidence.append("Strong, stable pulsatile signal (good signal-to-noise ratio).")

    if kept_ratio < 1.0:
        warnings.append(
            f"{round((1 - kept_ratio) * 100)}% of analysis windows were rejected by "
            "quality gating (low SNR / implausible peak)."
        )

    # --- Verdict-supporting measurements ---
    if label == "REAL":
        if freq_agree >= 0.66:
            evidence.append("Consistent dominant frequency across facial regions.")
        if lr_corr >= 0.3:
            evidence.append(f"Left and right cheeks pulse in sync (r={lr_corr:+.2f}).")
        if phase_coh >= 0.6:
            evidence.append("Regions stay phase-locked over time (stable pulse-transit timing).")
        if bpm_cv < 0.08:
            evidence.append("Stable BPM across analysis windows.")
        if harmonics >= 0.08:
            evidence.append("Cardiac harmonics present (2nd/3rd harmonic energy detected).")
        if template >= 0.6:
            evidence.append("Repeatable pulse waveform morphology (consistent cycle shape).")
        if corr >= 0.4:
            evidence.append("Strong correlation between facial regions.")
    elif label == "FAKE":
        if corr <= 0.15:
            evidence.append(f"Facial regions show no pulse correlation (r={corr:+.2f}).")
        if freq_agree <= 0.34:
            evidence.append("Regions disagree on dominant frequency (no shared pulse).")
        if lr_corr < 0.15:
            evidence.append(f"Left/right cheeks lack bilateral synchrony (r={lr_corr:+.2f}).")
        if phase_coh < 0.4:
            evidence.append("Phase relationship between regions drifts (not one shared pulse).")
        if harmonics < 0.03 and quality >= QUALITY_MARGINAL:
            evidence.append("No cardiac harmonic structure despite measurable signal.")
        if bpm_spread > 20:
            evidence.append(f"Regions estimate very different heart rates (spread {bpm_spread:.0f} bpm).")
        if amp_cv > 0.7:
            evidence.append("Pulse amplitude is inconsistent across regions.")
        if not evidence:
            evidence.append("Measured features lack the coherence pattern of a real pulse.")
    else:  # UNCERTAIN
        evidence.append(
            f"Uncertain: {decision.uncertain_reason or 'insufficient measurable signal'}."
        )
        warnings.append(
            "The video is below the measurability threshold; this is an "
            "inconclusive result, not a negative one."
        )

    # --- Contrary measurements (honesty: show what argues against too) ---
    if label == "FAKE" and quality >= QUALITY_GOOD and freq_agree >= 0.66:
        warnings.append(
            "Some regions do agree on frequency; verdict leans on the "
            "remaining incoherence cues."
        )
    if label == "REAL" and corr < 0.15 and quality >= QUALITY_MARGINAL:
        warnings.append("Overall cross-region correlation is weaker than typical.")

    # --- Confounds that affect reliability regardless of authenticity ---
    if motion > 70:
        warnings.append("High head motion detected; results are less reliable.")
    if illumination < 50:
        warnings.append("Unstable lighting detected during the video.")
    if decision.measurement_strength < 0.35 and label != "UNCERTAIN":
        warnings.append(
            "Measurement strength is low; confidence has been discounted "
            "(consider a longer, steadier, closer-range recording)."
        )

    return evidence, warnings
