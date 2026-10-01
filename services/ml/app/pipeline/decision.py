"""Model-free rPPG decision based on measured physiology cues.

The live label is derived only from the extracted rPPG feature set. The
repository's optional classifier was trained on synthetic signal examples
and has no real/fake video validation, so it cannot responsibly influence
the verdict or confidence.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from app.core.logging import get_logger
from app.features.cue_scores import rule_score

logger = get_logger(__name__)
SNR_FLOOR_DB = -8.0


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


@dataclass
class Decision:
    label: str
    probability_real: float
    confidence: float
    measurement_strength: float
    explanation: str
    cues: Dict[str, float] = field(default_factory=dict)
    rule_score: float = 0.5
    model_score: Optional[float] = None  # always None for rPPG-only inference
    uncertain_reason: str = ""
    model_version: str = "rppg-multicue-v2"
    is_heuristic_fallback: bool = True

    @property
    def probability_fake(self) -> float:
        return 1.0 - self.probability_real


def measurement_strength(f: Dict[str, float], detection_rate: float) -> float:
    quality = _clip01(f.get("signal_quality_mean", 0.0) / 60.0)
    kept = _clip01(f.get("effective_kept_window_ratio", f.get("kept_window_ratio", 0.0)))
    detection = _clip01((detection_rate - 0.3) / 0.5)
    return float(np.clip(0.40 * quality + 0.30 * kept + 0.30 * detection, 0.0, 1.0))


def _extreme_poor(f: Dict[str, float], detection_rate: float) -> tuple[bool, str]:
    quality = f.get("signal_quality_mean", 0.0)
    kept = f.get("kept_window_ratio", 0.0)
    snr = f.get("kept_snr_mean", f.get("snr_db_mean", -20.0))
    motion = f.get("motion_score", 0.0)
    illum = f.get("illumination_stability", 0.0)
    if detection_rate < 0.25:
        return True, "no usable face detected"
    if kept <= 0.0 and quality < 10.0 and snr <= -10.0:
        return True, "no measurable pulse survived quality gating"
    if snr <= SNR_FLOOR_DB and motion > 65.0 and illum < 40.0 and quality < 30.0:
        return True, "very low SNR with high motion and unstable lighting"
    if quality < 10.0 and kept < 0.2:
        return True, "signal quality far below the measurable threshold"
    return False, ""


def _explanation(f: Dict[str, float], cues: Dict[str, float], label: str) -> str:
    corr = f.get("cross_region_mean_correlation", 0.0)
    lr = f.get("lr_cheek_correlation", 0.0)
    freq = f.get("cross_region_frequency_agreement", 0.0)
    phase = f.get("phase_coherence", 0.0)
    snr = f.get("kept_snr_mean", f.get("snr_db_mean", -20.0))
    methods = int(f.get("available_method_count", 0))
    method_agreement = f.get("method_frequency_agreement", 0.5)
    if label == "REAL":
        text = (
            f"rPPG evidence supports Real: cheek correlation {lr:+.2f}, "
            f"cross-region frequency agreement {freq:.0%}, SNR {snr:+.1f} dB"
        )
    else:
        text = (
            f"rPPG evidence leans Fake: cheek correlation {lr:+.2f}, "
            f"cross-region frequency agreement {freq:.0%}, phase stability {phase:.2f}"
        )
    if cues.get("coherence", 0.5) < 0.4 or label == "FAKE":
        text += f" (mean region correlation {corr:+.2f})"
    if methods >= 2:
        text += f"; GREEN/POS/CHROM frequency agreement {method_agreement:.0%}"
    text += "; lack of a coherent pulse is not standalone proof of manipulation"
    return text


def decide(feature_values: Dict[str, float], detection_rate: float = 1.0) -> Decision:
    """Convert the measured rPPG features into a quality-aware verdict."""
    rppg_score, cues = rule_score(feature_values)
    strength = measurement_strength(feature_values, detection_rate)
    extreme, reason = _extreme_poor(feature_values, detection_rate)

    if extreme:
        label = "UNCERTAIN"
        p_final = 0.5
        confidence = float(np.clip(strength * 0.5, 0.05, 0.45))
        explanation = (
            f"Video quality too poor to judge ({reason}; "
            f"quality {feature_values.get('signal_quality_mean', 0.0):.0f}/100, "
            f"motion {feature_values.get('motion_score', 0.0):.0f}, "
            f"lighting stability {feature_values.get('illumination_stability', 0.0):.0f})"
        )
    else:
        # No real/fake prior: weak or mixed evidence stays near 50%, while
        # the label follows the aggregate direction. Quality controls its pull.
        p_final = 0.5 + strength * (rppg_score - 0.5)
        label = "REAL" if p_final >= 0.5 else "FAKE"
        side_probability = p_final if label == "REAL" else 1.0 - p_final
        discount = _clip01(0.35 + 0.65 * strength)
        confidence = float(np.clip(0.5 + (side_probability - 0.5) * discount, 0.05, 0.95))
        explanation = _explanation(feature_values, cues, label)
        if strength < 0.35:
            explanation += " (low measurement strength; confidence discounted)"
        reason = ""

    decision = Decision(
        label=label,
        probability_real=float(np.clip(p_final, 0.01, 0.99)),
        confidence=confidence,
        measurement_strength=strength,
        explanation=explanation,
        cues=cues,
        rule_score=rppg_score,
        uncertain_reason=reason,
        model_version="rppg-multicue-v2",
        is_heuristic_fallback=True,
    )
    logger.info(
        "rPPG decision: %s (p_real=%.3f, confidence=%.2f, strength=%.2f, cues=%.3f)",
        decision.label, decision.probability_real, decision.confidence,
        decision.measurement_strength, decision.rule_score,
    )
    return decision
