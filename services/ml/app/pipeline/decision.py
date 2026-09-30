"""Decision fusion: the layer that turns probabilities and cue scores
into a REAL / FAKE / UNCERTAIN verdict with confidence and explanation.

Design (see docs/architecture.md "Decision fusion"):

1. Cues  -- transparent rule scores (app.features.cue_scores) over
   coherence, SNR, stability, morphology, spectral concentration.
2. Model -- trained RandomForest probability over the full feature
   vector (app.models.inference), when a model file exists.
3. Fusion -- p_cue = 0.65 * model + 0.35 * rules (rules only, when
   untrained). Never a single weak signal decides alone.
4. Measurement strength -- how much the video actually lets us judge
   (signal quality + kept-window ratio + face detection rate). Weak
   measurements blend the fused score toward a conservative REAL-leaning
   prior instead of flip-flopping around 0.5.
5. Uncertainty -- reserved for EXTREMELY poor videos (per the spec):
   no detectable face, or nothing survived quality gating on top of
   garbage quality, or very low SNR + high motion + unstable lighting.
   Every other clip gets a forced Real/Fake decision.

Confidence = P(favoured class) discounted by measurement strength, so
the system stays honest on bad videos without refusing to answer.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from app.core.logging import get_logger
from app.features.cue_scores import rule_score

logger = get_logger(__name__)

MODEL_WEIGHT = 0.65      # weight of trained model vs rule cues when both exist
REAL_PRIOR = 0.62        # conservative lean when evidence can't discriminate
SNR_FLOOR_DB = -6.0      # measurement floor (matches quality_gating.SNR_FLOOR_DB)


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


@dataclass
class Decision:
    label: str                      # REAL | FAKE | UNCERTAIN
    probability_real: float          # final fused P(REAL)
    confidence: float                # 0-1 confidence IN THE STATED LABEL
    measurement_strength: float      # 0-1 how judgeable the clip is
    explanation: str
    cues: Dict[str, float] = field(default_factory=dict)
    rule_score: float = 0.5
    model_score: Optional[float] = None
    uncertain_reason: str = ""
    model_version: str = "heuristic-rule-fusion-v1"
    is_heuristic_fallback: bool = True

    @property
    def probability_fake(self) -> float:
        return 1.0 - self.probability_real


def measurement_strength(f: Dict[str, float], detection_rate: float) -> float:
    """How much the clip lets us judge, 0-1.

    Three independent contributors: signal quality of the analysed
    windows, how many windows survived quality gating, and how reliably
    a face was detectable."""
    quality = _clip01(f.get("signal_quality_mean", 0.0) / 60.0)
    kept = _clip01(f.get("kept_window_ratio", 0.0))
    detection = _clip01((detection_rate - 0.3) / 0.5)  # <30% -> 0, >=80% -> 1
    return float(np.clip(0.40 * quality + 0.30 * kept + 0.30 * detection, 0.0, 1.0))


def _extreme_poor(f: Dict[str, float], detection_rate: float) -> tuple[bool, str]:
    """Spec: 'Uncertain only when the video quality is extremely poor
    (very low SNR + high motion + poor lighting)' -- or when there is
    provably nothing to analyse (no face, or every window rejected)."""
    quality = f.get("signal_quality_mean", 0.0)
    kept = f.get("kept_window_ratio", 0.0)
    snr = f.get("kept_snr_mean", f.get("snr_db_mean", -20.0))
    motion = f.get("motion_score", 0.0)
    illum = f.get("illumination_stability", 0.0)

    if detection_rate < 0.25:
        return True, "no usable face detected"
    if kept <= 0.0 and quality < 30.0:
        return True, "no analysis window passed the quality gate"
    if snr <= SNR_FLOOR_DB and motion > 65.0 and illum < 40.0 and quality < 30.0:
        return True, "very low SNR with high motion and unstable lighting"
    if quality < 10.0 and kept < 0.2:
        return True, "signal quality far below the measurable threshold"
    return False, ""


def _explanation_real(f: Dict[str, float], cues: Dict[str, float]) -> str:
    fa = f.get("cross_region_frequency_agreement", 0.0)
    phase = f.get("phase_coherence", 0.0)
    snr = f.get("kept_snr_mean", f.get("snr_db_mean", -20.0))
    lr = f.get("lr_cheek_correlation", 0.0)
    if cues.get("coherence", 0.0) >= 0.55:
        return (
            f"Synchronized pulse across facial regions (cheek correlation {lr:+.2f}, "
            f"frequency agreement {fa:.0%}, phase stability {phase:.2f}) with "
            f"{snr:+.1f} dB SNR -> Real"
        )
    return (
        f"Weak cross-region coherence (cheek correlation {lr:+.2f}, "
        f"frequency agreement {fa:.0%}, phase stability {phase:.2f}) but no "
        f"manipulation evidence survived fusion -> Real"
    )


def _explanation_fake(
    f: Dict[str, float], cues: Dict[str, float], model_score: float | None
) -> str:
    corr = f.get("cross_region_mean_correlation", 0.0)
    fa = f.get("cross_region_frequency_agreement", 0.0)
    phase = f.get("phase_coherence", 0.0)
    if cues.get("coherence", 0.0) >= 0.55:
        # Coherence alone can look fine (e.g. synthetic sine pulses) --
        # say WHY it's fake instead of falsely claiming incoherence.
        why = "waveform lacks a natural pulse shape" if (model_score or 0.5) >= 0.5 else "classifier found synthetic artefacts"
        return (
            f"Regions look pulse-synchronized (cheek correlation {corr:+.2f}, "
            f"frequency agreement {fa:.0%}, phase stability {phase:.2f}) but the "
            f"{why} -> Fake"
        )
    return (
        f"No coherent physiological signal (cross-region correlation {corr:+.2f}, "
        f"frequency agreement {fa:.0%}, phase stability {phase:.2f}) -> Fake"
    )


def decide(
    feature_values: Dict[str, float],
    detection_rate: float = 1.0,
) -> Decision:
    """Fuse model + rules into the final verdict. feature_values is the
    flat dict from FeatureVector.to_dict()."""
    from app.models.inference import model_info, predict_proba
    from app.features.feature_vector import FEATURE_NAMES

    vector = np.array([feature_values.get(name, 0.0) for name in FEATURE_NAMES])

    rule, cues = rule_score(feature_values)
    model_version, is_heuristic = model_info()
    model_score: float | None = None
    if not is_heuristic:
        model_score = predict_proba(vector)

    if model_score is not None:
        p_cue = MODEL_WEIGHT * model_score + (1.0 - MODEL_WEIGHT) * rule
    else:
        p_cue = rule

    strength = measurement_strength(feature_values, detection_rate)
    extreme, extreme_reason = _extreme_poor(feature_values, detection_rate)

    if extreme:
        label = "UNCERTAIN"
        p_final = 0.5
        confidence = float(np.clip(strength * 0.5, 0.05, 0.45))
        explanation = (
            f"Video quality too poor to judge ({extreme_reason}; "
            f"quality {feature_values.get('signal_quality_mean', 0.0):.0f}/100, "
            f"motion {feature_values.get('motion_score', 0.0):.0f}, "
            f"lighting stability {feature_values.get('illumination_stability', 0.0):.0f})"
        )
        uncertain_reason = extreme_reason
    else:
        # Weak measurements drift toward the conservative prior instead of
        # flipping labels on noise; strong measurements decide outright.
        p_final = strength * p_cue + (1.0 - strength) * REAL_PRIOR
        label = "REAL" if p_final >= 0.5 else "FAKE"
        side_p = p_final if label == "REAL" else 1.0 - p_final
        discount = _clip01(0.35 + 0.65 * strength)
        confidence = float(np.clip(0.5 + (side_p - 0.5) * discount, 0.05, 0.99))
        explanation = (
            _explanation_real(feature_values, cues)
            if label == "REAL"
            else _explanation_fake(feature_values, cues, model_score)
        )
        if (
            model_score is not None
            and abs(model_score - rule) > 0.3
            and ((model_score >= 0.5) != (rule >= 0.5))
        ):
            explanation += (
                f" (model P(real)={model_score:.2f} vs rule cues {rule:.2f} -- "
                f"fusion resolved the disagreement)"
            )
        if strength < 0.35:
            explanation += " (low measurement strength: confidence discounted)"
        uncertain_reason = ""

    decision = Decision(
        label=label,
        probability_real=float(np.clip(p_final, 0.01, 0.99)),
        confidence=confidence,
        measurement_strength=strength,
        explanation=explanation,
        cues=cues,
        rule_score=rule,
        model_score=model_score,
        uncertain_reason=uncertain_reason,
        model_version=model_version,
        is_heuristic_fallback=is_heuristic,
    )
    logger.info(
        "Decision: %s (p_real=%.3f, confidence=%.2f, strength=%.2f, rule=%.3f, model=%s)",
        decision.label, decision.probability_real, decision.confidence,
        decision.measurement_strength, decision.rule_score,
        f"{model_score:.3f}" if model_score is not None else "n/a",
    )
    return decision
