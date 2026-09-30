"""Feature-vector assembly -> decision fusion -> calibrated confidence."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

from app.features.feature_vector import build_feature_vector, FeatureVector
from app.models.baseline import PredictionResult
from app.explainability.confidence import calibrate, CalibratedConfidence
from app.pipeline.analysis import RegionAnalysis
from app.pipeline.decision import Decision, decide


@dataclass
class PredictionBundle:
    feature_vector: FeatureVector
    prediction: PredictionResult
    confidence: CalibratedConfidence
    decision: Decision


def run_prediction(
    region_analysis: RegionAnalysis,
    detection_rate: float = 1.0,
) -> PredictionBundle:
    fv = build_feature_vector(
        region_analysis.region_results,
        region_analysis.consistency,
        region_analysis.motion,
        region_analysis.illumination,
        extras=region_analysis.extras,
        gated_bpm=region_analysis.gating.kept_bpm,
        kept_snr=region_analysis.gating.kept_snr,
        region_kept_bpm=region_analysis.gating.region_kept_bpm,
    )
    decision = decide(fv.values, detection_rate=detection_rate)
    prediction = PredictionResult(
        label=decision.label,
        probability_real=decision.probability_real,
        probability_fake=decision.probability_fake,
        model_version=decision.model_version,
        is_heuristic_fallback=decision.is_heuristic_fallback,
        explanation=decision.explanation,
    )
    quality = fv.values.get("signal_quality_mean", 0.0)
    confidence = calibrate(decision, quality)
    return PredictionBundle(
        feature_vector=fv, prediction=prediction, confidence=confidence, decision=decision
    )


def run_prediction_from_features(
    feature_values: Dict[str, float], detection_rate: float = 1.0
) -> tuple[Decision, PredictionResult, CalibratedConfidence]:
    """Feature-dict entry point used by the /v1/inference route."""
    decision = decide(feature_values, detection_rate=detection_rate)
    prediction = PredictionResult(
        label=decision.label,
        probability_real=decision.probability_real,
        probability_fake=decision.probability_fake,
        model_version=decision.model_version,
        is_heuristic_fallback=decision.is_heuristic_fallback,
        explanation=decision.explanation,
    )
    confidence = calibrate(decision, feature_values.get("signal_quality_mean", 0.0))
    return decision, prediction, confidence
