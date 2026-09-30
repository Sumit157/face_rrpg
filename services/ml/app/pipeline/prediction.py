"""Feature-vector assembly + model inference + calibrated confidence."""
from __future__ import annotations
from dataclasses import dataclass

from app.features.feature_vector import build_feature_vector, FeatureVector
from app.models.inference import predict as model_predict
from app.models.baseline import PredictionResult
from app.explainability.confidence import calibrate, CalibratedConfidence
from app.pipeline.analysis import RegionAnalysis


@dataclass
class PredictionBundle:
    feature_vector: FeatureVector
    prediction: PredictionResult
    confidence: CalibratedConfidence


def run_prediction(region_analysis: RegionAnalysis) -> PredictionBundle:
    fv = build_feature_vector(
        region_analysis.region_results,
        region_analysis.consistency,
        region_analysis.motion,
        region_analysis.illumination,
        harmonic_ratio_value=region_analysis.harmonic_ratio,
        phase_coherence_value=region_analysis.phase_coherence,
        amplitude_cv_value=region_analysis.amplitude_cv,
    )
    prediction = model_predict(fv.as_array())
    confidence = calibrate(prediction.probability_real, fv.values.get("signal_quality_mean", 0.0))
    return PredictionBundle(feature_vector=fv, prediction=prediction, confidence=confidence)
