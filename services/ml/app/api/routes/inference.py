"""POST /v1/inference -- run only the classifier stage on a precomputed
feature vector (used by training/evaluation scripts and the Signal
Explorer's "what-if" comparisons without re-running video processing).
"""
from typing import Dict
from fastapi import APIRouter
from pydantic import BaseModel

from app.features.feature_vector import FEATURE_NAMES
from app.models.inference import predict
from app.explainability.confidence import calibrate
import numpy as np

router = APIRouter()


class InferenceRequest(BaseModel):
    features: Dict[str, float]


class InferenceResponse(BaseModel):
    prediction: dict
    confidence: dict


@router.post("/v1/inference", response_model=InferenceResponse)
def inference(request: InferenceRequest) -> InferenceResponse:
    vector = np.array([request.features.get(name, 0.0) for name in FEATURE_NAMES])
    result = predict(vector)
    conf = calibrate(result.probability_real, request.features.get("signal_quality_mean", 0.0))
    return InferenceResponse(prediction=result.to_dict(), confidence=conf.to_dict())
