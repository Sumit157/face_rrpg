"""POST /v1/inference -- run only the decision stage on a precomputed
feature vector (used by training/evaluation scripts and the Signal
Explorer's "what-if" comparisons without re-running video processing).
"""
from typing import Dict
from fastapi import APIRouter
from pydantic import BaseModel

from app.pipeline.prediction import run_prediction_from_features

router = APIRouter()


class InferenceRequest(BaseModel):
    features: Dict[str, float]


class InferenceResponse(BaseModel):
    prediction: dict
    confidence: dict


@router.post("/v1/inference", response_model=InferenceResponse)
def inference(request: InferenceRequest) -> InferenceResponse:
    # Default detection_rate=1.0: callers of this endpoint only send features.
    _, result, conf = run_prediction_from_features(request.features, detection_rate=1.0)
    return InferenceResponse(prediction=result.to_dict(), confidence=conf.to_dict())
