"""GET /health"""
from fastapi import APIRouter

from app.api.schemas.analysis import HealthResponse
from app.models.inference import get_classifier

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    classifier = get_classifier()
    return HealthResponse(status="ok", service="rppg-ml-service", modelLoaded=classifier.is_trained)
