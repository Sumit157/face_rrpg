"""FastAPI application entrypoint for the rPPG ML service.

Run locally with:
    uvicorn app.main:app --reload --port 8000

This service is only ever called by the Node/Express API (see
apps/api/src/services/python.service.ts) -- it is not exposed publicly.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import get_logger
from app.api.routes import health, analysis, inference

logger = get_logger(__name__)

app = FastAPI(
    title="rPPG Deepfake Detection - ML Service",
    description=(
        "Explainable forensic analysis of physiological (rPPG) signal "
        "consistency in facial video, as an additional cue alongside "
        "visual deepfake detection -- not a standalone proof of authenticity."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Node API only
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(analysis.router)
app.include_router(inference.router)


@app.on_event("startup")
def on_startup() -> None:
    logger.info("rPPG ML service starting on %s:%s", settings.host, settings.port)
