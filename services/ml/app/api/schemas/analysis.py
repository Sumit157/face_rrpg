"""Pydantic request/response schemas for the internal FastAPI service.

Kept intentionally close to the shared TypeScript types in
packages/shared-types so the Node service's expectations and the
Python service's actual output never drift silently out of sync.
"""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


class AnalyseRequest(BaseModel):
    analysis_id: str
    video_path: str
    method: str = Field(default="auto", pattern="^(green|chrom|pos|auto)$")
    regions: List[str] = Field(default_factory=lambda: ["forehead", "left_cheek", "right_cheek"])
    window_seconds: float = 12.0
    step_seconds: float = 3.0


class AnalyseResponse(BaseModel):
    analysisId: str
    videoMetadata: dict
    faceQuality: dict
    signalQuality: dict
    method: str
    regions: List[str]
    windowSeconds: float
    stepSeconds: float
    bpm: dict
    consistency: dict
    featureVector: dict
    prediction: dict
    confidence: dict
    evidence: List[str]
    warnings: List[str]
    artifacts: dict = {}


class HealthResponse(BaseModel):
    status: str
    service: str
    modelLoaded: bool
    modelVersion: str = "unknown"
    validationStatus: str = "unknown"


class ProgressResponse(BaseModel):
    analysisId: str
    stage: str
    progress: int
    message: str
    done: bool
    # Stage transitions with seconds-relative timestamps, for live
    # stage-duration timelines in the UI.
    history: Optional[List[dict]] = None
