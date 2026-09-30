"""Assembles the final structured analysis result returned to the caller
(and, in the full system, persisted by the Node API / used to render a
downloadable report)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AnalysisReport:
    analysis_id: str
    video_metadata: dict
    face_quality: dict
    signal_quality: dict
    method: str
    regions: List[str]
    window_seconds: float
    step_seconds: float
    bpm: dict
    consistency: dict
    feature_vector: dict
    prediction: dict
    confidence: dict
    evidence: List[str]
    warnings: List[str]
    artifacts: Dict[str, str] = field(default_factory=dict)  # region -> file path for waveform/spectrum arrays

    def to_dict(self) -> dict:
        return {
            "analysisId": self.analysis_id,
            "videoMetadata": self.video_metadata,
            "faceQuality": self.face_quality,
            "signalQuality": self.signal_quality,
            "method": self.method,
            "regions": self.regions,
            "windowSeconds": self.window_seconds,
            "stepSeconds": self.step_seconds,
            "bpm": self.bpm,
            "consistency": self.consistency,
            "featureVector": self.feature_vector,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "warnings": self.warnings,
            "artifacts": self.artifacts,
        }
