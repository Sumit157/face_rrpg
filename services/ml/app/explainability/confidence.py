"""Confidence calibration/annotation.

Confidence is the probability of the STATED label, discounted by how
judgeable the video was (measurement strength). A model probability of
0.9 computed from a weak measurement is not trustworthy at 0.9 -- the
discount makes the reported number reflect that without forcing an
"uncertain" verdict on otherwise decidable clips.
"""
from __future__ import annotations
from dataclasses import dataclass

from app.pipeline.decision import Decision


@dataclass
class CalibratedConfidence:
    raw_probability: float          # fused P(REAL), pre-calibration
    signal_quality: float           # 0-100 quality score for context
    adjusted_confidence: float      # confidence IN THE STATED LABEL (0-1)

    def to_dict(self) -> dict:
        return {
            "rawProbability": round(self.raw_probability, 4),
            "signalQuality": round(self.signal_quality, 1),
            "adjustedConfidence": round(self.adjusted_confidence, 4),
        }


def calibrate(decision: Decision, signal_quality_score: float) -> CalibratedConfidence:
    """Map the decision's side-aware confidence onto CalibratedConfidence.

    adjustedConfidence is P(stated label) after the measurement-strength
    discount computed in app.pipeline.decision (so a FAKE verdict with
    P(REAL)=0.2 reports 0.8, not 0.2 -- the UI displays it directly)."""
    return CalibratedConfidence(
        raw_probability=decision.probability_real,
        signal_quality=signal_quality_score,
        adjusted_confidence=decision.confidence,
    )
