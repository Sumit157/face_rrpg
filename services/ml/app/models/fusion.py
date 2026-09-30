"""Visual-only + rPPG-only late-fusion (Milestone 14 / Objective: fusion).

Placeholder interface: a visual-only deepfake classifier is out of scope
for the initial rPPG-focused build. This module defines the fusion
contract so a visual model can be dropped in later without changing the
rest of the pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class FusionResult:
    p_real_rppg: float
    p_real_visual: float | None
    p_real_fused: float
    strategy: str

    def to_dict(self) -> dict:
        return {
            "pRealRppg": round(self.p_real_rppg, 4),
            "pRealVisual": round(self.p_real_visual, 4) if self.p_real_visual is not None else None,
            "pRealFused": round(self.p_real_fused, 4),
            "strategy": self.strategy,
        }


def late_fusion(p_real_rppg: float, p_real_visual: float | None,
                 weight_rppg: float = 0.5) -> FusionResult:
    if p_real_visual is None:
        return FusionResult(p_real_rppg, None, p_real_rppg, strategy="rppg_only")
    fused = weight_rppg * p_real_rppg + (1 - weight_rppg) * p_real_visual
    return FusionResult(p_real_rppg, p_real_visual, fused, strategy="weighted_average")
