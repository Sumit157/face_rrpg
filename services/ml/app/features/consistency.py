"""Cross-region physiological consistency.

Real facial blood flow should pulse near-synchronously across forehead
and both cheeks. Low cross-region correlation or disagreement between
each region's dominant frequency is treated as evidence *against* a
physiologically coherent signal (which some synthesis pipelines fail to
reproduce faithfully), not proof of manipulation on its own.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List
import numpy as np


@dataclass
class ConsistencyResult:
    mean_correlation: float
    pairwise_correlations: Dict[str, float]
    frequency_agreement: float  # fraction of region pairs whose peak Hz agree within tolerance

    def to_dict(self) -> dict:
        return {
            "meanCorrelation": round(self.mean_correlation, 4),
            "pairwiseCorrelations": {
                k: round(v, 4) for k, v in self.pairwise_correlations.items()
            },
            "frequencyAgreement": round(self.frequency_agreement, 4),
        }


def cross_region_consistency(
    region_signals: Dict[str, np.ndarray],
    region_peak_hz: Dict[str, float],
    freq_tolerance_hz: float = 0.15,
) -> ConsistencyResult:
    regions = list(region_signals.keys())
    pairwise: Dict[str, float] = {}
    freq_agree_count = 0
    total_pairs = 0

    for a, b in combinations(regions, 2):
        sig_a, sig_b = region_signals[a], region_signals[b]
        n = min(len(sig_a), len(sig_b))
        if n < 2:
            continue
        corr = float(np.corrcoef(sig_a[:n], sig_b[:n])[0, 1])
        if np.isnan(corr):
            corr = 0.0
        pairwise[f"{a}__{b}"] = corr

        total_pairs += 1
        if abs(region_peak_hz.get(a, 0) - region_peak_hz.get(b, 0)) <= freq_tolerance_hz:
            freq_agree_count += 1

    mean_corr = float(np.mean(list(pairwise.values()))) if pairwise else 0.0
    freq_agreement = freq_agree_count / total_pairs if total_pairs > 0 else 0.0

    return ConsistencyResult(
        mean_correlation=mean_corr,
        pairwise_correlations=pairwise,
        frequency_agreement=freq_agreement,
    )
