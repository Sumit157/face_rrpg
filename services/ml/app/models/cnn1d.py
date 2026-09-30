"""1D CNN over multi-region rPPG sequences (Milestone 13).

Optional deep-learning path. Requires PyTorch; import is deferred so the
service can run in classical-only mode without a torch install.
"""
from __future__ import annotations
from typing import List


def build_model(n_regions: int = 3, seq_len: int = 360):
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for the CNN1D model. Install with `pip install torch`."
        ) from exc

    class RPPGCNN1D(nn.Module):
        def __init__(self, in_channels: int, seq_len: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(128, 2))

        def forward(self, x):  # x: (B, C, T)
            return self.head(self.net(x))

    return RPPGCNN1D(in_channels=n_regions, seq_len=seq_len)
