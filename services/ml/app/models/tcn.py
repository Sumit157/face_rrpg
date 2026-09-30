"""Temporal Convolutional Network alternative to the 1D CNN (Milestone 13).

A TCN's dilated causal convolutions give it a larger effective receptive
field than a plain CNN at the same depth, which is useful for capturing
slower temporal-stability patterns across a whole rPPG window.
"""
from __future__ import annotations


def build_model(n_regions: int = 3, num_channels: list[int] | None = None, kernel_size: int = 3):
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for the TCN model. Install with `pip install torch`."
        ) from exc

    num_channels = num_channels or [32, 64, 128]

    class TemporalBlock(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size, dilation):
            super().__init__()
            padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
            self.bn = nn.BatchNorm1d(out_ch)
            self.relu = nn.ReLU()
            self.chomp = padding

        def forward(self, x):
            out = self.conv(x)
            out = out[:, :, : -self.chomp] if self.chomp > 0 else out
            return self.relu(self.bn(out))

    class TCN(nn.Module):
        def __init__(self, in_channels: int, channels: list[int], kernel_size: int):
            super().__init__()
            layers = []
            prev = in_channels
            for i, ch in enumerate(channels):
                layers.append(TemporalBlock(prev, ch, kernel_size, dilation=2 ** i))
                prev = ch
            self.tcn = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(prev, 2))

        def forward(self, x):
            return self.head(self.pool(self.tcn(x)))

    return TCN(in_channels=n_regions, channels=num_channels, kernel_size=kernel_size)
