"""Lightweight expert layers used by the compact public model."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TemporalSpectralExpert(nn.Module):
    """Shared temporal expert invoked for selected channels."""

    def __init__(self, bands: int, width: int, dropout: float = 0.10):
        super().__init__()
        self.band_norm = nn.LayerNorm(bands)
        self.band_encoder = nn.Linear(bands, width)
        self.depthwise = nn.Conv1d(
            width, width, kernel_size=3, padding=1, groups=width
        )
        self.temporal_mix = nn.Conv1d(width, width, kernel_size=1)
        self.temporal_score = nn.Linear(width, 1)
        self.coordinate = nn.Sequential(
            nn.Linear(3, width), nn.Tanh(), nn.Linear(width, width, bias=False)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width, width), nn.LayerNorm(width),
        )

    def forward(self, sequence: Tensor, coordinate: Tensor) -> Tensor:
        if sequence.ndim != 3:
            raise ValueError("channel expert expects [N,time,bands]")
        features = F.gelu(self.band_encoder(self.band_norm(sequence)))
        convolution = features.transpose(1, 2)
        residual = self.temporal_mix(
            F.gelu(self.depthwise(convolution))
        ).transpose(1, 2)
        features = features + residual
        attention = torch.softmax(
            self.temporal_score(features).squeeze(-1), dim=1
        )
        pooled = torch.einsum("nt,ntd->nd", attention, features)
        return self.output(pooled + self.coordinate(coordinate))


def geometry_membership(coordinates: Tensor, regions: int) -> Tensor:
    """Assign sensors to deterministic scalp sectors."""

    if regions != 8:
        raise ValueError("the compact base model defines eight geometry regions")
    xy = coordinates[:, :2]
    xy = (xy - xy.mean(dim=0, keepdim=True)) / xy.std(
        dim=0, keepdim=True, unbiased=False
    ).clamp_min(1e-5)
    anchors = xy.new_tensor([
        [-1.20, 1.00], [0.00, 1.40], [1.20, 1.00],
        [-1.40, 0.00], [1.40, 0.00],
        [-1.20, -1.00], [0.00, -1.40], [1.20, -1.00],
    ])
    assignment = (
        (xy[:, None] - anchors[None]).square().sum(dim=-1).argmin(dim=-1)
    )
    membership = F.one_hot(
        assignment, num_classes=regions
    ).transpose(0, 1).float()
    if torch.any(membership.sum(dim=1) == 0):
        raise RuntimeError("scalp geometry produced an empty region")
    return membership
