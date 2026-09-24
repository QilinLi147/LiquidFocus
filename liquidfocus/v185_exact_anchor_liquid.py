"""V185 exact stable coordinate with a development-gated liquid residual."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .compact import CompactConfig
from .v183_anchored_liquid import (
    AnchorPreservingReadout, AnchoredDualTimescaleEncoder,
    AnchoredDualTimescaleLiquidFocus,
)


class ExactAnchorLiquidReadout(AnchorPreservingReadout):
    """One readout whose zero residual exactly restores the stable coordinate."""

    def __init__(self, classes: int, regions: int, width: int, dropout: float) -> None:
        super().__init__(classes, regions, width, dropout)
        self.register_buffer("residual_scale", torch.tensor(1.0))

    def forward(self, flattened: Tensor) -> Tensor:
        summary = flattened.reshape(-1, self.regions, 4, self.width)
        stable = summary[:, :, 0, : self.classes].sum(dim=1)
        fast = summary[:, :, :, self.width // 2 :].reshape(summary.shape[0], -1)
        residual = self.residual_bound * torch.tanh(self.rhythm(fast))
        enabled = self.residual_enabled.to(residual.dtype)
        return stable + enabled * self.residual_scale.to(residual.dtype) * residual

    @torch.no_grad()
    def set_residual_scale(self, value: float) -> None:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("V185 residual scale must be in [0,1]")
        self.residual_scale.fill_(float(value))


class ExactAnchorLiquidEncoder(AnchoredDualTimescaleEncoder):
    def __init__(
        self, classes: int, channels: int, bands: int, regions: int,
        config: CompactConfig,
    ) -> None:
        super().__init__(classes, channels, bands, regions, config)
        self.primary_readout = ExactAnchorLiquidReadout(
            classes, regions, self.width, config.dropout
        )


class ExactAnchorLiquidFocus(AnchoredDualTimescaleLiquidFocus):
    def __init__(
        self, classes: int, channels: int = 62, time: int = 10,
        bands: int = 5, regions: int = 7,
        config: Optional[CompactConfig] = None,
    ) -> None:
        super().__init__(
            classes=classes, channels=channels, time=time, bands=bands,
            regions=regions, config=config,
        )
        self.encoder = ExactAnchorLiquidEncoder(
            classes, channels, bands, regions, self.config
        )
        if list(dict(self.named_children())) != ["encoder", "router", "refiner"]:
            raise AssertionError("V185 must expose exactly encoder/router/refiner")


def build_v185_model(
    *, classes: int, config: CompactConfig, channels: int = 62,
    time: int = 10, bands: int = 5,
) -> ExactAnchorLiquidFocus:
    return ExactAnchorLiquidFocus(
        classes=classes, channels=channels, time=time, bands=bands, config=config
    )
