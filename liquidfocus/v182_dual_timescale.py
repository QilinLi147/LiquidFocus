"""V182 simplified dual-timescale regional liquid primary model.

Stable evidence is a slow liquid observation, not a classifier.  Ordered
regional activity is a fast liquid observation.  Both states are fused before
the only primary readout.  The existing compact router/refiner remain the
second and third top-level modules.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .compact import (
    CausalLiquidCell, CompactConfig, CompactLiquidFocus,
)
from .geometry import default_channel_coordinates
from .v163_identity_liquid import exact_anatomical_membership_torch
from .v164_stable_liquid import StableEvidenceLiquidObservation


class DualTimescaleRegionalLiquidEncoder(nn.Module):
    """One slow evidence state and one fast rhythm state per region."""

    def __init__(
        self, classes: int, channels: int, bands: int, regions: int,
        config: CompactConfig,
    ) -> None:
        super().__init__()
        if config.width != 64:
            raise ValueError("V182 freezes total regional state width at 64")
        if config.temporal_cell != "liquid":
            raise ValueError("V182 requires the causal liquid cell")
        if config.spatial_stem != "stable_evidence_liquid_observation":
            raise ValueError("V182 requires stable evidence as a liquid observation")
        self.classes = classes
        self.channels = channels
        self.bands = bands
        self.regions = regions
        self.width = config.width
        self.slow_width = 32
        self.fast_width = 32
        self.use_delta_t = config.use_delta_t
        coordinates = default_channel_coordinates()
        if coordinates.shape[0] != channels:
            raise ValueError("V182 channel-coordinate count mismatch")
        if regions != 7:
            raise ValueError("V182 freezes seven anatomical regions")
        membership = exact_anatomical_membership_torch().to(coordinates)
        region_coordinates = (
            membership @ coordinates
            / membership.sum(dim=1, keepdim=True).clamp_min(1.0)
        )
        self.register_buffer("channel_coordinates", coordinates)
        self.register_buffer("region_membership", membership)
        self.register_buffer("channel_region", membership.argmax(dim=0))
        self.register_buffer("region_coordinates", region_coordinates)

        local_dimension = 3 * bands + 4
        self.fast_input = nn.Sequential(
            nn.LayerNorm(local_dimension),
            nn.Linear(local_dimension, self.fast_width),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.stable_observation = StableEvidenceLiquidObservation(
            channels, bands, regions, self.slow_width, rank=32
        )
        self.slow_cell = CausalLiquidCell(self.slow_width)
        self.fast_cell = CausalLiquidCell(self.fast_width)
        self.slow_to_fast = nn.Linear(self.slow_width, self.fast_width, bias=False)
        self.slow_norm = nn.LayerNorm(self.slow_width)
        self.fast_norm = nn.LayerNorm(self.fast_width)
        # Slower than the rhythm path but never a static bypass.
        self.register_buffer("slow_elapsed_scale", torch.tensor(0.25))
        primary_dimension = regions * 4 * self.width
        self.primary_readout = nn.Sequential(
            nn.LayerNorm(primary_dimension),
            nn.Dropout(config.dropout),
            nn.Linear(primary_dimension, classes),
        )

    @staticmethod
    def _spans(
        x: Tensor, delta_t: Optional[Tensor], use_delta_t: bool,
    ) -> Tensor:
        batch, time = x.shape[0], x.shape[2]
        if delta_t is None or not use_delta_t:
            return x.new_ones(batch, time)
        spans = delta_t.to(device=x.device, dtype=x.dtype)
        if spans.ndim == 1:
            spans = spans[None].expand(batch, -1)
        if spans.ndim == 3 and spans.shape[-1] == 1:
            spans = spans[..., 0]
        if spans.shape != (batch, time):
            raise ValueError("V182 delta_t must be [T], [B,T], or [B,T,1]")
        if not torch.isfinite(spans).all() or torch.any(spans < 0):
            raise ValueError("V182 delta_t must be finite and non-negative")
        if torch.any((spans > 0).sum(dim=1) == 0):
            raise ValueError("V182 every sample needs an observed time step")
        return spans

    def _local_sequence(
        self, x: Tensor, channel_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        mask = channel_mask.to(dtype=x.dtype)
        weighted = self.region_membership[None] * mask[:, None]
        count = weighted.sum(dim=2).clamp_min(1.0)
        mean = torch.einsum("brc,bctf->brtf", weighted, x)
        mean = mean / count[:, :, None, None]
        centered = x[:, None] - mean[:, :, None]
        standard_deviation = (
            torch.einsum("brc,brctf->brtf", weighted, centered.square())
            / count[:, :, None, None]
        ).clamp_min(0.0).add(1e-6).sqrt()
        absolute = torch.einsum("brc,bctf->brtf", weighted, x.abs())
        absolute = absolute / count[:, :, None, None]
        coverage = (
            weighted.sum(dim=2)
            / self.region_membership.sum(dim=1)[None].clamp_min(1.0)
        )
        coverage = coverage[:, :, None, None].expand(-1, -1, x.shape[2], -1)
        geometry = self.region_coordinates[None, :, None].expand(
            x.shape[0], -1, x.shape[2], -1
        )
        feature = torch.cat(
            (mean, standard_deviation, absolute, coverage, geometry), dim=-1
        )
        valid = weighted.sum(dim=2) > 0
        return mean, feature.permute(0, 2, 1, 3).contiguous(), valid

    def forward(
        self, x: Tensor, delta_t: Optional[Tensor], channel_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        region_sequence, local, valid = self._local_sequence(x, channel_mask)
        spans = self._spans(x, delta_t, self.use_delta_t)
        fast_observation = self.fast_input(local) * valid[:, None, :, None]
        slow_observation = self.stable_observation(
            x, spans, channel_mask, valid
        )
        batch, time = x.shape[0], x.shape[2]
        slow_state = x.new_zeros(batch * self.regions, self.slow_width)
        fast_state = x.new_zeros(batch * self.regions, self.fast_width)
        trajectory = []
        valid_flat = valid.reshape(-1, 1)
        for step in range(time):
            span = spans[:, step : step + 1, None].expand(
                -1, -1, self.regions
            ).reshape(-1, 1)
            observed = (span > 0) & valid_flat
            safe = torch.where(observed, span, torch.ones_like(span))
            slow_candidate = self.slow_cell(
                slow_observation[:, step].reshape(-1, self.slow_width),
                slow_state,
                safe * self.slow_elapsed_scale,
            )
            slow_next = torch.where(observed, slow_candidate, slow_state)
            fast_value = (
                fast_observation[:, step].reshape(-1, self.fast_width)
                + self.slow_to_fast(self.slow_norm(slow_next))
            )
            fast_candidate = self.fast_cell(fast_value, fast_state, safe)
            fast_next = torch.where(observed, fast_candidate, fast_state)
            slow_state, fast_state = slow_next, fast_next
            fused = torch.cat(
                (self.slow_norm(slow_state), self.fast_norm(fast_state)), dim=-1
            ).reshape(batch, self.regions, self.width)
            trajectory.append(fused)
        regional = torch.stack(trajectory, dim=1)
        weight = (spans > 0).to(dtype=x.dtype)
        count = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (regional * weight[:, :, None, None]).sum(dim=1) / count[:, :, None]
        centered = regional - mean[:, None]
        std = (
            centered.square() * weight[:, :, None, None]
        ).sum(dim=1) / count[:, :, None]
        std = std.clamp_min(0.0).add(1e-6).sqrt()
        position = torch.linspace(-1.0, 1.0, time, device=x.device, dtype=x.dtype)
        trend = (
            centered * position[None, :, None, None] * weight[:, :, None, None]
        ).sum(dim=1) / count[:, :, None]
        last_index = (
            (spans > 0).to(torch.int64)
            * torch.arange(1, time + 1, device=x.device)[None]
        ).argmax(dim=1)
        last = regional[torch.arange(batch, device=x.device), last_index]
        summary = torch.cat((last, mean, std, trend), dim=-1)
        primary_logits = self.primary_readout(summary.flatten(1))
        primary_state = summary.reshape(
            batch, self.regions, 4, self.width
        ).mean(dim=(1, 2))
        return primary_logits, primary_state, regional, region_sequence

    @torch.no_grad()
    def set_observation_input_transform(self, mean: Tensor, scale: Tensor) -> None:
        self.stable_observation.set_input_transform(mean, scale)

    @torch.no_grad()
    def set_stable_feature_transform(self, mean: Tensor, scale: Tensor) -> None:
        self.stable_observation.set_feature_transform(mean, scale)


class DualTimescaleLiquidFocus(CompactLiquidFocus):
    """Reuse only the compact hierarchical router/refiner around V182 primary."""

    def __init__(
        self, classes: int, channels: int = 62, time: int = 10,
        bands: int = 5, regions: int = 7,
        config: Optional[CompactConfig] = None,
    ) -> None:
        frozen = config or CompactConfig(
            width=64, dropout=0.10,
            spatial_stem="stable_evidence_liquid_observation",
        )
        # The legacy compact constructor internally knows only its historical
        # eight geometry sectors.  Use it solely to construct the shared
        # router/refiner, then replace the encoder and public region contract
        # with the frozen seven anatomical regions.
        super().__init__(
            classes=classes, channels=channels, time=time, bands=bands,
            regions=8, config=frozen,
        )
        self.regions = regions
        self.encoder = DualTimescaleRegionalLiquidEncoder(
            classes, channels, bands, regions, self.config
        )
        if list(name for name, _module in self.named_children()) != [
            "encoder", "router", "refiner"
        ]:
            raise AssertionError("V182 must expose exactly three top-level modules")


def build_v182_model(
    *, classes: int, config: CompactConfig, channels: int = 62,
    time: int = 10, bands: int = 5,
) -> DualTimescaleLiquidFocus:
    return DualTimescaleLiquidFocus(
        classes=classes, channels=channels, time=time, bands=bands,
        config=config,
    )
