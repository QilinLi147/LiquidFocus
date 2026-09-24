"""Compact three-module LiquidFocus mainline.

This module exposes the paper-facing path:

1. a regional liquid encoder makes the primary prediction;
2. a cheap REGION -> CHANNEL router estimates local evidence value;
3. a bounded refiner applies selected evidence, with zero as an exact STOP.

Dense candidate evaluation is training-only.  Evaluation invokes at most one
region expert and one channel expert for each sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .experts import TemporalSpectralExpert, geometry_membership
from .geometry import default_channel_coordinates
from .v163_identity_liquid import CausalIdentityObservation
from .v164_stable_liquid import StableEvidenceLiquidObservation


REGION = 0
CHANNEL = 1
STOP = -1


@dataclass(frozen=True)
class CompactConfig:
    """The complete compact model contract; intentionally few knobs."""

    width: int = 48
    dropout: float = 0.10
    temporal_cell: str = "liquid"
    use_delta_t: bool = True
    enable_router: bool = True
    enable_refiner: bool = True
    enable_safe_stop: bool = True
    stop_threshold: float = 0.0
    max_residual: float = 0.75
    policy_weight: float = 0.50
    candidate_weight: float = 0.25
    safety_weight: float = 1.0
    budget_weight: float = 0.01
    rhythm_scale_init: float = 0.10
    spatial_stem: str = "legacy_moments"

    def validate(self) -> "CompactConfig":
        if self.width < 8:
            raise ValueError("width must be at least 8")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if self.temporal_cell not in {"liquid", "gru", "static"}:
            raise ValueError("temporal_cell must be liquid, gru, or static")
        if self.spatial_stem not in {
            "legacy_moments",
            "coordinate_moments",
            "coordinate_film",
            "identity_channel_observation",
            "identity_combined_observation",
            "stable_evidence_liquid_observation",
        }:
            raise ValueError(
                "spatial_stem must be legacy_moments, coordinate_moments, "
                "coordinate_film, identity_channel_observation, or "
                "identity_combined_observation, or "
                "stable_evidence_liquid_observation"
            )
        if self.max_residual <= 0:
            raise ValueError("max_residual must be positive")
        if min(
            self.policy_weight,
            self.candidate_weight,
            self.safety_weight,
            self.budget_weight,
        ) < 0:
            raise ValueError("loss weights must be non-negative")
        if not 0.0 <= self.rhythm_scale_init <= 1.0:
            raise ValueError("rhythm_scale_init must be in [0,1]")
        return self


@dataclass
class CompactOutput:
    logits: Tensor
    primary_logits: Tensor
    region_index: Tensor
    channel_index: Tensor
    action_mask: Tensor
    predicted_gain: Tensor
    region_gain: Tensor
    channel_gain: Tensor
    region_trajectory: Tensor
    dense_region_logits: Optional[Tensor] = None
    dense_channel_logits: Optional[Tensor] = None


@dataclass
class CompactLoss:
    loss: Tensor
    primary: Tensor
    final: Tensor
    candidate: Tensor
    policy: Tensor
    safety: Tensor
    budget: Tensor


class CausalLiquidCell(nn.Module):
    """Stable closed-form first-order liquid state update."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.target = nn.Linear(2 * width, width)
        self.rate = nn.Linear(2 * width, width)
        self.log_clock = nn.Parameter(torch.zeros(()))

    def forward(self, value: Tensor, state: Tensor, delta_t: Tensor) -> Tensor:
        joined = torch.cat((value, state), dim=-1)
        target = torch.tanh(self.target(joined))
        rate = F.softplus(self.rate(joined)).clamp(max=20.0)
        clock = self.log_clock.exp().clamp(0.25, 4.0)
        interpolation = 1.0 - torch.exp(-rate * clock * delta_t)
        return state + interpolation * (target - state)


class RegionalLiquidEncoder(nn.Module):
    """Module 1: causal region-wise rhythm modelling and primary prediction."""

    def __init__(
        self,
        classes: int,
        channels: int,
        bands: int,
        regions: int,
        config: CompactConfig,
    ) -> None:
        super().__init__()
        self.classes = classes
        self.channels = channels
        self.bands = bands
        self.regions = regions
        self.width = config.width
        self.temporal_cell = config.temporal_cell
        self.use_delta_t = config.use_delta_t
        self.spatial_stem = config.spatial_stem
        coordinates = default_channel_coordinates()
        if coordinates.shape[0] != channels:
            raise ValueError("channel-coordinate count mismatch")
        membership = geometry_membership(coordinates, regions)
        region_coordinates = (
            membership @ coordinates
            / membership.sum(dim=1, keepdim=True).clamp_min(1.0)
        )
        self.register_buffer("channel_coordinates", coordinates)
        self.register_buffer("region_membership", membership)
        self.register_buffer("channel_region", membership.argmax(dim=0))
        self.register_buffer("region_coordinates", region_coordinates)
        feature_dim = 3 * bands + 4
        if self.spatial_stem == "coordinate_moments":
            relative = coordinates[None] - region_coordinates[:, None]
            scale = torch.sqrt(
                (
                    membership[:, :, None] * relative.square()
                ).sum(dim=1)
                / membership.sum(dim=1, keepdim=True).clamp_min(1.0)
            ).clamp_min(1e-4)
            coordinate_basis = (
                membership[:, :, None] * relative / scale[:, None]
            )
            self.register_buffer("spatial_coordinate_basis", coordinate_basis)
            feature_dim += 3 * bands
        self.input_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, config.width),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        if self.spatial_stem == "coordinate_film":
            coordinate_width = max(8, config.width // 6)
            self.channel_band_projection = nn.Linear(
                bands, config.width, bias=False
            )
            self.coordinate_modulation = nn.Sequential(
                nn.Linear(3, coordinate_width),
                nn.GELU(),
                nn.Linear(coordinate_width, 2 * config.width),
            )
            self.film_scale = nn.Parameter(torch.tensor(0.10))
        if self.spatial_stem in {
            "identity_channel_observation", "identity_combined_observation"
        }:
            self.identity_observation = CausalIdentityObservation(
                channels,
                bands,
                regions,
                config.width,
                include_relations=(
                    self.spatial_stem == "identity_combined_observation"
                ),
            )
        if self.spatial_stem == "stable_evidence_liquid_observation":
            self.stable_observation = StableEvidenceLiquidObservation(
                channels, bands, regions, config.width
            )
        if config.temporal_cell in {"liquid", "static"}:
            # ``static`` keeps this module (and therefore every downstream
            # initialization) identical to ``liquid``.  Its forward path
            # alone bypasses the update, yielding a paired ablation.
            self.cell: nn.Module = CausalLiquidCell(config.width)
        elif config.temporal_cell == "gru":
            self.cell = nn.GRUCell(config.width, config.width)
        self.state_norm = nn.LayerNorm(config.width)
        self.rhythm_scale = nn.Parameter(
            torch.full((4 * config.width,), float(config.rhythm_scale_init))
        )
        primary_features = regions * 4 * config.width
        self.primary_readout = nn.Sequential(
            nn.LayerNorm(primary_features),
            nn.Dropout(config.dropout),
            nn.Linear(primary_features, classes),
        )

    @staticmethod
    def _spans(
        x: Tensor, delta_t: Optional[Tensor], use_delta_t: bool
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
            raise ValueError("delta_t must be [T], [B,T], or [B,T,1]")
        if not torch.isfinite(spans).all() or torch.any(spans < 0):
            raise ValueError("delta_t must contain finite non-negative spans")
        if torch.any((spans > 0).sum(dim=1) == 0):
            raise ValueError("every sample needs at least one positive span")
        return spans

    def _region_sequence(
        self, x: Tensor, channel_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        mask = channel_mask.to(dtype=x.dtype)
        weighted = self.region_membership[None] * mask[:, None]
        count = weighted.sum(dim=2).clamp_min(1.0)
        mean = torch.einsum("brc,bctf->brtf", weighted, x)
        mean = mean / count[:, :, None, None]
        centered = x[:, None] - mean[:, :, None]
        variance = torch.einsum(
            "brc,brctf->brtf", weighted, centered.square()
        ) / count[:, :, None, None]
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
        features = torch.cat(
            (mean, variance.add(1e-6).sqrt(), absolute, coverage, geometry),
            dim=-1,
        )
        if self.spatial_stem == "coordinate_moments":
            # A fixed, low-rank coordinate covariance preserves which
            # electrodes produced the regional signal without adding a
            # channel classifier or a learned electrode lookup table.
            first = torch.einsum(
                "brc,rcd,bctf->brtfd",
                weighted,
                self.spatial_coordinate_basis,
                x,
            ) / count[:, :, None, None, None]
            coordinate_mean = torch.einsum(
                "brc,rcd->brd", weighted, self.spatial_coordinate_basis
            ) / count[:, :, None]
            covariance = first - (
                mean[..., None] * coordinate_mean[:, :, None, None]
            )
            features = torch.cat((features, covariance.flatten(3)), dim=-1)
        valid = weighted.sum(dim=2) > 0
        return mean, features.permute(0, 2, 1, 3).contiguous(), valid

    def forward(
        self,
        x: Tensor,
        delta_t: Optional[Tensor],
        channel_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        region_sequence, features, valid = self._region_sequence(x, channel_mask)
        spans = self._spans(x, delta_t, self.use_delta_t)
        projected = self.input_projection(features)
        if self.spatial_stem == "coordinate_film":
            channel_value = self.channel_band_projection(x)
            gamma, beta = self.coordinate_modulation(
                self.channel_coordinates
            ).chunk(2, dim=-1)
            channel_token = F.gelu(
                (1.0 + 0.25 * torch.tanh(gamma))[None, :, None]
                * channel_value
                + 0.25 * torch.tanh(beta)[None, :, None]
            )
            weighted = (
                self.region_membership[None]
                * channel_mask.to(dtype=x.dtype)[:, None]
            )
            count = weighted.sum(dim=2).clamp_min(1.0)
            film_region = torch.einsum(
                "brc,bctw->brtw", weighted, channel_token
            ) / count[:, :, None, None]
            projected = projected + torch.tanh(self.film_scale) * film_region.permute(
                0, 2, 1, 3
            )
        if self.spatial_stem != "legacy_moments":
            projected = projected * valid[:, None, :, None]
        liquid_input = projected
        if self.spatial_stem in {
            "identity_channel_observation", "identity_combined_observation"
        }:
            # No parallel readout: the causal identity statistics can affect
            # primary logits only after entering the regional liquid state.
            liquid_input = liquid_input + self.identity_observation(
                x, spans, channel_mask, valid
            )
        elif self.spatial_stem == "stable_evidence_liquid_observation":
            liquid_input = liquid_input + self.stable_observation(
                x, spans, channel_mask, valid
            )
        batch, time, regions, _ = projected.shape
        state = x.new_zeros(batch * regions, self.width)
        trajectory = []
        valid_flat = valid.reshape(-1, 1)
        for step in range(time):
            value = liquid_input[:, step].reshape(-1, self.width)
            span = spans[:, step : step + 1, None].expand(
                -1, -1, regions
            ).reshape(-1, 1)
            observed = (span > 0) & valid_flat
            safe_span = torch.where(observed, span, torch.ones_like(span))
            if self.temporal_cell == "liquid":
                candidate = self.cell(value, state, safe_span)
            elif self.temporal_cell == "gru":
                raw = self.cell(value, state)
                alpha = 1.0 - torch.exp(-safe_span)
                candidate = state + alpha * (raw - state)
            else:
                candidate = state
            state = torch.where(observed, candidate, state)
            trajectory.append(
                self.state_norm(state).reshape(batch, regions, self.width)
            )
        regional = torch.stack(trajectory, dim=1)
        # Preserve region identity in the primary readout.  Stable regional
        # statistics are the base evidence; the liquid trajectory contributes
        # only a bounded residual rhythm summary.  Harmful liquid dimensions
        # can therefore shrink to zero instead of suppressing the whole base.
        position = torch.linspace(
            -1.0, 1.0, time, device=x.device, dtype=x.dtype
        )
        if self.spatial_stem == "legacy_moments":
            centered = projected - projected.mean(dim=1, keepdim=True)
            projected_trend = (
                centered * position[None, :, None, None]
            ).mean(dim=1)
            base_summary = torch.cat((
                projected[:, -1],
                projected.mean(dim=1),
                projected.std(dim=1, unbiased=False),
                projected_trend,
            ), dim=-1)
        else:
            # A zero elapsed span is a causal hold, not an observation.  The
            # new stem therefore excludes it from the non-recurrent summary
            # as well as from the liquid update.
            observed_time = spans > 0
            weight = observed_time.to(dtype=x.dtype)
            count_time = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean_projected = (
                projected * weight[:, :, None, None]
            ).sum(dim=1) / count_time[:, :, None]
            centered = projected - mean_projected[:, None]
            variance_projected = (
                centered.square() * weight[:, :, None, None]
            ).sum(dim=1) / count_time[:, :, None]
            projected_trend = (
                centered
                * position[None, :, None, None]
                * weight[:, :, None, None]
            ).sum(dim=1) / count_time[:, :, None]
            last_index = (
                observed_time.to(torch.int64)
                * torch.arange(1, time + 1, device=x.device)[None]
            ).argmax(dim=1)
            last_projected = projected[
                torch.arange(batch, device=x.device), last_index
            ]
            base_summary = torch.cat((
                last_projected,
                mean_projected,
                variance_projected.clamp_min(0.0).add(1e-6).sqrt(),
                projected_trend,
            ), dim=-1)
        regional_centered = regional - regional.mean(dim=1, keepdim=True)
        rhythm_summary = torch.cat((
            regional[:, -1],
            regional.mean(dim=1),
            regional.std(dim=1, unbiased=False),
            (
                regional_centered * position[None, :, None, None]
            ).mean(dim=1),
        ), dim=-1)
        if self.spatial_stem == "stable_evidence_liquid_observation":
            # The stable observation is evidence for the regional liquid
            # dynamics, never a parallel anchor classifier.  Its primary logits
            # therefore come only from the recurrent liquid belief summary.
            fused = rhythm_summary
        else:
            fused = base_summary + 0.5 * torch.tanh(
                self.rhythm_scale
            )[None, None] * rhythm_summary
        primary_logits = self.primary_readout(fused.flatten(1))
        primary_state = fused.reshape(
            batch, regions, 4, self.width
        ).mean(dim=(1, 2))
        return primary_logits, primary_state, regional, region_sequence

    def set_observation_input_transform(
        self, mean: Tensor, scale: Tensor
    ) -> None:
        if hasattr(self, "identity_observation"):
            self.identity_observation.set_input_transform(mean, scale)
        elif hasattr(self, "stable_observation"):
            self.stable_observation.set_input_transform(mean, scale)
        else:
            raise RuntimeError("encoder has no stable/identity observation")

    def set_stable_feature_transform(
        self, mean: Tensor, scale: Tensor
    ) -> None:
        if not hasattr(self, "stable_observation"):
            raise RuntimeError("encoder has no stable-evidence observation")
        self.stable_observation.set_feature_transform(mean, scale)


class HierarchicalRouter(nn.Module):
    """Module 2: cheap value scoring along one REGION -> CHANNEL path."""

    def __init__(self, bands: int, width: int, dropout: float) -> None:
        super().__init__()
        channel_dim = 4 * bands + 3
        self.channel_token = nn.Sequential(
            nn.LayerNorm(channel_dim),
            nn.Linear(channel_dim, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(width),
        )
        self.type_embedding = nn.Embedding(2, width)
        self.gain_head = nn.Sequential(
            nn.LayerNorm(4 * width),
            nn.Linear(4 * width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    def _score(self, query: Tensor, tokens: Tensor, kind: int) -> Tensor:
        tokens = tokens + self.type_embedding.weight[kind]
        expanded = query[:, None].expand(-1, tokens.shape[1], -1)
        features = torch.cat((
            expanded,
            tokens,
            expanded * tokens,
            (expanded - tokens).abs(),
        ), dim=-1)
        return self.gain_head(features).squeeze(-1)

    def _channel_tokens(self, x: Tensor, coordinates: Tensor) -> Tensor:
        position = torch.linspace(-1.0, 1.0, x.shape[2], device=x.device, dtype=x.dtype)
        centered = x - x.mean(dim=2, keepdim=True)
        slope = (centered * position[None, None, :, None]).mean(dim=2)
        difference = (x[:, :, 1:] - x[:, :, :-1]).square().mean(dim=2).sqrt()
        features = torch.cat((
            x.mean(dim=2),
            x.std(dim=2, unbiased=False),
            slope,
            difference,
            coordinates[None].expand(x.shape[0], -1, -1),
        ), dim=-1)
        return self.channel_token(features)

    def forward(
        self,
        x: Tensor,
        primary_state: Tensor,
        regional_state: Tensor,
        channel_coordinates: Tensor,
        channel_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        region_gain = self._score(primary_state, regional_state, REGION)
        channel_tokens = self._channel_tokens(x, channel_coordinates)
        channel_gain = self._score(primary_state, channel_tokens, CHANNEL)
        channel_gain = channel_gain.masked_fill(~channel_mask, -torch.inf)
        return region_gain, channel_gain, channel_tokens


class SafeEvidenceRefiner(nn.Module):
    """Module 3: bounded local correction; STOP is the exact zero residual."""

    def __init__(self, bands: int, width: int, classes: int, config: CompactConfig) -> None:
        super().__init__()
        self.max_residual = float(config.max_residual)
        # The auxiliary branch is deterministic so enabling it cannot change
        # the main encoder's RNG stream on the following batch.
        self.expert = TemporalSpectralExpert(bands, width, 0.0)
        self.type_embedding = nn.Embedding(2, width)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(2 * width),
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Linear(width, classes),
        )

    def delta(self, state: Tensor, evidence: Tensor, kind: int) -> Tensor:
        evidence = evidence + self.type_embedding.weight[kind]
        return self.max_residual * torch.tanh(
            self.delta_head(torch.cat((state, evidence), dim=-1))
        )


class CompactLiquidFocus(nn.Module):
    """Three-module model with a single, inspectable inference path."""

    def __init__(
        self,
        classes: int = 3,
        channels: int = 62,
        time: int = 10,
        bands: int = 5,
        regions: int = 8,
        config: Optional[CompactConfig] = None,
    ) -> None:
        super().__init__()
        self.config = (config or CompactConfig()).validate()
        self.classes = classes
        self.channels = channels
        self.time = time
        self.bands = bands
        self.regions = regions
        self.register_buffer(
            "stop_threshold", torch.tensor(float(self.config.stop_threshold))
        )
        self.encoder = RegionalLiquidEncoder(
            classes, channels, bands, regions, self.config
        )
        self.router = HierarchicalRouter(
            bands, self.config.width, 0.0
        )
        self.refiner = SafeEvidenceRefiner(
            bands, self.config.width, classes, self.config
        )

    def set_stop_threshold(self, value: float) -> None:
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError("STOP threshold must be finite")
        self.stop_threshold.fill_(float(value))

    def _expert_all(self, sequence: Tensor, coordinates: Tensor) -> Tensor:
        batch, items, time, bands = sequence.shape
        expanded_coordinates = coordinates[None].expand(batch, -1, -1)
        return self.refiner.expert(
            sequence.reshape(batch * items, time, bands),
            expanded_coordinates.reshape(batch * items, 3),
        ).reshape(batch, items, self.config.width)

    def _dense_candidates(
        self,
        primary_logits: Tensor,
        primary_state: Tensor,
        region_sequence: Tensor,
        x: Tensor,
    ) -> tuple[Tensor, Tensor]:
        region_evidence = self._expert_all(
            region_sequence, self.encoder.region_coordinates
        )
        channel_evidence = self._expert_all(
            x, self.encoder.channel_coordinates
        )
        region_state = primary_state[:, None].expand(-1, self.regions, -1)
        channel_state = primary_state[:, None].expand(-1, self.channels, -1)
        region_delta = self.refiner.delta(
            region_state, region_evidence, REGION
        )
        channel_delta = self.refiner.delta(
            channel_state, channel_evidence, CHANNEL
        )
        return (
            primary_logits[:, None] + region_delta,
            primary_logits[:, None] + channel_delta,
        )

    def _selected_delta(
        self,
        sequence: Tensor,
        coordinates: Tensor,
        primary_state: Tensor,
        active: Tensor,
        kind: int,
    ) -> Tensor:
        """Invoke the expert only for rows that did not STOP."""

        delta = primary_state.new_zeros(primary_state.shape[0], self.classes)
        if not torch.any(active):
            return delta
        selected = torch.nonzero(active, as_tuple=False).squeeze(1)
        evidence = self.refiner.expert(
            sequence[selected], coordinates[selected]
        )
        delta[selected] = self.refiner.delta(
            primary_state[selected], evidence, kind
        )
        return delta

    def forward(
        self,
        x: Tensor,
        delta_t: Optional[Tensor] = None,
        channel_mask: Optional[Tensor] = None,
        dense_teacher: Optional[bool] = None,
    ) -> CompactOutput:
        if x.ndim != 4 or x.shape[1:] != (
            self.channels, self.time, self.bands
        ):
            raise ValueError("x must be [B,C,T,F] with configured dimensions")
        if channel_mask is None:
            channel_mask = torch.ones(
                x.shape[0], self.channels, dtype=torch.bool, device=x.device
            )
        if channel_mask.shape != x.shape[:2]:
            raise ValueError("channel_mask must be [B,C]")
        if torch.any(~channel_mask.any(dim=1)):
            raise ValueError("every sample needs at least one observed channel")
        primary_logits, primary_state, trajectory, region_sequence = self.encoder(
            x, delta_t, channel_mask
        )
        auxiliary_state = primary_state.detach()
        auxiliary_base = (
            primary_logits.detach() if self.training else primary_logits
        )
        final_regional = trajectory[:, -1]
        region_gain, channel_gain, _ = self.router(
            x,
            auxiliary_state,
            final_regional.detach(),
            self.encoder.channel_coordinates,
            channel_mask,
        )
        available_region = torch.einsum(
            "rc,bc->br",
            self.encoder.region_membership,
            channel_mask.to(dtype=x.dtype),
        ) > 0
        region_gain = region_gain.masked_fill(~available_region, -torch.inf)
        batch_index = torch.arange(x.shape[0], device=x.device)
        region_index = region_gain.argmax(dim=1)
        region_value = region_gain[batch_index, region_index]
        if not self.config.enable_router:
            region_salience = final_regional.square().mean(dim=2).masked_fill(
                ~available_region, -torch.inf
            )
            region_index = region_salience.argmax(dim=1)
            region_value = torch.full_like(region_value, float("inf"))
        region_active = torch.ones_like(region_index, dtype=torch.bool)
        if self.config.enable_safe_stop:
            region_active = region_value > self.stop_threshold

        parent = self.encoder.channel_region[None] == region_index[:, None]
        legal_channel = parent & channel_mask
        routed_channel_gain = channel_gain.masked_fill(~legal_channel, -torch.inf)
        channel_index = routed_channel_gain.argmax(dim=1)
        channel_value = routed_channel_gain[batch_index, channel_index]
        if not self.config.enable_router:
            channel_salience = x.square().mean(dim=(2, 3)).masked_fill(
                ~legal_channel, -torch.inf
            )
            channel_index = channel_salience.argmax(dim=1)
            channel_value = torch.full_like(channel_value, float("inf"))
        channel_active = region_active.clone()
        if self.config.enable_safe_stop:
            channel_active &= channel_value > self.stop_threshold

        if not self.config.enable_refiner:
            inactive = torch.zeros(
                x.shape[0], 2, dtype=torch.bool, device=x.device
            )
            stopped = torch.full_like(region_index, STOP)
            return CompactOutput(
                logits=primary_logits,
                primary_logits=primary_logits,
                region_index=stopped,
                channel_index=stopped.clone(),
                action_mask=inactive,
                predicted_gain=torch.stack((region_value, channel_value), dim=1),
                region_gain=region_gain,
                channel_gain=channel_gain,
                region_trajectory=trajectory,
            )

        selected_region_sequence = region_sequence[batch_index, region_index]
        selected_region_coordinate = self.encoder.region_coordinates[region_index]
        region_delta = self._selected_delta(
            selected_region_sequence,
            selected_region_coordinate,
            auxiliary_state,
            region_active,
            REGION,
        )
        logits = auxiliary_base + region_delta

        selected_channel_sequence = x[batch_index, channel_index]
        selected_channel_coordinate = self.encoder.channel_coordinates[channel_index]
        channel_delta = self._selected_delta(
            selected_channel_sequence,
            selected_channel_coordinate,
            auxiliary_state,
            channel_active,
            CHANNEL,
        )
        logits = logits + channel_delta

        use_dense = self.training if dense_teacher is None else dense_teacher
        dense_region_logits = None
        dense_channel_logits = None
        if use_dense:
            dense_region_logits, dense_channel_logits = self._dense_candidates(
                auxiliary_base, auxiliary_state, region_sequence, x
            )
        action_mask = torch.stack((region_active, channel_active), dim=1)
        return CompactOutput(
            logits=logits,
            primary_logits=primary_logits,
            region_index=torch.where(
                region_active, region_index, torch.full_like(region_index, STOP)
            ),
            channel_index=torch.where(
                channel_active, channel_index, torch.full_like(channel_index, STOP)
            ),
            action_mask=action_mask,
            predicted_gain=torch.stack((region_value, channel_value), dim=1),
            region_gain=region_gain,
            channel_gain=channel_gain,
            region_trajectory=trajectory,
            dense_region_logits=dense_region_logits,
            dense_channel_logits=dense_channel_logits,
        )


def compact_objective(
    output: CompactOutput,
    labels: Tensor,
    config: CompactConfig,
    class_weight: Optional[Tensor] = None,
    channel_mask: Optional[Tensor] = None,
    label_smoothing: float = 0.0,
) -> CompactLoss:
    """Single-stage objective for the three compact modules."""

    if output.dense_region_logits is None or output.dense_channel_logits is None:
        raise ValueError("compact training requires dense_teacher=True")
    primary_per_row = F.cross_entropy(
        output.primary_logits, labels, weight=class_weight, reduction="none",
        label_smoothing=label_smoothing,
    )
    final_per_row = F.cross_entropy(
        output.logits, labels, weight=class_weight, reduction="none",
        label_smoothing=label_smoothing,
    )
    batch, regions, classes = output.dense_region_logits.shape
    region_ce = F.cross_entropy(
        output.dense_region_logits.reshape(batch * regions, classes),
        labels[:, None].expand(-1, regions).reshape(-1),
        weight=class_weight,
        reduction="none",
        label_smoothing=label_smoothing,
    ).reshape(batch, regions)
    channels = output.dense_channel_logits.shape[1]
    channel_ce = F.cross_entropy(
        output.dense_channel_logits.reshape(batch * channels, classes),
        labels[:, None].expand(-1, channels).reshape(-1),
        weight=class_weight,
        reduction="none",
        label_smoothing=label_smoothing,
    ).reshape(batch, channels)
    if channel_mask is None:
        channel_mask = torch.ones_like(channel_ce, dtype=torch.bool)
    valid_channel = channel_mask.to(dtype=channel_ce.dtype)
    candidate = 0.5 * (
        region_ce.mean()
        + (channel_ce * valid_channel).sum() / valid_channel.sum().clamp_min(1.0)
    )
    region_target = (primary_per_row[:, None] - region_ce).detach()
    channel_target = (primary_per_row[:, None] - channel_ce).detach()
    finite_region = torch.isfinite(output.region_gain)
    finite_channel = torch.isfinite(output.channel_gain) & channel_mask
    region_policy = F.smooth_l1_loss(
        output.region_gain[finite_region], region_target[finite_region]
    )
    channel_policy = F.smooth_l1_loss(
        output.channel_gain[finite_channel], channel_target[finite_channel]
    )
    policy = 0.5 * (region_policy + channel_policy)
    safety = F.relu(final_per_row - primary_per_row.detach()).mean()
    budget = output.action_mask.to(dtype=output.logits.dtype).sum(dim=1).mean()
    primary = primary_per_row.mean()
    final = final_per_row.mean()
    loss = (
        primary
        + final
        + config.candidate_weight * candidate
        + config.policy_weight * policy
        + config.safety_weight * safety
        + config.budget_weight * budget
    )
    return CompactLoss(loss, primary, final, candidate, policy, safety, budget)
