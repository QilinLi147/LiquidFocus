"""Channel-schema adapter for external Information Fusion confirmation.

The historical 62-channel/5-band path is delegated to the frozen V186
builder.  A non-canonical schema changes only geometry-shaped buffers and the
stable observation/anchor dimensions; the three-module encoder/router/refiner
hierarchy and prediction-preserving output contract remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from ..compact import CompactConfig
from ..geometry import SEED_CHANNELS, default_channel_coordinates
from .state_gates import (
    GatedScoreCenteredLiquidEncoder,
    InstrumentedEvidenceDecoupledLiquidFocus,
    build_ifusion_v186_model,
)


REGION_NAMES = (
    "frontal_left", "frontal_right", "temporal_left", "temporal_right",
    "central", "parietal", "occipital",
)


def anatomical_membership(channel_names: Sequence[str]) -> Tensor:
    """Map standard 10-20/10-10 names to the frozen seven-region ontology."""

    membership = torch.zeros(7, len(channel_names), dtype=torch.float32)
    for channel, raw_name in enumerate(channel_names):
        name = str(raw_name).strip()
        if not name:
            raise ValueError("channel names must be non-empty")
        digits = "".join(character for character in name if character.isdigit())
        if name.lower().endswith("z") or not digits:
            side = "mid"
        else:
            side = "left" if int(digits) % 2 else "right"
        if name.startswith(("Fp", "FP", "AF")) or (
            name.startswith("F") and not name.startswith(("FC", "FT"))
        ):
            if side == "left":
                membership[0, channel] = 1.0
            elif side == "right":
                membership[1, channel] = 1.0
            else:
                membership[0:2, channel] = 0.5
        elif name.startswith(("FT", "T", "TP", "A")):
            if side == "mid":
                raise ValueError(f"ambiguous temporal channel side: {name}")
            membership[2 if side == "left" else 3, channel] = 1.0
        elif name.startswith(("FC", "C", "CP")):
            membership[4, channel] = 1.0
        elif name.startswith(("P", "PO")):
            membership[5, channel] = 1.0
        elif name.startswith(("O", "CB")):
            membership[6, channel] = 1.0
        else:
            raise ValueError(f"unmapped channel {name}")
    if not torch.allclose(membership.sum(dim=0), torch.ones(len(channel_names))):
        raise AssertionError("schema membership is not exhaustive")
    if torch.any(membership.sum(dim=1) == 0):
        raise ValueError("every frozen anatomical region needs at least one channel")
    return membership


@dataclass(frozen=True)
class EEGSchema:
    name: str
    channel_names: tuple[str, ...]
    channel_coordinates: Tensor
    bands: int = 5
    regions: int = 7

    def validate(self) -> "EEGSchema":
        coordinates = torch.as_tensor(self.channel_coordinates, dtype=torch.float32)
        if coordinates.shape != (len(self.channel_names), 3):
            raise ValueError("schema coordinates must have shape [channels,3]")
        if not torch.isfinite(coordinates).all():
            raise ValueError("schema coordinates must be finite")
        if self.bands != 5:
            raise ValueError("the registered external adapter freezes five bands")
        if self.regions != 7:
            raise ValueError("the registered external adapter freezes seven regions")
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("schema channel names must be unique")
        anatomical_membership(self.channel_names)
        return self

    @property
    def channels(self) -> int:
        return len(self.channel_names)


def seed_62_schema() -> EEGSchema:
    return EEGSchema(
        name="seed_62x5_frozen",
        channel_names=tuple(SEED_CHANNELS),
        channel_coordinates=default_channel_coordinates().clone(),
    )


def _feature_ownership(membership: Tensor, bands: int) -> Tensor:
    channels = membership.shape[1]
    channel_dimension = channels * bands * 2
    relation_dimension = 7 * 6 // 2 * 3 * bands
    owner = torch.zeros(7, channel_dimension + relation_dimension)
    channel_region = membership.argmax(dim=0)
    for channel in range(channels):
        start = channel * bands * 2
        owner[int(channel_region[channel]), start : start + bands * 2] = 1.0
    pair = torch.triu_indices(7, 7, offset=1)
    for direction in range(3):
        for pair_index in range(pair.shape[1]):
            start = channel_dimension + (direction * pair.shape[1] + pair_index) * bands
            owner[int(pair[0, pair_index]), start : start + bands] = 0.5
            owner[int(pair[1, pair_index]), start : start + bands] = 0.5
    if not torch.equal(owner.sum(dim=0), torch.ones(owner.shape[1])):
        raise AssertionError("schema feature ownership changed")
    return owner


class SchemaStableEvidenceObservation(nn.Module):
    """Dynamic-channel form of the frozen causal stable observation."""

    def __init__(self, schema: EEGSchema, width: int, rank: int = 32) -> None:
        super().__init__()
        schema.validate()
        self.channels = schema.channels
        self.bands = schema.bands
        self.geometry_regions = schema.regions
        self.width = width
        self.rank = rank
        self.channel_dimension = self.channels * self.bands * 2
        self.relation_dimension = 7 * 6 // 2 * 3 * self.bands
        self.feature_dimension = self.channel_dimension + self.relation_dimension
        self.register_buffer("anatomical_membership", anatomical_membership(schema.channel_names))
        self.register_buffer("pair_index", torch.triu_indices(7, 7, offset=1))
        self.register_buffer("input_mean", torch.zeros(self.channels, self.bands))
        self.register_buffer("input_scale", torch.ones(self.channels, self.bands))
        self.register_buffer("feature_mean", torch.zeros(self.feature_dimension))
        self.register_buffer("feature_scale", torch.ones(self.feature_dimension))
        self.down_projection = nn.Linear(self.feature_dimension, rank, bias=False)
        self.low_rank_norm = nn.LayerNorm(rank)
        self.up_projection = nn.Linear(rank, self.geometry_regions * width, bias=False)
        self.injection_scale = nn.Parameter(torch.tensor(0.55))

    @torch.no_grad()
    def set_input_transform(self, mean: Tensor, scale: Tensor) -> None:
        def reshape(value: Tensor, name: str) -> Tensor:
            tensor = torch.as_tensor(value, dtype=self.input_mean.dtype, device=self.input_mean.device)
            if tensor.numel() == self.bands:
                return tensor.reshape(1, self.bands).expand(self.channels, self.bands)
            if tensor.numel() == self.channels * self.bands:
                return tensor.reshape(self.channels, self.bands)
            raise ValueError(f"schema input {name} has the wrong dimension")
        mean_value, scale_value = reshape(mean, "mean"), reshape(scale, "scale")
        if not torch.isfinite(mean_value).all():
            raise ValueError("schema input mean must be finite")
        if not torch.isfinite(scale_value).all() or torch.any(scale_value <= 0):
            raise ValueError("schema input scale must be finite and positive")
        self.input_mean.copy_(mean_value)
        self.input_scale.copy_(scale_value)

    @torch.no_grad()
    def set_feature_transform(self, mean: Tensor, scale: Tensor) -> None:
        mean_value = torch.as_tensor(mean, dtype=self.feature_mean.dtype, device=self.feature_mean.device).reshape(-1)
        scale_value = torch.as_tensor(scale, dtype=self.feature_scale.dtype, device=self.feature_scale.device).reshape(-1)
        if mean_value.numel() != self.feature_dimension or scale_value.numel() != self.feature_dimension:
            raise ValueError("schema feature transform dimension mismatch")
        if not torch.isfinite(mean_value).all():
            raise ValueError("schema feature mean must be finite")
        if not torch.isfinite(scale_value).all() or torch.any(scale_value <= 0):
            raise ValueError("schema feature scale must be finite and positive")
        self.feature_mean.copy_(mean_value)
        self.feature_scale.copy_(scale_value)

    @staticmethod
    def _masked_correlation(first: Tensor, second: Tensor, observed: Tensor) -> Tensor:
        weight = observed.to(dtype=first.dtype)[:, :, None, None]
        count = weight.sum(dim=1).clamp_min(1.0)
        left_mean = (first * weight).sum(dim=1) / count
        right_mean = (second * weight).sum(dim=1) / count
        left = (first - left_mean[:, None]) * weight
        right = (second - right_mean[:, None]) * weight
        covariance = (left * right).sum(dim=1) / count
        denominator = torch.sqrt(
            (left.square().sum(dim=1) / count + 1e-4)
            * (right.square().sum(dim=1) / count + 1e-4)
        )
        valid = observed.sum(dim=1) > 0
        return torch.where(valid[:, None, None], (covariance / denominator).clamp(-1.0, 1.0), torch.zeros_like(covariance))

    def causal_features(self, normalized: Tensor, spans: Tensor, channel_mask: Tensor) -> Tensor:
        if normalized.ndim != 4 or normalized.shape[1] != self.channels or normalized.shape[3] != self.bands:
            raise ValueError("schema stable observation input shape mismatch")
        if spans.shape != (normalized.shape[0], normalized.shape[2]):
            raise ValueError("schema stable observation span shape mismatch")
        if channel_mask.shape != normalized.shape[:2]:
            raise ValueError("schema stable observation channel mask mismatch")
        raw = normalized * self.input_scale[None, :, None] + self.input_mean[None, :, None]
        raw = torch.where(channel_mask[:, :, None, None], raw, torch.zeros_like(raw))
        anatomical_weight = self.anatomical_membership[None] * channel_mask.to(raw.dtype)[:, None]
        anatomical_count = anatomical_weight.sum(dim=2).clamp_min(1.0)
        anatomical = torch.einsum("brc,bctf->brtf", anatomical_weight, raw) / anatomical_count[:, :, None, None]
        anatomical = anatomical.permute(0, 2, 1, 3).contiguous()
        left, right = self.pair_index[0], self.pair_index[1]
        observed = spans > 0
        result = []
        for stop in range(1, normalized.shape[2] + 1):
            current_observed = observed[:, :stop]
            time_weight = current_observed.to(raw.dtype)[:, None, :, None]
            time_count = time_weight.sum(dim=2).clamp_min(1.0)
            prefix = raw[:, :, :stop]
            mean = (prefix * time_weight).sum(dim=2) / time_count
            centered = (prefix - mean[:, :, None]) * time_weight
            standard_deviation = (centered.square().sum(dim=2) / time_count).clamp_min(0.0).sqrt()
            channel = torch.cat((mean, standard_deviation), dim=-1) * channel_mask[:, :, None]
            region = anatomical[:, :stop]
            zero = self._masked_correlation(region[:, :, left], region[:, :, right], current_observed)
            if stop == 1:
                forward = torch.zeros_like(zero)
                backward = torch.zeros_like(zero)
            else:
                lag_observed = current_observed[:, :-1] & current_observed[:, 1:]
                forward = self._masked_correlation(region[:, :-1, left], region[:, 1:, right], lag_observed)
                backward = self._masked_correlation(region[:, :-1, right], region[:, 1:, left], lag_observed)
            relation = torch.cat((zero.flatten(1), forward.flatten(1), backward.flatten(1)), dim=1)
            result.append(torch.cat((channel.flatten(1), relation), dim=1))
        feature = torch.stack(result, dim=1)
        if feature.shape[-1] != self.feature_dimension or not torch.isfinite(feature).all():
            raise FloatingPointError("invalid schema stable feature")
        return feature

    def forward(self, normalized: Tensor, spans: Tensor, channel_mask: Tensor, geometry_valid: Tensor) -> Tensor:
        feature = self.causal_features(normalized, spans, channel_mask)
        feature = (feature - self.feature_mean[None, None]) / self.feature_scale[None, None]
        low_rank = self.low_rank_norm(self.down_projection(feature))
        update = self.up_projection(low_rank).reshape(normalized.shape[0], normalized.shape[2], self.geometry_regions, self.width)
        return torch.tanh(self.injection_scale) * update * geometry_valid[:, None, :, None]


class SchemaGatedScoreCenteredEncoder(GatedScoreCenteredLiquidEncoder):
    def __init__(self, classes: int, schema: EEGSchema, config: CompactConfig) -> None:
        # Construct the frozen parameter families, then replace only
        # schema-shaped state before any training or evaluation.
        super().__init__(classes, 62, schema.bands, schema.regions, config)
        coordinates = torch.as_tensor(schema.channel_coordinates, dtype=torch.float32).clone()
        membership = anatomical_membership(schema.channel_names).to(coordinates)
        region_coordinates = membership @ coordinates / membership.sum(dim=1, keepdim=True).clamp_min(1.0)
        self.channels = schema.channels
        self.bands = schema.bands
        self.channel_coordinates = coordinates
        self.region_membership = membership
        self.channel_region = membership.argmax(dim=0)
        self.region_coordinates = region_coordinates
        self.stable_observation = SchemaStableEvidenceObservation(schema, self.slow_width, rank=32)
        feature_dimension = self.stable_observation.feature_dimension
        self.feature_owner = _feature_ownership(membership, schema.bands)
        self.anchor_coefficient = torch.zeros(classes, feature_dimension)
        self.anchor_fitted.fill_(False)


class SchemaInstrumentedLiquidFocus(InstrumentedEvidenceDecoupledLiquidFocus):
    def __init__(self, classes: int, schema: EEGSchema, time: int, config: CompactConfig) -> None:
        schema.validate()
        super().__init__(classes=classes, channels=62, time=time, bands=schema.bands, regions=schema.regions, config=config)
        self.channels = schema.channels
        self.bands = schema.bands
        self.regions = schema.regions
        self.encoder = SchemaGatedScoreCenteredEncoder(classes, schema, self.config)
        if list(dict(self.named_children())) != ["encoder", "router", "refiner"]:
            raise AssertionError("schema adapter changed the three-module hierarchy")


def build_schema_ifusion_model(*, classes: int, schema: EEGSchema, config: CompactConfig, time: int = 10) -> InstrumentedEvidenceDecoupledLiquidFocus:
    """Build canonical V186 exactly, or its registered external schema form."""

    schema.validate()
    canonical = seed_62_schema()
    if schema.channel_names == canonical.channel_names and torch.equal(
        torch.as_tensor(schema.channel_coordinates), canonical.channel_coordinates
    ):
        return build_ifusion_v186_model(classes=classes, config=config, channels=62, time=time, bands=5, regions=7)
    return SchemaInstrumentedLiquidFocus(classes, schema, time, config)
