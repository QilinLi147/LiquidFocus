"""Causal identity-preserving observations for the V163 liquid encoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .geometry import SEED_CHANNELS


def exact_anatomical_membership_torch() -> Tensor:
    """Locked seven-region mapping, including CB1/CB2 Central precedence."""

    membership = torch.zeros(7, len(SEED_CHANNELS), dtype=torch.float32)
    for channel, name in enumerate(SEED_CHANNELS):
        if name.endswith("z"):
            side = "mid"
        else:
            digits = "".join(character for character in name if character.isdigit())
            side = "mid" if not digits else ("left" if int(digits) % 2 else "right")
        if name.startswith(("Fp", "AF")) or (
            name.startswith("F") and not name.startswith(("FC", "FT"))
        ):
            if side == "left":
                membership[0, channel] = 1.0
            elif side == "right":
                membership[1, channel] = 1.0
            else:
                membership[0:2, channel] = 0.5
        elif name.startswith(("FT", "T", "TP")):
            membership[2 if side == "left" else 3, channel] = 1.0
        elif name.startswith(("FC", "C", "CP")):
            membership[4, channel] = 1.0
        elif name.startswith(("P", "PO")):
            membership[5, channel] = 1.0
        elif name.startswith(("O", "CB")):
            membership[6, channel] = 1.0
        else:
            raise ValueError(f"unmapped channel {name}")
    if not torch.equal(membership.sum(dim=0), torch.ones(len(SEED_CHANNELS))):
        raise AssertionError("anatomical membership is not exhaustive")
    return membership


class CausalIdentityObservation(nn.Module):
    """Low-rank causal observation injected only into liquid state updates.

    This module has no classifier and never emits logits. Its complete-window
    statistics mirror the stable evidence family, while every earlier step is
    computed from the prefix available at that step only.
    """

    def __init__(
        self,
        channels: int,
        bands: int,
        geometry_regions: int,
        width: int,
        include_relations: bool,
        rank: int = 8,
    ) -> None:
        super().__init__()
        if channels != len(SEED_CHANNELS) or bands != 5:
            raise ValueError("V163 identity observation requires 62 channels/5 bands")
        if rank <= 0:
            raise ValueError("observation rank must be positive")
        self.channels = channels
        self.bands = bands
        self.geometry_regions = geometry_regions
        self.width = width
        self.include_relations = bool(include_relations)
        self.channel_dimension = channels * bands * 2
        self.relation_dimension = 7 * 6 // 2 * 3 * bands
        self.feature_dimension = self.channel_dimension + (
            self.relation_dimension if self.include_relations else 0
        )
        membership = exact_anatomical_membership_torch()
        self.register_buffer("anatomical_membership", membership)
        self.register_buffer("pair_index", torch.triu_indices(7, 7, offset=1))
        self.register_buffer("input_mean", torch.zeros(channels, bands))
        self.register_buffer("input_scale", torch.ones(channels, bands))
        self.feature_norm = nn.LayerNorm(self.feature_dimension)
        self.down_projection = nn.Linear(
            self.feature_dimension, rank, bias=False
        )
        self.up_projection = nn.Linear(
            rank, geometry_regions * width, bias=False
        )
        self.injection_scale = nn.Parameter(torch.tensor(0.10))

    @torch.no_grad()
    def set_input_transform(self, mean: Tensor, scale: Tensor) -> None:
        def channel_band_transform(value: Tensor, *, name: str) -> Tensor:
            tensor = torch.as_tensor(
                value,
                dtype=self.input_mean.dtype,
                device=self.input_mean.device,
            )
            if tensor.numel() == self.bands:
                # MPED's locked development protocol fits one transform per
                # frequency band.  Broadcast that train-only transform across
                # channel identities without changing the underlying values.
                return tensor.reshape(1, self.bands).expand(
                    self.channels, self.bands
                )
            if tensor.numel() == self.channels * self.bands:
                # SEED-family archives retain a separate transform for every
                # ordered channel-band pair.
                return tensor.reshape(self.channels, self.bands)
            raise ValueError(
                f"observation input {name} must contain either "
                f"{self.bands} band-wise or "
                f"{self.channels * self.bands} channel-band values"
            )

        mean_value = channel_band_transform(mean, name="mean")
        scale_value = channel_band_transform(scale, name="scale")
        if not torch.isfinite(mean_value).all():
            raise ValueError("observation input mean must be finite")
        if not torch.isfinite(scale_value).all() or torch.any(scale_value <= 0):
            raise ValueError("observation input scale must be finite and positive")
        self.input_mean.copy_(mean_value)
        self.input_scale.copy_(scale_value)

    @staticmethod
    def _masked_correlation(
        first: Tensor, second: Tensor, observed: Tensor
    ) -> Tensor:
        """Population correlation for [B,L,P,F] with a [B,L] mask."""

        weight = observed.to(dtype=first.dtype)[:, :, None, None]
        count = weight.sum(dim=1).clamp_min(1.0)
        first_mean = (first * weight).sum(dim=1) / count
        second_mean = (second * weight).sum(dim=1) / count
        first_centered = (first - first_mean[:, None]) * weight
        second_centered = (second - second_mean[:, None]) * weight
        covariance = (first_centered * second_centered).sum(dim=1) / count
        first_variance = first_centered.square().sum(dim=1) / count
        second_variance = second_centered.square().sum(dim=1) / count
        value = covariance / torch.sqrt(
            (first_variance + 1e-4) * (second_variance + 1e-4)
        )
        valid = observed.sum(dim=1) > 0
        return torch.where(
            valid[:, None, None], value.clamp(-1.0, 1.0), torch.zeros_like(value)
        )

    def causal_features(
        self,
        normalized: Tensor,
        spans: Tensor,
        channel_mask: Tensor,
    ) -> Tensor:
        """Return prefix features [B,T,620|935] without future access."""

        if normalized.ndim != 4 or normalized.shape[1] != self.channels or (
            normalized.shape[3] != self.bands
        ):
            raise ValueError("identity observation input shape mismatch")
        if spans.ndim != 2 or spans.shape != (
            normalized.shape[0], normalized.shape[2]
        ):
            raise ValueError("identity observation span shape mismatch")
        if channel_mask.shape != normalized.shape[:2]:
            raise ValueError("identity observation channel mask mismatch")
        raw = (
            normalized * self.input_scale[None, :, None, :]
            + self.input_mean[None, :, None, :]
        )
        raw = torch.where(
            channel_mask[:, :, None, None], raw, torch.zeros_like(raw)
        )
        anatomical_weight = (
            self.anatomical_membership[None]
            * channel_mask.to(dtype=raw.dtype)[:, None]
        )
        anatomical_count = anatomical_weight.sum(dim=2).clamp_min(1.0)
        anatomical = torch.einsum(
            "brc,bctf->brtf", anatomical_weight, raw
        ) / anatomical_count[:, :, None, None]
        anatomical = anatomical.permute(0, 2, 1, 3).contiguous()
        left, right = self.pair_index[0], self.pair_index[1]
        observed = spans > 0
        result = []
        for stop in range(1, normalized.shape[2] + 1):
            prefix_observed = observed[:, :stop]
            time_weight = prefix_observed.to(dtype=raw.dtype)[:, None, :, None]
            time_count = time_weight.sum(dim=2).clamp_min(1.0)
            prefix = raw[:, :, :stop]
            mean = (prefix * time_weight).sum(dim=2) / time_count
            centered = (prefix - mean[:, :, None]) * time_weight
            variance = centered.square().sum(dim=2) / time_count
            channel = torch.cat((mean, variance.clamp_min(0.0).sqrt()), dim=-1)
            channel = channel * channel_mask[:, :, None]
            blocks = [channel.flatten(1)]
            if self.include_relations:
                current = anatomical[:, :stop]
                zero = self._masked_correlation(
                    current[:, :, left], current[:, :, right], prefix_observed
                )
                if stop == 1:
                    forward = torch.zeros_like(zero)
                    backward = torch.zeros_like(zero)
                else:
                    lag_observed = (
                        prefix_observed[:, :-1] & prefix_observed[:, 1:]
                    )
                    forward = self._masked_correlation(
                        current[:, :-1, left],
                        current[:, 1:, right],
                        lag_observed,
                    )
                    backward = self._masked_correlation(
                        current[:, :-1, right],
                        current[:, 1:, left],
                        lag_observed,
                    )
                blocks.append(torch.cat((
                    zero.flatten(1), forward.flatten(1), backward.flatten(1)
                ), dim=1))
            result.append(torch.cat(blocks, dim=1))
        features = torch.stack(result, dim=1)
        if features.shape[-1] != self.feature_dimension:
            raise AssertionError("identity observation dimension changed")
        if not torch.isfinite(features).all():
            raise FloatingPointError("non-finite identity observation")
        return features

    def forward(
        self,
        normalized: Tensor,
        spans: Tensor,
        channel_mask: Tensor,
        geometry_valid: Tensor,
    ) -> Tensor:
        features = self.causal_features(normalized, spans, channel_mask)
        low_rank = self.down_projection(self.feature_norm(features))
        update = self.up_projection(low_rank).reshape(
            normalized.shape[0], normalized.shape[2],
            self.geometry_regions, self.width,
        )
        update = update * geometry_valid[:, None, :, None]
        return torch.tanh(self.injection_scale) * update
