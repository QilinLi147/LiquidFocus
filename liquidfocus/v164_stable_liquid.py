"""Stable evidence as a causal observation of the regional liquid belief.

This module intentionally has no classifier.  Channel moments and anatomical
relations are normalised with train-only statistics, compressed by a low-rank
map, and injected into the existing regional liquid state update.  The only
classification head remains ``RegionalLiquidEncoder.primary_readout``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .v163_identity_liquid import exact_anatomical_membership_torch
from .v163_stable_evidence import exact_stable_evidence


def fit_stable_feature_normalisation(
    transformed: np.ndarray,
    train_indices: np.ndarray,
    sessions: np.ndarray,
    session3_weight: float,
    std_floor: float = 1e-6,
    fit_partition: str = "development_archive_train_only",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit Ada-compatible feature scaling using only the training partition.

    ``transformed`` is the value domain consumed by the stable observation:
    raw DE for SEED-family datasets and ``log1p`` DE for MPED.  Labels are not
    accepted by this API.
    """

    values = np.asarray(transformed, dtype=np.float32)
    index = np.asarray(train_indices, dtype=np.int64)
    session = np.asarray(sessions, dtype=np.int64)
    if values.ndim != 4 or values.shape[1:] != (62, 10, 5):
        raise ValueError("stable observation values must be [N,62,10,5]")
    if index.ndim != 1 or not len(index) or np.any(index < 0) or np.any(index >= len(values)):
        raise ValueError("invalid stable observation training indices")
    if session.shape != (len(values),):
        raise ValueError("stable observation session vector mismatch")
    if session3_weight <= 0 or std_floor <= 0:
        raise ValueError("stable observation weights/floor must be positive")
    if fit_partition not in {
        "development_archive_train_only", "train+development_union"
    }:
        raise ValueError("unknown stable observation fit partition")
    feature = exact_stable_evidence(values[index]).combined.astype(
        np.float64, copy=False
    )
    weight = np.where(
        session[index] == 2, float(session3_weight), 1.0
    ).astype(np.float64)
    denominator = float(weight.sum())
    mean = (feature * weight[:, None]).sum(axis=0) / denominator
    variance = (feature - mean[None]) ** 2
    variance = (variance * weight[:, None]).sum(axis=0) / denominator
    scale = np.maximum(np.sqrt(np.maximum(variance, 0.0)), float(std_floor))
    contract = {
        "feature_dimension": int(feature.shape[1]),
        "fit_partition": fit_partition,
        "fit_samples": int(len(index)),
        "session3_weight": float(session3_weight),
        "sample_weight_sum": denominator,
        "labels_used": False,
    }
    return mean.astype(np.float32), scale.astype(np.float32), contract


class StableEvidenceLiquidObservation(nn.Module):
    """Causal 935-D stable evidence injected only into liquid state updates."""

    def __init__(
        self,
        channels: int,
        bands: int,
        geometry_regions: int,
        width: int,
        rank: int = 32,
    ) -> None:
        super().__init__()
        if channels != 62 or bands != 5:
            raise ValueError("V164 stable observation requires 62 channels/5 bands")
        if rank <= 0 or rank > width:
            raise ValueError("stable observation rank must be in [1,width]")
        self.channels = channels
        self.bands = bands
        self.geometry_regions = geometry_regions
        self.width = width
        self.rank = rank
        self.channel_dimension = channels * bands * 2
        self.relation_dimension = 7 * 6 // 2 * 3 * bands
        self.feature_dimension = self.channel_dimension + self.relation_dimension
        self.register_buffer(
            "anatomical_membership", exact_anatomical_membership_torch()
        )
        self.register_buffer("pair_index", torch.triu_indices(7, 7, offset=1))
        self.register_buffer("input_mean", torch.zeros(channels, bands))
        self.register_buffer("input_scale", torch.ones(channels, bands))
        self.register_buffer("feature_mean", torch.zeros(self.feature_dimension))
        self.register_buffer("feature_scale", torch.ones(self.feature_dimension))
        self.down_projection = nn.Linear(self.feature_dimension, rank, bias=False)
        self.low_rank_norm = nn.LayerNorm(rank)
        self.up_projection = nn.Linear(
            rank, geometry_regions * width, bias=False
        )
        # Stable evidence should be visible to the liquid cell from epoch one;
        # it is still bounded and must pass through the same recurrent belief.
        self.injection_scale = nn.Parameter(torch.tensor(0.55))

    @torch.no_grad()
    def set_input_transform(self, mean: Tensor, scale: Tensor) -> None:
        def channel_band(value: Tensor, name: str) -> Tensor:
            tensor = torch.as_tensor(
                value, dtype=self.input_mean.dtype, device=self.input_mean.device
            )
            if tensor.numel() == self.bands:
                return tensor.reshape(1, self.bands).expand(
                    self.channels, self.bands
                )
            if tensor.numel() == self.channels * self.bands:
                return tensor.reshape(self.channels, self.bands)
            raise ValueError(
                f"stable observation input {name} must contain 5 or 310 values"
            )

        mean_value = channel_band(mean, "mean")
        scale_value = channel_band(scale, "scale")
        if not torch.isfinite(mean_value).all():
            raise ValueError("stable observation input mean must be finite")
        if not torch.isfinite(scale_value).all() or torch.any(scale_value <= 0):
            raise ValueError("stable observation input scale must be finite/positive")
        self.input_mean.copy_(mean_value)
        self.input_scale.copy_(scale_value)

    @torch.no_grad()
    def set_feature_transform(self, mean: Tensor, scale: Tensor) -> None:
        mean_value = torch.as_tensor(
            mean, dtype=self.feature_mean.dtype, device=self.feature_mean.device
        ).reshape(-1)
        scale_value = torch.as_tensor(
            scale, dtype=self.feature_scale.dtype, device=self.feature_scale.device
        ).reshape(-1)
        if mean_value.numel() != self.feature_dimension:
            raise ValueError("stable observation feature mean dimension mismatch")
        if scale_value.numel() != self.feature_dimension:
            raise ValueError("stable observation feature scale dimension mismatch")
        if not torch.isfinite(mean_value).all():
            raise ValueError("stable observation feature mean must be finite")
        if not torch.isfinite(scale_value).all() or torch.any(scale_value <= 0):
            raise ValueError("stable observation feature scale must be finite/positive")
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
        value = (covariance / denominator).clamp(-1.0, 1.0)
        return torch.where(
            (observed.sum(dim=1) > 0)[:, None, None],
            value,
            torch.zeros_like(value),
        )

    def causal_features(
        self, normalized: Tensor, spans: Tensor, channel_mask: Tensor
    ) -> Tensor:
        if normalized.ndim != 4 or normalized.shape[1:] != (
            self.channels, spans.shape[1], self.bands
        ):
            raise ValueError("stable observation input shape mismatch")
        if spans.shape != (normalized.shape[0], normalized.shape[2]):
            raise ValueError("stable observation span shape mismatch")
        if channel_mask.shape != normalized.shape[:2]:
            raise ValueError("stable observation channel mask mismatch")
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
            current_observed = observed[:, :stop]
            time_weight = current_observed.to(raw.dtype)[:, None, :, None]
            time_count = time_weight.sum(dim=2).clamp_min(1.0)
            prefix = raw[:, :, :stop]
            mean = (prefix * time_weight).sum(dim=2) / time_count
            centered = (prefix - mean[:, :, None]) * time_weight
            standard_deviation = (
                centered.square().sum(dim=2) / time_count
            ).clamp_min(0.0).sqrt()
            channel = torch.cat((mean, standard_deviation), dim=-1)
            channel = channel * channel_mask[:, :, None]
            region = anatomical[:, :stop]
            zero = self._masked_correlation(
                region[:, :, left], region[:, :, right], current_observed
            )
            if stop == 1:
                forward = torch.zeros_like(zero)
                backward = torch.zeros_like(zero)
            else:
                lag_observed = current_observed[:, :-1] & current_observed[:, 1:]
                forward = self._masked_correlation(
                    region[:, :-1, left], region[:, 1:, right], lag_observed
                )
                backward = self._masked_correlation(
                    region[:, :-1, right], region[:, 1:, left], lag_observed
                )
            relation = torch.cat((
                zero.flatten(1), forward.flatten(1), backward.flatten(1)
            ), dim=1)
            result.append(torch.cat((channel.flatten(1), relation), dim=1))
        feature = torch.stack(result, dim=1)
        if feature.shape[-1] != self.feature_dimension:
            raise AssertionError("stable observation feature dimension changed")
        if not torch.isfinite(feature).all():
            raise FloatingPointError("non-finite stable observation feature")
        return feature

    def forward(
        self,
        normalized: Tensor,
        spans: Tensor,
        channel_mask: Tensor,
        geometry_valid: Tensor,
    ) -> Tensor:
        feature = self.causal_features(normalized, spans, channel_mask)
        feature = (feature - self.feature_mean[None, None]) / self.feature_scale[
            None, None
        ]
        low_rank = self.low_rank_norm(self.down_projection(feature))
        update = self.up_projection(low_rank).reshape(
            normalized.shape[0], normalized.shape[2],
            self.geometry_regions, self.width,
        )
        update = update * geometry_valid[:, None, :, None]
        return torch.tanh(self.injection_scale) * update
