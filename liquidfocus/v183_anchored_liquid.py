"""V183 anchor-preserving regional liquid model.

A train-fitted stable affine coordinate is decomposed over seven anatomical
regions and written into the encoder's slow liquid state.  It is never exposed
as a parallel classifier: the deployed prediction is the single readout of
the fused slow/fast regional liquid belief.  A bounded fast rhythm residual
can improve the stable coordinate without changing its scale arbitrarily.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .compact import CompactConfig
from .v182_dual_timescale import (
    DualTimescaleLiquidFocus, DualTimescaleRegionalLiquidEncoder,
)


class AnchorPreservingReadout(nn.Module):
    """One readout: exact regional stable coordinate plus bounded rhythm."""

    def __init__(self, classes: int, regions: int, width: int, dropout: float) -> None:
        super().__init__()
        self.classes, self.regions, self.width = classes, regions, width
        self.residual_bound = 0.10
        fast_dimension = regions * 4 * (width // 2)
        self.rhythm = nn.Sequential(
            nn.LayerNorm(fast_dimension), nn.Dropout(dropout),
            nn.Linear(fast_dimension, classes),
        )
        self.register_buffer("residual_enabled", torch.tensor(True))

    def forward(self, flattened: Tensor) -> Tensor:
        summary = flattened.reshape(-1, self.regions, 4, self.width)
        anchor = summary[:, :, 0, : self.classes].sum(dim=1)
        fast = summary[:, :, :, self.width // 2 :].reshape(summary.shape[0], -1)
        residual = self.residual_bound * torch.tanh(self.rhythm(fast))
        return anchor + self.residual_enabled.to(residual.dtype) * residual

    @torch.no_grad()
    def set_residual_enabled(self, enabled: bool) -> None:
        self.residual_enabled.fill_(bool(enabled))


def _feature_ownership() -> Tensor:
    """Return 7 x 935 non-negative ownership with columns summing to one."""

    from .v163_identity_liquid import exact_anatomical_membership_torch

    membership = exact_anatomical_membership_torch()
    owner = torch.zeros(7, 935, dtype=torch.float32)
    channel_region = membership.argmax(dim=0)
    for channel in range(62):
        owner[int(channel_region[channel]), channel * 10 : (channel + 1) * 10] = 1.0
    pair = torch.triu_indices(7, 7, offset=1)
    offset = 620
    for direction in range(3):
        for pair_index in range(pair.shape[1]):
            start = offset + (direction * pair.shape[1] + pair_index) * 5
            owner[int(pair[0, pair_index]), start : start + 5] = 0.5
            owner[int(pair[1, pair_index]), start : start + 5] = 0.5
    if owner.shape != (7, 935) or not torch.equal(owner.sum(dim=0), torch.ones(935)):
        raise AssertionError("V183 stable feature ownership changed")
    return owner


class AnchoredDualTimescaleEncoder(DualTimescaleRegionalLiquidEncoder):
    def __init__(
        self, classes: int, channels: int, bands: int, regions: int,
        config: CompactConfig,
    ) -> None:
        super().__init__(classes, channels, bands, regions, config)
        if classes > self.slow_width:
            raise ValueError("V183 class coordinate exceeds slow liquid width")
        self.register_buffer("feature_owner", _feature_ownership())
        self.register_buffer("anchor_coefficient", torch.zeros(classes, 935))
        self.register_buffer("anchor_intercept", torch.zeros(classes))
        self.register_buffer("anchor_fitted", torch.tensor(False))
        self.primary_readout = AnchorPreservingReadout(
            classes, regions, self.width, config.dropout
        )

    @torch.no_grad()
    def set_anchor_state(self, coefficient: Tensor, intercept: Tensor) -> None:
        coefficient = torch.as_tensor(
            coefficient, dtype=self.anchor_coefficient.dtype,
            device=self.anchor_coefficient.device,
        )
        intercept = torch.as_tensor(
            intercept, dtype=self.anchor_intercept.dtype,
            device=self.anchor_intercept.device,
        )
        if coefficient.shape != self.anchor_coefficient.shape:
            raise ValueError("V183 anchor coefficient shape mismatch")
        if intercept.shape != self.anchor_intercept.shape:
            raise ValueError("V183 anchor intercept shape mismatch")
        if not torch.isfinite(coefficient).all() or not torch.isfinite(intercept).all():
            raise ValueError("V183 anchor state must be finite")
        self.anchor_coefficient.copy_(coefficient)
        self.anchor_intercept.copy_(intercept)
        self.anchor_fitted.fill_(True)

    def _anchor_observation(
        self, x: Tensor, spans: Tensor, channel_mask: Tensor,
    ) -> Tensor:
        if not bool(self.anchor_fitted):
            raise RuntimeError("V183 anchor must be fitted on the current train partition")
        feature = self.stable_observation.causal_features(x, spans, channel_mask)
        standardized = (
            feature - self.stable_observation.feature_mean[None, None]
        ) / self.stable_observation.feature_scale[None, None]
        regional = torch.einsum(
            "btd,kd,rd->btrk", standardized,
            self.anchor_coefficient, self.feature_owner,
        )
        regional = regional + self.anchor_intercept[None, None, None] / self.regions
        observation = x.new_zeros(
            x.shape[0], x.shape[2], self.regions, self.slow_width
        )
        observation[..., : self.classes] = regional
        return observation

    def forward(
        self, x: Tensor, delta_t: Optional[Tensor], channel_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        region_sequence, local, valid = self._local_sequence(x, channel_mask)
        spans = self._spans(x, delta_t, self.use_delta_t)
        fast_observation = self.fast_input(local) * valid[:, None, :, None]
        anchor_observation = self._anchor_observation(x, spans, channel_mask)
        batch, time = x.shape[0], x.shape[2]
        slow_state = x.new_zeros(batch, self.regions, self.slow_width)
        fast_state = x.new_zeros(batch * self.regions, self.fast_width)
        trajectory = []
        valid_flat = valid.reshape(-1, 1)
        for step in range(time):
            span = spans[:, step : step + 1, None].expand(-1, -1, self.regions).reshape(-1, 1)
            observed = (span > 0) & valid_flat
            # Event-driven slow liquid state: the causal prefix coordinate is
            # the equilibrium and zero elapsed time is an exact hold.
            slow_candidate = anchor_observation[:, step]
            slow_state = torch.where(
                observed.reshape(batch, self.regions, 1), slow_candidate, slow_state
            )
            safe = torch.where(observed, span, torch.ones_like(span))
            fast_value = (
                fast_observation[:, step].reshape(-1, self.fast_width)
                + self.slow_to_fast(self.slow_norm(slow_state.reshape(-1, self.slow_width)))
            )
            fast_candidate = self.fast_cell(fast_value, fast_state, safe)
            fast_state = torch.where(observed, fast_candidate, fast_state)
            fused = torch.cat((slow_state.reshape(-1, self.slow_width), self.fast_norm(fast_state)), dim=-1)
            trajectory.append(fused.reshape(batch, self.regions, self.width))
        regional = torch.stack(trajectory, dim=1)
        weight = (spans > 0).to(dtype=x.dtype)
        count = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (regional * weight[:, :, None, None]).sum(dim=1) / count[:, :, None]
        centered = regional - mean[:, None]
        std = ((centered.square() * weight[:, :, None, None]).sum(dim=1) / count[:, :, None]).clamp_min(0).add(1e-6).sqrt()
        position = torch.linspace(-1.0, 1.0, time, device=x.device, dtype=x.dtype)
        trend = (centered * position[None, :, None, None] * weight[:, :, None, None]).sum(dim=1) / count[:, :, None]
        last_index = ((spans > 0).to(torch.int64) * torch.arange(1, time + 1, device=x.device)[None]).argmax(dim=1)
        last = regional[torch.arange(batch, device=x.device), last_index]
        summary = torch.cat((last, mean, std, trend), dim=-1)
        primary_logits = self.primary_readout(summary.flatten(1))
        primary_state = summary.reshape(batch, self.regions, 4, self.width).mean(dim=(1, 2))
        return primary_logits, primary_state, regional, region_sequence


class AnchoredDualTimescaleLiquidFocus(DualTimescaleLiquidFocus):
    def __init__(
        self, classes: int, channels: int = 62, time: int = 10,
        bands: int = 5, regions: int = 7,
        config: Optional[CompactConfig] = None,
    ) -> None:
        super().__init__(
            classes=classes, channels=channels, time=time, bands=bands,
            regions=regions, config=config,
        )
        self.encoder = AnchoredDualTimescaleEncoder(
            classes, channels, bands, regions, self.config
        )
        if list(dict(self.named_children())) != ["encoder", "router", "refiner"]:
            raise AssertionError("V183 must expose exactly encoder/router/refiner")


def build_v183_model(
    *, classes: int, config: CompactConfig, channels: int = 62,
    time: int = 10, bands: int = 5,
) -> AnchoredDualTimescaleLiquidFocus:
    return AnchoredDualTimescaleLiquidFocus(
        classes=classes, channels=channels, time=time, bands=bands, config=config
    )
