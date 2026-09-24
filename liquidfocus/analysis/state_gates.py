"""Prediction-preserving V186 state gates and evaluation instrumentation.

This module does not modify the historical V186 implementation.  It subclasses
the frozen encoder, keeps the same parameters and arithmetic in the default
path, and records the model-computation states required by the Information
Fusion experiments.  The recorded states are model states, not observations of
an error-free biological latent state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ..compact import CompactConfig, CompactOutput
from ..v186_evidence_decoupled import (
    EvidenceDecoupledLiquidFocus,
    ScoreCenteredLiquidEncoder,
)


NODE_DIM_62X5 = 620
RELATION_DIM_7X5 = 315


@dataclass(frozen=True)
class StateGateConfig:
    """Evaluation-only gates for the frozen state-transition graph.

    Values in ``[0, 1]`` permit continuous sensitivity checks, while the main
    experiments use exact zero/one settings.  ``temporal_mode='liquid'`` is the
    frozen recurrent update, ``'hold'`` removes the transition, and ``'static'``
    uses the current observation without recurrent carry-over.
    """

    node: float = 1.0
    relation: float = 1.0
    slow_anchor: float = 1.0
    fast_liquid: float = 1.0
    use_delta_t: bool = True
    temporal_mode: str = "liquid"

    def validate(self) -> "StateGateConfig":
        for name in ("node", "relation", "slow_anchor", "fast_liquid"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} gate must be in [0,1]")
        if self.temporal_mode not in {"liquid", "hold", "static"}:
            raise ValueError("temporal_mode must be liquid, hold, or static")
        return self


@dataclass
class EncoderTrace:
    node_prefix_features: Tensor
    relation_prefix_features: Tensor
    regional_slow_trajectory: Tensor
    regional_fast_trajectory: Tensor
    delta_t: Tensor
    observation_mask: Tensor


@dataclass
class IFusionOutput:
    """Auditable evaluation output; final logits remain exactly primary."""

    logits: Tensor
    primary_logits: Tensor
    final_logits: Tensor
    region_index: Tensor
    channel_index: Tensor
    selected_region: Tensor
    selected_channel: Tensor
    action_mask: Tensor
    predicted_gain: Tensor
    region_gain: Tensor
    channel_gain: Tensor
    dense_candidate_utilities: Tensor
    actual_expert_call_counts: Tensor
    region_trajectory: Tensor
    node_prefix_features: Tensor
    relation_prefix_features: Tensor
    regional_slow_trajectory: Tensor
    regional_fast_trajectory: Tensor
    delta_t: Tensor
    observation_mask: Tensor
    dense_region_logits: Optional[Tensor] = None
    dense_channel_logits: Optional[Tensor] = None


class GatedScoreCenteredLiquidEncoder(ScoreCenteredLiquidEncoder):
    """V186 encoder with exact block gates and non-persistent traces."""

    def __init__(
        self,
        classes: int,
        channels: int,
        bands: int,
        regions: int,
        config: CompactConfig,
    ) -> None:
        super().__init__(classes, channels, bands, regions, config)
        self.register_buffer("ifusion_node_gate", torch.tensor(1.0), persistent=False)
        self.register_buffer("ifusion_relation_gate", torch.tensor(1.0), persistent=False)
        self.register_buffer("ifusion_slow_gate", torch.tensor(1.0), persistent=False)
        self.register_buffer("ifusion_fast_gate", torch.tensor(1.0), persistent=False)
        self._ifusion_use_delta_t = True
        self._ifusion_temporal_mode = "liquid"
        self._ifusion_trace: Optional[EncoderTrace] = None

    @torch.no_grad()
    def set_state_gates(self, gates: StateGateConfig) -> None:
        value = gates.validate()
        self.ifusion_node_gate.fill_(float(value.node))
        self.ifusion_relation_gate.fill_(float(value.relation))
        self.ifusion_slow_gate.fill_(float(value.slow_anchor))
        self.ifusion_fast_gate.fill_(float(value.fast_liquid))
        self._ifusion_use_delta_t = bool(value.use_delta_t)
        self._ifusion_temporal_mode = value.temporal_mode

    @property
    def ifusion_trace(self) -> EncoderTrace:
        if self._ifusion_trace is None:
            raise RuntimeError("no IFusion trace is available before forward")
        return self._ifusion_trace

    def _gated_anchor_observation(
        self,
        feature: Tensor,
    ) -> Tensor:
        if not bool(self.anchor_fitted):
            raise RuntimeError("V186 anchor must be fitted on the current partition")
        standardized = (
            feature - self.stable_observation.feature_mean[None, None]
        ) / self.stable_observation.feature_scale[None, None]
        gated = self.gated_standardized_features(standardized)
        regional = torch.einsum(
            "btd,kd,rd->btrk",
            gated,
            self.anchor_coefficient,
            self.feature_owner,
        )
        regional = (
            regional
            + self.anchor_intercept[None, None, None] / self.regions
        )
        observation = feature.new_zeros(
            feature.shape[0], feature.shape[1], self.regions, self.slow_width
        )
        observation[..., : self.classes] = regional
        if float(self.ifusion_slow_gate) != 1.0:
            observation = observation * self.ifusion_slow_gate
        return observation

    def gated_standardized_features(self, standardized: Tensor) -> Tensor:
        """Apply exact N/E block gates without changing feature ordering."""

        node_dim = self.stable_observation.channel_dimension
        relation_dim = self.stable_observation.relation_dimension
        if standardized.shape[-1] != node_dim + relation_dim:
            raise AssertionError("stable feature partition changed")
        node = standardized[..., :node_dim]
        relation = standardized[..., node_dim:]
        # Skip multiply-by-one in the frozen path to preserve the original
        # arithmetic byte-for-byte wherever PyTorch kernels are deterministic.
        if float(self.ifusion_node_gate) != 1.0:
            node = node * self.ifusion_node_gate
        if float(self.ifusion_relation_gate) != 1.0:
            relation = relation * self.ifusion_relation_gate
        return torch.cat((node, relation), dim=-1)

    def forward(
        self,
        x: Tensor,
        delta_t: Optional[Tensor],
        channel_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        region_sequence, local, valid = self._local_sequence(x, channel_mask)
        spans = self._spans(x, delta_t, self.use_delta_t)
        if not self._ifusion_use_delta_t:
            # Preserve zero-span missingness while removing the magnitude of
            # observed elapsed-time intervals.
            spans = torch.where(spans > 0, torch.ones_like(spans), spans)
        fast_observation = self.fast_input(local) * valid[:, None, :, None]
        feature = self.stable_observation.causal_features(
            x, spans, channel_mask
        )
        node_dim = self.stable_observation.channel_dimension
        node_prefix = feature[..., :node_dim]
        relation_prefix = feature[..., node_dim:]
        anchor_observation = self._gated_anchor_observation(feature)

        batch, time = x.shape[0], x.shape[2]
        slow_state = x.new_zeros(batch, self.regions, self.slow_width)
        fast_state = x.new_zeros(batch * self.regions, self.fast_width)
        trajectory = []
        slow_trajectory = []
        fast_trajectory = []
        valid_flat = valid.reshape(-1, 1)
        for step in range(time):
            span = spans[:, step : step + 1, None].expand(
                -1, -1, self.regions
            ).reshape(-1, 1)
            observed = (span > 0) & valid_flat
            slow_candidate = anchor_observation[:, step]
            slow_state = torch.where(
                observed.reshape(batch, self.regions, 1),
                slow_candidate,
                slow_state,
            )
            safe = torch.where(observed, span, torch.ones_like(span))
            fast_value = (
                fast_observation[:, step].reshape(-1, self.fast_width)
                + self.slow_to_fast(
                    self.slow_norm(slow_state.reshape(-1, self.slow_width))
                )
            )
            if self._ifusion_temporal_mode == "liquid":
                fast_candidate = self.fast_cell(fast_value, fast_state, safe)
            elif self._ifusion_temporal_mode == "static":
                fast_candidate = torch.tanh(fast_value)
            else:
                fast_candidate = fast_state
            fast_state = torch.where(observed, fast_candidate, fast_state)
            slow_view = slow_state.reshape(batch, self.regions, self.slow_width)
            fast_view = self.fast_norm(fast_state).reshape(
                batch, self.regions, self.fast_width
            )
            if float(self.ifusion_fast_gate) != 1.0:
                fast_view = fast_view * self.ifusion_fast_gate
            fused = torch.cat((slow_view, fast_view), dim=-1)
            slow_trajectory.append(slow_view)
            fast_trajectory.append(fast_view)
            trajectory.append(fused)

        regional = torch.stack(trajectory, dim=1)
        slow_values = torch.stack(slow_trajectory, dim=1)
        fast_values = torch.stack(fast_trajectory, dim=1)
        weight = (spans > 0).to(dtype=x.dtype)
        count = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (
            regional * weight[:, :, None, None]
        ).sum(dim=1) / count[:, :, None]
        centered = regional - mean[:, None]
        std = (
            (centered.square() * weight[:, :, None, None]).sum(dim=1)
            / count[:, :, None]
        ).clamp_min(0).add(1e-6).sqrt()
        position = torch.linspace(
            -1.0, 1.0, time, device=x.device, dtype=x.dtype
        )
        trend = (
            centered
            * position[None, :, None, None]
            * weight[:, :, None, None]
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
        self._ifusion_trace = EncoderTrace(
            node_prefix_features=node_prefix,
            relation_prefix_features=relation_prefix,
            regional_slow_trajectory=slow_values,
            regional_fast_trajectory=fast_values,
            delta_t=spans,
            observation_mask=spans > 0,
        )
        return primary_logits, primary_state, regional, region_sequence


class InstrumentedEvidenceDecoupledLiquidFocus(
    EvidenceDecoupledLiquidFocus
):
    """Three-module V186 with additive Information Fusion outputs."""

    def __init__(
        self,
        classes: int,
        channels: int = 62,
        time: int = 10,
        bands: int = 5,
        regions: int = 7,
        config: Optional[CompactConfig] = None,
    ) -> None:
        super().__init__(
            classes=classes,
            channels=channels,
            time=time,
            bands=bands,
            regions=regions,
            config=config,
        )
        self.encoder = GatedScoreCenteredLiquidEncoder(
            classes, channels, bands, regions, self.config
        )
        if list(dict(self.named_children())) != ["encoder", "router", "refiner"]:
            raise AssertionError("IFusion V186 must retain exactly three modules")

    def set_state_gates(self, gates: StateGateConfig) -> None:
        self.encoder.set_state_gates(gates)

    def forward(
        self,
        x: Tensor,
        delta_t: Optional[Tensor] = None,
        channel_mask: Optional[Tensor] = None,
        dense_teacher: Optional[bool] = None,
    ) -> IFusionOutput:
        base: CompactOutput = super().forward(
            x,
            delta_t=delta_t,
            channel_mask=channel_mask,
            dense_teacher=dense_teacher,
        )
        trace = self.encoder.ifusion_trace
        if not torch.equal(base.logits, base.primary_logits):
            raise AssertionError("IFusion evidence path changed the primary prediction")
        dense_utilities = torch.cat((base.region_gain, base.channel_gain), dim=1)
        call_counts = base.action_mask.to(dtype=torch.int16)
        return IFusionOutput(
            logits=base.primary_logits,
            primary_logits=base.primary_logits,
            final_logits=base.primary_logits,
            region_index=base.region_index,
            channel_index=base.channel_index,
            selected_region=base.region_index,
            selected_channel=base.channel_index,
            action_mask=base.action_mask,
            predicted_gain=base.predicted_gain,
            region_gain=base.region_gain,
            channel_gain=base.channel_gain,
            dense_candidate_utilities=dense_utilities,
            actual_expert_call_counts=call_counts,
            region_trajectory=base.region_trajectory,
            node_prefix_features=trace.node_prefix_features,
            relation_prefix_features=trace.relation_prefix_features,
            regional_slow_trajectory=trace.regional_slow_trajectory,
            regional_fast_trajectory=trace.regional_fast_trajectory,
            delta_t=trace.delta_t,
            observation_mask=trace.observation_mask,
            dense_region_logits=base.dense_region_logits,
            dense_channel_logits=base.dense_channel_logits,
        )


def build_ifusion_v186_model(
    *,
    classes: int,
    config: CompactConfig,
    channels: int = 62,
    time: int = 10,
    bands: int = 5,
    regions: int = 7,
) -> InstrumentedEvidenceDecoupledLiquidFocus:
    """Build the additive evaluator for an unchanged V186 state dict."""

    return InstrumentedEvidenceDecoupledLiquidFocus(
        classes=classes,
        channels=channels,
        time=time,
        bands=bands,
        regions=regions,
        config=config,
    )
