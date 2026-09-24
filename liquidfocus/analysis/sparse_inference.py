"""Sparse evaluation path for the released LiquidFocus model.

The optimized class retains the encoder/router/refiner contract and batches
the selected region and channel expert calls into one invocation.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ..compact import (
    CHANNEL, REGION, STOP, CompactConfig, CompactOutput, HierarchicalRouter,
)
from ..v186_evidence_decoupled import EvidenceDecoupledLiquidFocus
from .state_gates import (
    IFusionOutput,
    InstrumentedEvidenceDecoupledLiquidFocus,
)


class EvidenceOnlyRefinerLiquidFocus(
    InstrumentedEvidenceDecoupledLiquidFocus
):
    """Prevent the auxiliary refiner from copying the global primary state.

    Candidate supervision assigns the same label to every local candidate.  A
    primary-conditioned delta head can therefore learn a candidate-invariant
    correction and make REGION/CHANNEL selection mathematically irrelevant.
    V2 zeros only that shortcut at the refiner input; the router remains
    primary-conditioned and the final prediction remains exactly primary.
    """

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
        region_state = primary_state.new_zeros(
            primary_state.shape[0], self.regions, primary_state.shape[1]
        )
        channel_state = primary_state.new_zeros(
            primary_state.shape[0], self.channels, primary_state.shape[1]
        )
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
        delta = primary_state.new_zeros(primary_state.shape[0], self.classes)
        if not torch.any(active):
            return delta
        selected = torch.nonzero(active, as_tuple=False).squeeze(1)
        evidence = self.refiner.expert(sequence[selected], coordinates[selected])
        zero_state = primary_state.new_zeros(len(selected), primary_state.shape[1])
        delta[selected] = self.refiner.delta(zero_state, evidence, kind)
        return delta


def build_ifusion_evidence_only_revision_v2_model(
    *,
    classes: int,
    config: CompactConfig,
    channels: int = 62,
    time: int = 10,
    bands: int = 5,
    regions: int = 7,
) -> EvidenceOnlyRefinerLiquidFocus:
    return EvidenceOnlyRefinerLiquidFocus(
        classes=classes, channels=channels, time=time, bands=bands,
        regions=regions, config=config,
    )


class StopAwareHierarchicalRouter(HierarchicalRouter):
    """Separate sample-level action value from candidate ranking."""

    def __init__(self, bands: int, width: int, dropout: float) -> None:
        super().__init__(bands, width, dropout)
        hidden = max(8, width // 2)
        self.stop_head = torch.nn.Sequential(
            torch.nn.LayerNorm(width),
            torch.nn.Linear(width, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(
        self,
        x: Tensor,
        primary_state: Tensor,
        regional_state: Tensor,
        channel_coordinates: Tensor,
        channel_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        region_gain, channel_gain, channel_tokens = super().forward(
            x, primary_state, regional_state, channel_coordinates, channel_mask
        )
        action_value = self.stop_head(primary_state)
        return region_gain + action_value, channel_gain + action_value, channel_tokens


class StopAwareEvidenceOnlyRefinerLiquidFocus(
    EvidenceOnlyRefinerLiquidFocus
):
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
            classes=classes, channels=channels, time=time, bands=bands,
            regions=regions, config=config,
        )
        self.router = StopAwareHierarchicalRouter(
            bands, self.config.width, 0.0
        )
        if list(dict(self.named_children())) != ["encoder", "router", "refiner"]:
            raise AssertionError("revision-v2 model must retain exactly three modules")


def build_ifusion_stop_aware_revision_v2_model(
    *,
    classes: int,
    config: CompactConfig,
    channels: int = 62,
    time: int = 10,
    bands: int = 5,
    regions: int = 7,
) -> StopAwareEvidenceOnlyRefinerLiquidFocus:
    return StopAwareEvidenceOnlyRefinerLiquidFocus(
        classes=classes, channels=channels, time=time, bands=bands,
        regions=regions, config=config,
    )


class FusedSparseEvidenceDecoupledLiquidFocus(
    InstrumentedEvidenceDecoupledLiquidFocus
):
    """V186-compatible model with one batched sparse expert invocation."""

    def _fused_sparse_forward(
        self,
        x: Tensor,
        delta_t: Optional[Tensor],
        channel_mask: Tensor,
    ) -> IFusionOutput:
        primary_logits, primary_state, trajectory, region_sequence = self.encoder(
            x, delta_t, channel_mask
        )
        auxiliary_state = primary_state.detach()
        final_regional = trajectory[:, -1]
        # Execute the hierarchy literally: score REGION first and do not build
        # 62 channel tokens for rows whose first action is STOP.
        region_gain = self.router._score(
            auxiliary_state, final_regional.detach(), REGION
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

        channel_gain = primary_logits.new_full(
            (x.shape[0], self.channels), -torch.inf
        )
        if self.config.enable_router:
            active_rows = torch.nonzero(region_active, as_tuple=False).squeeze(1)
            if len(active_rows):
                channel_tokens = self.router._channel_tokens(
                    x[active_rows], self.encoder.channel_coordinates
                )
                active_gain = self.router._score(
                    auxiliary_state[active_rows], channel_tokens, CHANNEL
                )
                active_gain = active_gain.masked_fill(
                    ~channel_mask[active_rows], -torch.inf
                )
                channel_gain[active_rows] = active_gain
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
            # This rare ablation path is not part of E4; retaining the frozen
            # implementation avoids duplicating its no-refiner contract.
            return super().forward(
                x, delta_t=delta_t, channel_mask=channel_mask,
                dense_teacher=False,
            )

        selected_region_sequence = region_sequence[batch_index, region_index]
        selected_region_coordinate = self.encoder.region_coordinates[region_index]
        selected_channel_sequence = x[batch_index, channel_index]
        selected_channel_coordinate = self.encoder.channel_coordinates[channel_index]

        region_rows = torch.nonzero(region_active, as_tuple=False).squeeze(1)
        channel_rows = torch.nonzero(channel_active, as_tuple=False).squeeze(1)
        sequences = torch.cat((
            selected_region_sequence[region_rows],
            selected_channel_sequence[channel_rows],
        ), dim=0)
        coordinates = torch.cat((
            selected_region_coordinate[region_rows],
            selected_channel_coordinate[channel_rows],
        ), dim=0)
        states = torch.cat((
            auxiliary_state[region_rows], auxiliary_state[channel_rows],
        ), dim=0)
        kinds = torch.cat((
            torch.full_like(region_rows, REGION),
            torch.full_like(channel_rows, CHANNEL),
        ), dim=0)
        region_delta = primary_logits.new_zeros(x.shape[0], self.classes)
        channel_delta = primary_logits.new_zeros(x.shape[0], self.classes)
        if len(sequences):
            evidence = self.refiner.expert(sequences, coordinates)
            evidence = evidence + self.refiner.type_embedding(kinds)
            delta = self.refiner.max_residual * torch.tanh(
                self.refiner.delta_head(torch.cat((states, evidence), dim=-1))
            )
            split = len(region_rows)
            region_delta[region_rows] = delta[:split]
            channel_delta[channel_rows] = delta[split:]

        # Execute the complete evidence path for profiling/audit, while V186's
        # locked prediction-decoupling rule keeps final exactly primary.
        _evidence_logits = primary_logits + region_delta + channel_delta
        del _evidence_logits
        action_mask = torch.stack((region_active, channel_active), dim=1)
        trace = self.encoder.ifusion_trace
        stopped_region = torch.where(
            region_active, region_index, torch.full_like(region_index, STOP)
        )
        stopped_channel = torch.where(
            channel_active, channel_index, torch.full_like(channel_index, STOP)
        )
        return IFusionOutput(
            logits=primary_logits,
            primary_logits=primary_logits,
            final_logits=primary_logits,
            region_index=stopped_region,
            channel_index=stopped_channel,
            selected_region=stopped_region,
            selected_channel=stopped_channel,
            action_mask=action_mask,
            predicted_gain=torch.stack((region_value, channel_value), dim=1),
            region_gain=region_gain,
            channel_gain=channel_gain,
            dense_candidate_utilities=torch.cat((region_gain, channel_gain), dim=1),
            actual_expert_call_counts=action_mask.to(dtype=torch.int16),
            region_trajectory=trajectory,
            node_prefix_features=trace.node_prefix_features,
            relation_prefix_features=trace.relation_prefix_features,
            regional_slow_trajectory=trace.regional_slow_trajectory,
            regional_fast_trajectory=trace.regional_fast_trajectory,
            delta_t=trace.delta_t,
            observation_mask=trace.observation_mask,
            dense_region_logits=None,
            dense_channel_logits=None,
        )

    def forward(
        self,
        x: Tensor,
        delta_t: Optional[Tensor] = None,
        channel_mask: Optional[Tensor] = None,
        dense_teacher: Optional[bool] = None,
    ) -> IFusionOutput:
        # Training and dense enumeration remain byte-for-byte on the original
        # implementation.  Only explicit sparse evaluation uses the fused path.
        use_dense = self.training if dense_teacher is None else bool(dense_teacher)
        if use_dense:
            return super().forward(
                x, delta_t=delta_t, channel_mask=channel_mask,
                dense_teacher=dense_teacher,
            )
        if x.ndim != 4 or x.shape[1:] != (self.channels, self.time, self.bands):
            raise ValueError("x must be [B,C,T,F] with configured dimensions")
        if channel_mask is None:
            channel_mask = torch.ones(
                x.shape[0], self.channels, dtype=torch.bool, device=x.device
            )
        if channel_mask.shape != x.shape[:2]:
            raise ValueError("channel_mask must be [B,C]")
        if torch.any(~channel_mask.any(dim=1)):
            raise ValueError("every sample needs at least one observed channel")
        return self._fused_sparse_forward(x, delta_t, channel_mask)


def build_ifusion_revision_v2_model(
    *,
    classes: int,
    config: CompactConfig,
    channels: int = 62,
    time: int = 10,
    bands: int = 5,
    regions: int = 7,
) -> FusedSparseEvidenceDecoupledLiquidFocus:
    return FusedSparseEvidenceDecoupledLiquidFocus(
        classes=classes, channels=channels, time=time, bands=bands,
        regions=regions, config=config,
    )


class ProductionFusedSparseLiquidFocus(EvidenceDecoupledLiquidFocus):
    """Production V186 with literal hierarchical STOP and one expert batch."""

    def forward(
        self,
        x: Tensor,
        delta_t: Optional[Tensor] = None,
        channel_mask: Optional[Tensor] = None,
        dense_teacher: Optional[bool] = None,
    ) -> CompactOutput:
        use_dense = self.training if dense_teacher is None else bool(dense_teacher)
        if use_dense:
            return super().forward(
                x, delta_t=delta_t, channel_mask=channel_mask,
                dense_teacher=dense_teacher,
            )
        if x.ndim != 4 or x.shape[1:] != (self.channels, self.time, self.bands):
            raise ValueError("x must be [B,C,T,F] with configured dimensions")
        if channel_mask is None:
            channel_mask = torch.ones(
                x.shape[0], self.channels, dtype=torch.bool, device=x.device
            )
        if channel_mask.shape != x.shape[:2] or torch.any(~channel_mask.any(dim=1)):
            raise ValueError("channel_mask must keep at least one channel per row")
        primary_logits, primary_state, trajectory, region_sequence = self.encoder(
            x, delta_t, channel_mask
        )
        auxiliary_state = primary_state.detach()
        final_regional = trajectory[:, -1]
        region_gain = self.router._score(
            auxiliary_state, final_regional.detach(), REGION
        )
        available_region = torch.einsum(
            "rc,bc->br", self.encoder.region_membership,
            channel_mask.to(dtype=x.dtype),
        ) > 0
        region_gain = region_gain.masked_fill(~available_region, -torch.inf)
        batch_index = torch.arange(x.shape[0], device=x.device)
        region_index = region_gain.argmax(dim=1)
        region_value = region_gain[batch_index, region_index]
        region_active = region_value > self.stop_threshold

        channel_gain = primary_logits.new_full(
            (x.shape[0], self.channels), -torch.inf
        )
        active_rows = torch.nonzero(region_active, as_tuple=False).squeeze(1)
        if len(active_rows):
            tokens = self.router._channel_tokens(
                x[active_rows], self.encoder.channel_coordinates
            )
            active_gain = self.router._score(
                auxiliary_state[active_rows], tokens, CHANNEL
            ).masked_fill(~channel_mask[active_rows], -torch.inf)
            channel_gain[active_rows] = active_gain
        parent = self.encoder.channel_region[None] == region_index[:, None]
        legal_channel = parent & channel_mask
        routed = channel_gain.masked_fill(~legal_channel, -torch.inf)
        channel_index = routed.argmax(dim=1)
        channel_value = routed[batch_index, channel_index]
        channel_active = region_active & (channel_value > self.stop_threshold)

        region_rows = torch.nonzero(region_active, as_tuple=False).squeeze(1)
        channel_rows = torch.nonzero(channel_active, as_tuple=False).squeeze(1)
        sequences = torch.cat((
            region_sequence[batch_index, region_index][region_rows],
            x[batch_index, channel_index][channel_rows],
        ), dim=0)
        coordinates = torch.cat((
            self.encoder.region_coordinates[region_index][region_rows],
            self.encoder.channel_coordinates[channel_index][channel_rows],
        ), dim=0)
        states = torch.cat((
            auxiliary_state[region_rows], auxiliary_state[channel_rows],
        ), dim=0)
        kinds = torch.cat((
            torch.full_like(region_rows, REGION),
            torch.full_like(channel_rows, CHANNEL),
        ), dim=0)
        if len(sequences):
            evidence = self.refiner.expert(sequences, coordinates)
            evidence = evidence + self.refiner.type_embedding(kinds)
            _delta = self.refiner.max_residual * torch.tanh(
                self.refiner.delta_head(torch.cat((states, evidence), dim=-1))
            )
            del _delta
        action_mask = torch.stack((region_active, channel_active), dim=1)
        return CompactOutput(
            logits=primary_logits,
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
        )


def build_ifusion_production_fused_revision_v2_model(
    *, classes: int, config: CompactConfig, channels: int = 62,
    time: int = 10, bands: int = 5, regions: int = 7,
) -> ProductionFusedSparseLiquidFocus:
    return ProductionFusedSparseLiquidFocus(
        classes=classes, channels=channels, time=time, bands=bands,
        regions=regions, config=config,
    )
