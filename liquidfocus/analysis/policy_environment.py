"""Finite-horizon constrained evidence-acquisition environment for E2/E8."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class RewardContract:
    action_cost: float
    latency_cost: float
    useful_gain_delta: float
    max_actions: int = 2

    def validate(self) -> "RewardContract":
        if self.action_cost < 0 or self.latency_cost < 0:
            raise ValueError("cost coefficients must be non-negative")
        if self.useful_gain_delta < 0:
            raise ValueError("useful gain threshold must be non-negative")
        if self.max_actions != 2:
            raise ValueError("the frozen evidence horizon is exactly at most two")
        return self


@dataclass
class DensePathTable:
    primary_nll: Tensor
    region_nll: Tensor
    path_nll: Tensor
    region_utility: Tensor
    path_utility: Tensor
    legal_path: Tensor


def per_sample_nll(logits: Tensor, labels: Tensor) -> Tensor:
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("logit/label batch mismatch")
    return F.cross_entropy(logits, labels.to(torch.long), reduction="none")


def enumerate_dense_paths(
    primary_logits: Tensor,
    dense_region_logits: Tensor,
    dense_channel_logits: Tensor,
    labels: Tensor,
    membership: Tensor,
    channel_mask: Tensor | None = None,
) -> DensePathTable:
    """Enumerate all legal STOP/REGION/REGION->CHANNEL counterfactuals.

    Frozen dense candidates store ``primary + region_delta`` and ``primary +
    channel_delta``.  A two-action path is therefore their additive evidence
    composition, ``region + channel - primary``.
    """

    batch, regions, classes = dense_region_logits.shape
    channels = dense_channel_logits.shape[1]
    if primary_logits.shape != (batch, classes):
        raise ValueError("primary and region candidate shapes disagree")
    if dense_channel_logits.shape != (batch, channels, classes):
        raise ValueError("invalid channel candidate shape")
    membership_value = torch.as_tensor(
        membership, device=primary_logits.device
    )
    if membership_value.shape != (regions, channels):
        raise ValueError("membership shape does not match candidates")
    if channel_mask is None:
        channel_mask = torch.ones(
            batch, channels, dtype=torch.bool, device=primary_logits.device
        )
    if channel_mask.shape != (batch, channels):
        raise ValueError("channel_mask must be [B,C]")
    legal = (membership_value > 0)[None] & channel_mask[:, None]
    combined = (
        dense_region_logits[:, :, None, :]
        + dense_channel_logits[:, None, :, :]
        - primary_logits[:, None, None, :]
    )
    primary_nll = per_sample_nll(primary_logits, labels)
    region_labels = labels[:, None].expand(-1, regions).reshape(-1)
    region_nll = per_sample_nll(
        dense_region_logits.reshape(batch * regions, classes), region_labels
    ).reshape(batch, regions)
    path_labels = labels[:, None, None].expand(-1, regions, channels).reshape(-1)
    path_nll = per_sample_nll(
        combined.reshape(batch * regions * channels, classes), path_labels
    ).reshape(batch, regions, channels)
    path_nll = path_nll.masked_fill(~legal, torch.inf)
    return DensePathTable(
        primary_nll=primary_nll,
        region_nll=region_nll,
        path_nll=path_nll,
        region_utility=primary_nll[:, None] - region_nll,
        path_utility=primary_nll[:, None, None] - path_nll,
        legal_path=legal,
    )


def gather_policy_utility(
    table: DensePathTable,
    region_index: Tensor,
    channel_index: Tensor,
    action_mask: Tensor,
) -> Tensor:
    batch = table.primary_nll.shape[0]
    if action_mask.shape != (batch, 2):
        raise ValueError("action_mask must be [B,2]")
    if torch.any(action_mask[:, 1] & ~action_mask[:, 0]):
        raise ValueError("channel action cannot occur after region STOP")
    index = torch.arange(batch, device=table.primary_nll.device)
    safe_region = region_index.clamp_min(0).to(torch.long)
    safe_channel = channel_index.clamp_min(0).to(torch.long)
    region_utility = table.region_utility[index, safe_region]
    path_utility = table.path_utility[index, safe_region, safe_channel]
    if torch.any(action_mask[:, 1] & ~table.legal_path[index, safe_region, safe_channel]):
        raise ValueError("policy selected an illegal region-channel path")
    zero = torch.zeros_like(region_utility)
    return torch.where(
        action_mask[:, 1],
        path_utility,
        torch.where(action_mask[:, 0], region_utility, zero),
    )


def label_oracle_policy(table: DensePathTable, budget: int = 2) -> tuple[Tensor, Tensor, Tensor]:
    if budget not in {0, 1, 2}:
        raise ValueError("budget must be 0, 1, or 2")
    batch, regions, channels = table.path_utility.shape
    device = table.path_utility.device
    if budget == 0:
        return (
            torch.full((batch,), -1, dtype=torch.long, device=device),
            torch.full((batch,), -1, dtype=torch.long, device=device),
            torch.zeros(batch, 2, dtype=torch.bool, device=device),
        )
    if budget == 1:
        value, region = table.region_utility.max(dim=1)
        active = value > 0
        return (
            torch.where(active, region, torch.full_like(region, -1)),
            torch.full_like(region, -1),
            torch.stack((active, torch.zeros_like(active)), dim=1),
        )
    flat_value, flat_index = table.path_utility.reshape(batch, -1).max(dim=1)
    path_region = flat_index // channels
    path_channel = flat_index % channels
    region_value, region_only = table.region_utility.max(dim=1)
    choose_path = (flat_value > region_value) & (flat_value > 0)
    choose_region = (region_value >= flat_value) & (region_value > 0)
    region = torch.where(choose_path, path_region, region_only)
    region = torch.where(
        choose_path | choose_region, region, torch.full_like(region, -1)
    )
    channel = torch.where(choose_path, path_channel, torch.full_like(path_channel, -1))
    return region, channel, torch.stack((choose_path | choose_region, choose_path), dim=1)


def matched_random_policy(
    action_mask: Tensor,
    membership: Tensor,
    channel_mask: Tensor,
    *,
    repeats: int,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Draw legal random paths with the learned per-sample 0/1/2 budget."""

    if repeats < 1:
        raise ValueError("repeats must be positive")
    batch, channels = channel_mask.shape
    regions = membership.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    region_result = torch.full((repeats, batch), -1, dtype=torch.long)
    channel_result = torch.full((repeats, batch), -1, dtype=torch.long)
    mask_result = action_mask.detach().cpu()[None].expand(repeats, -1, -1).clone()
    membership_cpu = torch.as_tensor(membership).detach().cpu() > 0
    observed_cpu = channel_mask.detach().cpu().to(torch.bool)
    for repeat in range(repeats):
        for row in range(batch):
            if not bool(mask_result[repeat, row, 0]):
                continue
            available_region = torch.nonzero(
                (membership_cpu & observed_cpu[row][None]).any(dim=1),
                as_tuple=False,
            ).squeeze(1)
            choice = int(available_region[
                torch.randint(len(available_region), (1,), generator=generator)
            ].item())
            region_result[repeat, row] = choice
            if bool(mask_result[repeat, row, 1]):
                legal_channel = torch.nonzero(
                    membership_cpu[choice] & observed_cpu[row],
                    as_tuple=False,
                ).squeeze(1)
                channel_result[repeat, row] = int(legal_channel[
                    torch.randint(len(legal_channel), (1,), generator=generator)
                ].item())
    return (
        region_result.to(action_mask.device),
        channel_result.to(action_mask.device),
        mask_result.to(action_mask.device),
    )
