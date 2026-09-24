"""Neutral, mask-respecting localization interventions for E3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass
class InterventionBatch:
    values: Tensor
    channel_mask: Tensor
    target_mask: Tensor
    replacement_channel: Tensor | None = None


def _neutral_values(x: Tensor, neutral: Tensor | float) -> Tensor:
    value = torch.as_tensor(neutral, dtype=x.dtype, device=x.device)
    if value.numel() == 1:
        return value.expand_as(x)
    if value.shape == (x.shape[1], x.shape[3]):
        return value[None, :, None, :].expand_as(x)
    if value.shape == (1, x.shape[1], 1, x.shape[3]):
        return value.expand_as(x)
    raise ValueError("neutral value must be scalar or channel-by-band")


def intervene_selected_channels(
    x: Tensor,
    channel_mask: Tensor,
    selected_channel: Tensor,
    *,
    mode: str,
    neutral: Tensor | float = 0.0,
) -> InterventionBatch:
    if x.ndim != 4 or channel_mask.shape != x.shape[:2]:
        raise ValueError("expected x [B,C,T,F] and channel_mask [B,C]")
    if selected_channel.shape != (x.shape[0],):
        raise ValueError("selected_channel must be [B]")
    target = torch.zeros_like(channel_mask, dtype=torch.bool)
    active = selected_channel >= 0
    target[torch.arange(x.shape[0], device=x.device)[active], selected_channel[active].long()] = True
    target &= channel_mask
    if mode not in {"deletion", "only"}:
        raise ValueError("channel intervention mode must be deletion or only")
    keep = channel_mask & (~target if mode == "deletion" else target)
    neutral_full = _neutral_values(x, neutral)
    values = torch.where(keep[:, :, None, None], x, neutral_full)
    return InterventionBatch(values, keep, target)


def intervene_selected_regions(
    x: Tensor,
    channel_mask: Tensor,
    selected_region: Tensor,
    membership: Tensor,
    *,
    mode: str,
    neutral: Tensor | float = 0.0,
) -> InterventionBatch:
    membership_value = torch.as_tensor(membership, device=x.device) > 0
    if membership_value.shape[1] != x.shape[1]:
        raise ValueError("membership channel count mismatch")
    active = selected_region >= 0
    safe = selected_region.clamp_min(0).long()
    target = membership_value[safe] & active[:, None] & channel_mask
    if mode not in {"deletion", "only"}:
        raise ValueError("region intervention mode must be deletion or only")
    keep = channel_mask & (~target if mode == "deletion" else target)
    if torch.any(~keep.any(dim=1)):
        # A full selected-only region always contains at least one observed
        # channel; deletion can remove the only observed region and is retained
        # as an explicit invalid intervention rather than silently altered.
        raise ValueError("intervention removed all observed channels")
    values = torch.where(
        keep[:, :, None, None], x, _neutral_values(x, neutral)
    )
    return InterventionBatch(values, keep, target)


def matched_random_channels(
    channel_mask: Tensor,
    selected_region: Tensor,
    membership: Tensor,
    *,
    seed: int,
) -> Tensor:
    """Select one legal random channel within each learned region."""

    membership_value = torch.as_tensor(membership).detach().cpu() > 0
    observed = channel_mask.detach().cpu().to(torch.bool)
    regions = selected_region.detach().cpu().long()
    result = torch.full_like(regions, -1)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for row in range(len(result)):
        if int(regions[row]) < 0:
            continue
        candidates = torch.nonzero(
            membership_value[int(regions[row])] & observed[row],
            as_tuple=False,
        ).squeeze(1)
        if len(candidates):
            result[row] = int(candidates[
                torch.randint(len(candidates), (1,), generator=generator)
            ].item())
    return result.to(channel_mask.device)


def counterfactual_channel_swap(
    x: Tensor,
    channel_mask: Tensor,
    selected_channel: Tensor,
    membership: Tensor,
    *,
    seed: int,
) -> InterventionBatch:
    region_by_channel = torch.as_tensor(membership).argmax(dim=0).to(x.device)
    selected_region = torch.where(
        selected_channel >= 0,
        region_by_channel[selected_channel.clamp_min(0).long()],
        torch.full_like(selected_channel, -1),
    )
    replacement = matched_random_channels(
        channel_mask, selected_region, membership, seed=seed
    )
    # Ensure replacement differs when a same-region alternative exists.
    generator = np.random.default_rng(int(seed))
    membership_cpu = torch.as_tensor(membership).argmax(dim=0).cpu().numpy()
    observed_cpu = channel_mask.cpu().numpy()
    for row in range(len(replacement)):
        selected = int(selected_channel[row])
        if selected < 0:
            continue
        candidates = np.flatnonzero(
            (membership_cpu == membership_cpu[selected]) & observed_cpu[row]
        )
        candidates = candidates[candidates != selected]
        if len(candidates):
            replacement[row] = int(generator.choice(candidates))
    values = x.clone()
    target = torch.zeros_like(channel_mask, dtype=torch.bool)
    for row in range(x.shape[0]):
        selected, alternate = int(selected_channel[row]), int(replacement[row])
        if selected >= 0 and alternate >= 0:
            values[row, selected] = x[row, alternate]
            target[row, selected] = True
    return InterventionBatch(values, channel_mask.clone(), target, replacement)
