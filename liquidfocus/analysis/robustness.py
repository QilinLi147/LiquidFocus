"""Deterministic incomplete/noisy/irregular EEG perturbations for E5."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def random_channel_missing(
    channel_mask: Tensor,
    membership: Tensor,
    fraction: float,
    *,
    seed: int,
) -> Tensor:
    if not 0 <= fraction < 1:
        raise ValueError("channel missing fraction must be in [0,1)")
    result = channel_mask.clone().to(torch.bool)
    generator = np.random.default_rng(int(seed))
    membership_np = torch.as_tensor(membership).argmax(dim=0).cpu().numpy()
    for row in range(len(result)):
        observed = torch.nonzero(result[row], as_tuple=False).squeeze(1).cpu().numpy()
        remove_count = min(int(round(fraction * len(observed))), len(observed) - 1)
        if remove_count <= 0:
            continue
        for _attempt in range(100):
            remove = generator.choice(observed, size=remove_count, replace=False)
            candidate = result[row].clone()
            candidate[torch.as_tensor(remove, device=result.device)] = False
            regions = np.unique(membership_np[candidate.cpu().numpy()])
            if len(regions) >= 1:
                result[row] = candidate
                break
    return result


def contiguous_region_missing(
    channel_mask: Tensor,
    membership: Tensor,
    count: int,
    *,
    seed: int,
) -> Tensor:
    membership_value = torch.as_tensor(membership, device=channel_mask.device) > 0
    regions = membership_value.shape[0]
    if not 1 <= count < regions:
        raise ValueError("region missing count must leave at least one region")
    result = channel_mask.clone().to(torch.bool)
    generator = np.random.default_rng(int(seed))
    for row in range(len(result)):
        available = torch.nonzero(
            (membership_value & result[row][None]).any(dim=1),
            as_tuple=False,
        ).squeeze(1).cpu().numpy()
        remove_count = min(count, max(0, len(available) - 1))
        removed = generator.choice(available, size=remove_count, replace=False)
        result[row, membership_value[torch.as_tensor(removed, device=result.device)].any(dim=0)] = False
    return result


def random_time_missing(delta_t: Tensor, fraction: float, *, seed: int) -> Tensor:
    if not 0 <= fraction < 1:
        raise ValueError("time missing fraction must be in [0,1)")
    result = delta_t.clone()
    if result.ndim == 1:
        result = result[None]
    generator = np.random.default_rng(int(seed))
    for row in range(len(result)):
        observed = torch.nonzero(
            result[row] > 0, as_tuple=False
        ).squeeze(1).cpu().numpy()
        count = min(int(round(fraction * len(observed))), len(observed) - 1)
        if count > 0:
            removed = generator.choice(observed, size=count, replace=False)
            result[row, torch.as_tensor(removed, device=result.device)] = 0
    return result


def irregular_sampling(
    delta_t: Tensor,
    sigma: float,
    *,
    seed: int,
) -> Tensor:
    if sigma < 0:
        raise ValueError("log-normal sigma must be non-negative")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(delta_t.shape, generator=generator, dtype=torch.float32)
    multiplier = torch.exp(sigma * noise - 0.5 * sigma * sigma).to(delta_t.device, delta_t.dtype)
    return torch.where(delta_t > 0, delta_t * multiplier, delta_t)


def additive_channel_noise(
    x: Tensor,
    scale: Tensor,
    severity: float,
    *,
    seed: int,
    burst: bool = False,
) -> Tensor:
    if severity < 0:
        raise ValueError("noise severity must be non-negative")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(x.shape, generator=generator, dtype=torch.float32).to(x.device, x.dtype)
    scale_value = torch.as_tensor(scale, dtype=x.dtype, device=x.device)
    if scale_value.numel() == x.shape[-1]:
        scale_value = scale_value.reshape(1, 1, 1, -1)
    elif scale_value.shape == (x.shape[1], x.shape[-1]):
        scale_value = scale_value[None, :, None, :]
    else:
        raise ValueError("noise scale must be per-band or channel-by-band")
    if burst:
        burst_mask = torch.zeros(x.shape[:3], dtype=torch.bool)
        for row in range(x.shape[0]):
            start = int(torch.randint(max(1, x.shape[2] - 1), (1,), generator=generator))
            burst_mask[row, :, start : min(x.shape[2], start + 2)] = True
        noise = noise * burst_mask.to(x.device)[..., None]
    return x + float(severity) * scale_value * noise


def missing_bands(x: Tensor, bands: list[int], *, neutral: float = 0.0) -> Tensor:
    if not bands or any(index < 0 or index >= x.shape[-1] for index in bands):
        raise ValueError("band identities must be valid and non-empty")
    result = x.clone()
    result[..., bands] = float(neutral)
    return result
