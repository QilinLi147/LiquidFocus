"""Interventions and exact three-player attribution for E7."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np
import torch
from torch import Tensor


TRANSITIONS = ("N", "R", "E")


@dataclass(frozen=True)
class TransitionCoalition:
    node: bool
    recurrent: bool
    relation: bool

    @property
    def name(self) -> str:
        active = [
            label
            for label, enabled in zip(
                TRANSITIONS, (self.node, self.recurrent, self.relation)
            )
            if enabled
        ]
        return "+".join(active) if active else "empty"


def all_transition_coalitions() -> list[TransitionCoalition]:
    return [
        TransitionCoalition(bool(mask & 1), bool(mask & 2), bool(mask & 4))
        for mask in range(8)
    ]


def randomized_membership_preserving_counts(
    membership: Tensor,
    *,
    seed: int,
) -> Tensor:
    """Randomize channel-to-region membership while preserving region sizes."""

    value = torch.as_tensor(membership)
    if value.ndim != 2 or not torch.allclose(
        value.sum(dim=0), torch.ones(value.shape[1], device=value.device)
    ):
        raise ValueError("membership must have one unit-mass assignment per channel")
    labels = value.argmax(dim=0).detach().cpu().numpy()
    shuffled = np.random.default_rng(int(seed)).permutation(labels)
    result = torch.zeros_like(value)
    result[torch.as_tensor(shuffled, device=value.device), torch.arange(
        value.shape[1], device=value.device
    )] = 1
    if not torch.equal(
        torch.bincount(result.argmax(dim=0), minlength=value.shape[0]),
        torch.bincount(value.argmax(dim=0), minlength=value.shape[0]),
    ):
        raise AssertionError("random membership changed region counts")
    return result


def node_feature_owner_from_membership(
    membership: Tensor,
    *,
    bands: int = 5,
) -> Tensor:
    """Map channel mean/std blocks to regions under a structural null graph."""

    value = torch.as_tensor(membership, dtype=torch.float32)
    regions, channels = value.shape
    owner = torch.zeros(
        regions, channels * 2 * bands, dtype=value.dtype, device=value.device
    )
    labels = value.argmax(dim=0)
    width = 2 * bands
    for channel in range(channels):
        owner[int(labels[channel]), channel * width : (channel + 1) * width] = 1
    return owner


def shuffle_relation_pairs(
    relation: Tensor,
    *,
    regions: int,
    bands: int,
    seed: int,
) -> Tensor:
    """Shuffle region-pair identity within direction and band."""

    pairs = regions * (regions - 1) // 2
    expected = pairs * 3 * bands
    if relation.shape[-1] != expected:
        raise ValueError("relation feature dimension does not match schema")
    shaped = relation.reshape(*relation.shape[:-1], 3, pairs, bands)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    chunks = []
    for direction in range(3):
        order = torch.randperm(pairs, generator=generator).to(relation.device)
        chunks.append(shaped[..., direction, order, :])
    return torch.stack(chunks, dim=-3).reshape_as(relation)


def swap_forward_backward(
    relation: Tensor,
    *,
    regions: int,
    bands: int,
) -> Tensor:
    pairs = regions * (regions - 1) // 2
    expected = pairs * 3 * bands
    if relation.shape[-1] != expected:
        raise ValueError("relation feature dimension does not match schema")
    shaped = relation.reshape(*relation.shape[:-1], 3, pairs, bands)
    return shaped[..., [0, 2, 1], :, :].reshape_as(relation)


def _validate_coalition_values(
    values: Mapping[frozenset[str], float],
) -> None:
    expected = {
        frozenset(combo)
        for size in range(4)
        for combo in combinations(TRANSITIONS, size)
    }
    if set(values) != expected:
        missing = sorted("+".join(sorted(item)) or "empty" for item in expected - set(values))
        extra = sorted("+".join(sorted(item)) or "empty" for item in set(values) - expected)
        raise ValueError(f"coalition table mismatch; missing={missing}, extra={extra}")
    if not all(math.isfinite(float(value)) for value in values.values()):
        raise ValueError("coalition values must be finite")


def exact_shapley_three(
    values: Mapping[frozenset[str], float],
) -> dict[str, float]:
    """Exact Shapley contribution for N/R/E without Monte Carlo sampling."""

    _validate_coalition_values(values)
    players = frozenset(TRANSITIONS)
    result: dict[str, float] = {}
    factorial = math.factorial
    for player in TRANSITIONS:
        total = 0.0
        others = players - {player}
        for size in range(3):
            for subset_tuple in combinations(sorted(others), size):
                subset = frozenset(subset_tuple)
                weight = (
                    factorial(size) * factorial(3 - size - 1) / factorial(3)
                )
                total += weight * (
                    float(values[subset | {player}]) - float(values[subset])
                )
        result[player] = total
    return result


def exact_pair_interactions(
    values: Mapping[frozenset[str], float],
) -> dict[str, float]:
    """Exact second-order Shapley interaction for three transitions."""

    _validate_coalition_values(values)
    result: dict[str, float] = {}
    players = frozenset(TRANSITIONS)
    for first, second in combinations(TRANSITIONS, 2):
        remaining = players - {first, second}
        interaction = 0.0
        # Shapley interaction weights for n=3: |S|!(n-|S|-2)!/(n-1)!
        for size in range(2):
            for subset_tuple in combinations(sorted(remaining), size):
                subset = frozenset(subset_tuple)
                weight = math.factorial(size) * math.factorial(1 - size) / 2
                interaction += weight * (
                    float(values[subset | {first, second}])
                    - float(values[subset | {first}])
                    - float(values[subset | {second}])
                    + float(values[subset])
                )
        result[f"{first},{second}"] = interaction
    return result
