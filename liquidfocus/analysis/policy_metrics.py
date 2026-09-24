"""Participant-ready constrained policy metrics for E8."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .policy_environment import (
    DensePathTable,
    RewardContract,
    gather_policy_utility,
    label_oracle_policy,
)


@dataclass
class PolicyEpisodeMetrics:
    utility: Tensor
    expected_return: Tensor
    actions: Tensor
    latency_normalized: Tensor
    regret: Tensor
    oracle_normalized_regret: Tensor
    false_stop: Tensor
    unnecessary_continue: Tensor
    action_budget_violation: Tensor
    latency_budget_violation: Tensor


def evaluate_policy_episodes(
    table: DensePathTable,
    region_index: Tensor,
    channel_index: Tensor,
    action_mask: Tensor,
    contract: RewardContract,
    *,
    normalized_latency: Tensor | None = None,
    latency_budget: float = float("inf"),
    random_return: Tensor | None = None,
) -> PolicyEpisodeMetrics:
    contract.validate()
    utility = gather_policy_utility(table, region_index, channel_index, action_mask)
    actions = action_mask.sum(dim=1).to(utility.dtype)
    if normalized_latency is None:
        normalized_latency = torch.zeros_like(utility)
    if normalized_latency.shape != utility.shape:
        raise ValueError("normalized latency must be one value per episode")
    expected_return = (
        utility
        - contract.action_cost * actions
        - contract.latency_cost * normalized_latency
    )
    oracle_region, oracle_channel, oracle_mask = label_oracle_policy(table, budget=2)
    oracle_utility = gather_policy_utility(
        table, oracle_region, oracle_channel, oracle_mask
    )
    oracle_actions = oracle_mask.sum(dim=1).to(utility.dtype)
    oracle_return = (
        oracle_utility
        - contract.action_cost * oracle_actions
        - contract.latency_cost * normalized_latency
    )
    regret = oracle_return - expected_return
    if random_return is None:
        normalized_regret = torch.full_like(regret, torch.nan)
    else:
        if random_return.shape != regret.shape:
            raise ValueError("random return must be one value per episode")
        denominator = (oracle_return - random_return).abs().clamp_min(1e-8)
        normalized_regret = regret / denominator
    best_available = torch.maximum(
        table.region_utility.max(dim=1).values,
        table.path_utility.reshape(table.path_utility.shape[0], -1).max(dim=1).values,
    )
    stopped = actions == 0
    false_stop = stopped & (best_available > contract.useful_gain_delta)
    unnecessary_continue = (~stopped) & (best_available <= 0)
    return PolicyEpisodeMetrics(
        utility=utility,
        expected_return=expected_return,
        actions=actions,
        latency_normalized=normalized_latency,
        regret=regret,
        oracle_normalized_regret=normalized_regret,
        false_stop=false_stop,
        unnecessary_continue=unnecessary_continue,
        action_budget_violation=actions > contract.max_actions,
        latency_budget_violation=normalized_latency > latency_budget,
    )


def lower_tail_cvar(values: np.ndarray, fraction: float = 0.10) -> float:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if not len(data) or not 0 < fraction <= 1:
        raise ValueError("CVaR needs finite values and a fraction in (0,1]")
    count = max(1, int(np.ceil(fraction * len(data))))
    return float(np.sort(data)[:count].mean())


def interquartile_mean(values: np.ndarray) -> float:
    data = np.sort(np.asarray(values, dtype=np.float64))
    data = data[np.isfinite(data)]
    if not len(data):
        raise ValueError("IQM needs at least one finite value")
    lower, upper = np.quantile(data, [0.25, 0.75])
    retained = data[(data >= lower) & (data <= upper)]
    return float(retained.mean())


def performance_profile(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    grid = np.asarray(thresholds, dtype=np.float64)
    data = data[np.isfinite(data)]
    if not len(data) or grid.ndim != 1:
        raise ValueError("performance profile needs finite values and a 1-D grid")
    return np.asarray([(data > threshold).mean() for threshold in grid])


def gain_calibration(
    predicted: np.ndarray,
    realized: np.ndarray,
    *,
    bins: int = 10,
) -> list[dict[str, float | int]]:
    prediction = np.asarray(predicted, dtype=np.float64)
    outcome = np.asarray(realized, dtype=np.float64)
    finite = np.isfinite(prediction) & np.isfinite(outcome)
    prediction, outcome = prediction[finite], outcome[finite]
    if not len(prediction) or bins < 2:
        raise ValueError("gain calibration needs observations and at least two bins")
    edges = np.unique(np.quantile(prediction, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        edges = np.array([prediction.min() - 1e-9, prediction.max() + 1e-9])
    assignment = np.clip(np.digitize(prediction, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for index in range(len(edges) - 1):
        selected = assignment == index
        if not np.any(selected):
            continue
        rows.append({
            "bin": index,
            "bin_low": float(edges[index]),
            "bin_high": float(edges[index + 1]),
            "n": int(selected.sum()),
            "predicted_mean": float(prediction[selected].mean()),
            "realized_mean": float(outcome[selected].mean()),
            "absolute_calibration_error": float(abs(
                prediction[selected].mean() - outcome[selected].mean()
            )),
        })
    return rows
