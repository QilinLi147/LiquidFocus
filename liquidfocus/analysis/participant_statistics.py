"""Participant-level aggregation and prespecified paired inference.

Windows, folds, perturbation repetitions and random-policy repetitions are
never treated as independent participants.  In particular, all four MPED folds
are combined before participant metrics or bootstrap sampling are computed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)


PRIMARY_METRIC = "balanced_accuracy"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260831


def multiclass_brier(labels: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    one_hot = np.eye(probability.shape[1], dtype=np.float64)[y]
    return float(np.mean(np.sum((probability - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 15,
) -> float:
    y = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    confidence = probability.max(axis=1)
    correct = probability.argmax(axis=1) == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            selected = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if np.any(selected):
            result += selected.mean() * abs(
                correct[selected].mean() - confidence[selected].mean()
            )
    return float(result if total else np.nan)


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or probability.ndim != 2 or len(y) != len(probability):
        raise ValueError("labels/probabilities must be [N] and [N,K]")
    if not len(y) or not np.isfinite(probability).all():
        raise ValueError("classification metrics require finite non-empty data")
    row_sum = probability.sum(axis=1)
    if np.any(probability < 0) or not np.allclose(row_sum, 1.0, atol=1e-6):
        raise ValueError("probability rows must be non-negative and sum to one")
    prediction = probability.argmax(axis=1)
    classes = np.arange(probability.shape[1])
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "nll": float(log_loss(y, probability, labels=classes)),
        "brier": multiclass_brier(y, probability),
        "ece": expected_calibration_error(y, probability),
        "n_samples": float(len(y)),
    }


def participant_metrics_from_long(
    frame: pd.DataFrame,
    *,
    probability_prefix: str = "prob_",
) -> pd.DataFrame:
    required = {"dataset", "participant_id", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prediction table missing columns: {sorted(missing)}")
    probability_columns = sorted(
        (name for name in frame.columns if name.startswith(probability_prefix)),
        key=lambda name: int(name[len(probability_prefix):]),
    )
    if not probability_columns:
        raise ValueError("prediction table has no probability columns")
    rows: list[dict[str, object]] = []
    # Grouping by participant intentionally absorbs session/fold rows; MPED's
    # four folds therefore become one n=participant unit before inference.
    for (dataset, participant), group in frame.groupby(
        ["dataset", "participant_id"], sort=True, dropna=False
    ):
        values = classification_metrics(
            group["label"].to_numpy(), group[probability_columns].to_numpy()
        )
        for metric, value in values.items():
            rows.append({
                "dataset": dataset,
                "participant_id": participant,
                "metric": metric,
                "value": value,
                "n_folds_merged": int(group["fold"].nunique()) if "fold" in group else 1,
            })
    result = pd.DataFrame(rows)
    mped = result[(result["dataset"] == "mped") & (result["metric"] == "balanced_accuracy")]
    if len(mped) and len(mped) != 23:
        raise ValueError(f"MPED participant aggregation expected n=23, observed n={len(mped)}")
    return result


def paired_bootstrap(
    differences: Sequence[float] | np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("paired bootstrap needs a finite 1-D participant vector")
    if replicates < 100:
        raise ValueError("paired bootstrap requires at least 100 replicates")
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, len(values), size=(replicates, len(values)))
    draws = values[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    # Bootstrap tail probability with a +1 finite-sample correction.
    lower_tail = (np.count_nonzero(draws <= 0) + 1) / (replicates + 1)
    upper_tail = (np.count_nonzero(draws >= 0) + 1) / (replicates + 1)
    p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))
    standard_deviation = values.std(ddof=1) if len(values) > 1 else np.nan
    effect = values.mean() / standard_deviation if standard_deviation > 0 else np.nan
    tolerance = 1e-12
    return {
        "n_participants": int(len(values)),
        "mean_difference": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": float(p_value),
        "paired_effect_dz": float(effect),
        "wins": int(np.count_nonzero(values > tolerance)),
        "ties": int(np.count_nonzero(np.abs(values) <= tolerance)),
        "losses": int(np.count_nonzero(values < -tolerance)),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
    }


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("Holm adjustment requires finite p values in [0,1]")
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        (len(values) - np.arange(len(values))) * values[order]
    )
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(values)
    adjusted[order] = adjusted_sorted
    return adjusted


def paired_family_statistics(
    frame: pd.DataFrame,
    *,
    family_columns: Sequence[str] = ("experiment_chapter", "metric"),
    comparison_columns: Sequence[str] = ("variant_a", "variant_b"),
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    required = set(family_columns) | set(comparison_columns) | {
        "dataset", "participant_id", "paired_difference"
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"paired table missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    grouping = list(family_columns) + ["dataset"] + list(comparison_columns)
    for key, group in frame.groupby(grouping, sort=True, dropna=False):
        if group["participant_id"].duplicated().any():
            raise ValueError("paired inference received duplicate participant rows")
        stats = paired_bootstrap(
            group["paired_difference"].to_numpy(),
            replicates=replicates,
            seed=seed,
        )
        rows.append({**dict(zip(grouping, key if isinstance(key, tuple) else (key,))), **stats})
    result = pd.DataFrame(rows)
    result["q_value_holm"] = np.nan
    for _, indices in result.groupby(list(family_columns), sort=True).groups.items():
        result.loc[indices, "q_value_holm"] = holm_adjust(
            result.loc[indices, "p_value"].to_numpy()
        )
    return result


def two_level_dataset_equal_bootstrap(
    frame: pd.DataFrame,
    *,
    value_column: str = "paired_difference",
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    required = {"dataset", "participant_id", value_column}
    if required - set(frame.columns):
        raise ValueError("two-level bootstrap table is incomplete")
    grouped = {
        dataset: group[value_column].to_numpy(dtype=np.float64)
        for dataset, group in frame.groupby("dataset", sort=True)
    }
    if not grouped or any(not len(values) for values in grouped.values()):
        raise ValueError("each dataset needs participant differences")
    datasets = sorted(grouped)
    generator = np.random.default_rng(int(seed))
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected_datasets = generator.choice(datasets, size=len(datasets), replace=True)
        dataset_means = []
        for dataset in selected_datasets:
            values = grouped[str(dataset)]
            sample = generator.choice(values, size=len(values), replace=True)
            dataset_means.append(sample.mean())
        draws[replicate] = np.mean(dataset_means)
    low, high = np.quantile(draws, [0.025, 0.975])
    point = float(np.mean([values.mean() for values in grouped.values()]))
    return {
        "n_datasets": len(datasets),
        "n_participants": int(sum(len(value) for value in grouped.values())),
        "dataset_equal_mean_difference": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
    }


def classify_effect(
    ci_low: float,
    ci_high: float,
    *,
    noninferiority_margin: float = -0.01,
) -> str:
    if ci_low > 0:
        return "strictly_superior"
    if ci_low >= noninferiority_margin:
        return "noninferior_or_close"
    if ci_low < noninferiority_margin < ci_high:
        return "uncertain"
    return "inferior_beyond_margin"
