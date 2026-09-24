"""Leakage-safe participant-level training for LiquidFocus."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .analysis.policy_environment import enumerate_dense_paths, gather_policy_utility
from .compact import CompactConfig, compact_objective
from .data import DATASET_SPECS, FeatureArchive, Fold
from .v185_anchor_fit import fit_exact_anchor_state
from .v186_evidence_decoupled import (
    EvidenceDecoupledLiquidFocus,
    build_v186_model,
    fit_union_global_residual_bias,
)


DATASET_PROFILES: dict[str, dict[str, Any]] = {
    "seed": {
        "input_transform": "identity", "normalisation": "channel_band",
        "std_floor": 1e-6, "sampler": "session3_weighted",
        "session3_weight": 20.0, "class_balance": "none",
        "augmentation": "none", "label_smoothing": 0.04,
    },
    "seediv": {
        "input_transform": "identity", "normalisation": "channel_band",
        "std_floor": 1e-6, "sampler": "session3_weighted",
        "session3_weight": 40.0, "class_balance": "sqrt_inverse",
        "augmentation": "none", "label_smoothing": 0.04,
    },
    "seedv": {
        "input_transform": "identity", "normalisation": "channel_band",
        "std_floor": 1e-6, "sampler": "session3_weighted",
        "session3_weight": 40.0, "class_balance": "none",
        "augmentation": "none", "label_smoothing": 0.04,
    },
    "mped": {
        "input_transform": "log1p", "normalisation": "band_global",
        "std_floor": 0.15, "sampler": "shuffle",
        "session3_weight": 1.0, "class_balance": "sqrt_inverse",
        "augmentation": "channel_noise_mask", "label_smoothing": 0.04,
    },
}


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    minimum_epochs: int = 8
    patience: int = 6
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 8635
    gradient_clip: float = 5.0


class ParticipantDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        augmentation: str,
    ) -> None:
        self.values = values
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[Tensor, Tensor]:
        index = int(self.indices[item])
        value = self.values[index].copy()
        if self.augmentation == "channel_noise_mask":
            if np.random.random() < 0.65:
                missing = np.random.random(value.shape[0]) < np.random.uniform(0.03, 0.12)
                value[missing] = 0.0
            if np.random.random() < 0.40:
                value += np.random.normal(0.0, 0.035, size=value.shape).astype(np.float32)
            if np.random.random() < 0.30:
                value *= np.random.uniform(
                    0.92, 1.08, size=(1, value.shape[1], value.shape[2])
                ).astype(np.float32)
        elif self.augmentation != "none":
            raise ValueError(f"unknown augmentation: {self.augmentation}")
        return torch.from_numpy(value), torch.tensor(
            int(self.labels[index]), dtype=torch.long
        )


def set_random_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def model_config() -> CompactConfig:
    """Return the architecture and objective settings reported in the paper."""

    return CompactConfig(
        width=64,
        dropout=0.10,
        temporal_cell="liquid",
        use_delta_t=True,
        enable_router=True,
        enable_refiner=True,
        enable_safe_stop=True,
        stop_threshold=0.0,
        max_residual=0.10,
        policy_weight=0.50,
        candidate_weight=0.25,
        safety_weight=1.00,
        budget_weight=0.01,
        spatial_stem="stable_evidence_liquid_observation",
    ).validate()


def transform_features(values: np.ndarray, mode: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if mode == "identity":
        return result
    if mode == "log1p":
        if np.any(result < 0):
            raise ValueError("the log1p profile requires non-negative features")
        return np.log1p(result).astype(np.float32, copy=False)
    raise ValueError(f"unknown input transform: {mode}")


def fit_normalisation(
    values: np.ndarray, indices: np.ndarray, mode: str, std_floor: float
) -> tuple[np.ndarray, np.ndarray]:
    selected = values[np.asarray(indices, dtype=np.int64)]
    axes = (0, 2) if mode == "channel_band" else (0, 1, 2)
    if mode not in {"channel_band", "band_global"}:
        raise ValueError(f"unknown normalisation: {mode}")
    mean = selected.mean(axis=axes, keepdims=True, dtype=np.float64).astype(np.float32)
    scale = selected.std(axis=axes, keepdims=True, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(scale, float(std_floor)).astype(np.float32, copy=False)


def class_weight(labels: np.ndarray, indices: np.ndarray, classes: int, mode: str) -> np.ndarray:
    counts = np.bincount(labels[np.asarray(indices, dtype=np.int64)], minlength=classes)
    if np.any(counts == 0):
        raise ValueError("the training partition is missing a class")
    if mode == "none":
        return np.ones(classes, dtype=np.float32)
    if mode == "sqrt_inverse":
        weight = counts.astype(np.float64) ** -0.5
        return (weight / weight.mean()).astype(np.float32)
    raise ValueError(f"unknown class balance: {mode}")


def make_loader(
    normalized: np.ndarray,
    archive: FeatureArchive,
    indices: np.ndarray,
    profile: dict[str, Any],
    settings: TrainingConfig,
    *,
    training: bool,
) -> DataLoader:
    dataset = ParticipantDataset(
        normalized,
        archive.y,
        indices,
        str(profile["augmentation"]) if training else "none",
    )
    if not training:
        return DataLoader(dataset, batch_size=settings.batch_size, shuffle=False, num_workers=0)
    generator = torch.Generator().manual_seed(int(settings.seed))
    if profile["sampler"] == "session3_weighted":
        selected_session = archive.session[np.asarray(indices, dtype=np.int64)]
        weight = np.where(
            selected_session == 2, float(profile["session3_weight"]), 1.0
        ).astype(np.float64)
        sampler = WeightedRandomSampler(
            torch.from_numpy(weight), len(indices), replacement=True, generator=generator
        )
        return DataLoader(dataset, batch_size=settings.batch_size, sampler=sampler, num_workers=0)
    return DataLoader(
        dataset, batch_size=settings.batch_size, shuffle=True,
        generator=generator, num_workers=0
    )


@torch.no_grad()
def evaluate_model(
    model: EvidenceDecoupledLiquidFocus,
    loader: DataLoader,
    device: torch.device,
    *,
    include_confusion: bool = False,
) -> dict[str, Any]:
    model.eval()
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for x, y in loader:
        output = model(x.to(device), dense_teacher=False)
        labels.append(y.numpy())
        predictions.append(output.logits.argmax(dim=1).cpu().numpy())
    truth = np.concatenate(labels)
    predicted = np.concatenate(predictions)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro")),
    }
    if include_confusion:
        metrics["confusion_matrix"] = confusion_matrix(
            truth, predicted, labels=np.arange(model.classes)
        ).tolist()
    return metrics


def _train_epochs(
    model: EvidenceDecoupledLiquidFocus,
    loader: DataLoader,
    epochs: int,
    settings: TrainingConfig,
    profile: dict[str, Any],
    weights: Tensor | None,
    device: torch.device,
    validation_loader: DataLoader | None = None,
) -> tuple[dict[str, Tensor], int, list[dict[str, float]]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    best_state: dict[str, Tensor] | None = None
    best_epoch = epochs
    best_score = -float("inf")
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        batches = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(x, dense_teacher=True)
            terms = compact_objective(
                output,
                y,
                model.config,
                class_weight=weights,
                label_smoothing=float(profile["label_smoothing"]),
            )
            if not torch.isfinite(terms.loss):
                raise FloatingPointError("training produced a non-finite loss")
            terms.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip)
            optimizer.step()
            total += float(terms.loss.detach())
            batches += 1
        scheduler.step()
        row = {"epoch": float(epoch), "loss": total / max(1, batches)}
        if validation_loader is not None:
            validation = evaluate_model(model, validation_loader, device)
            score = validation["balanced_accuracy"]
            row.update({f"validation_{name}": value for name, value in validation.items()})
            if score > best_score + 1e-9:
                best_score = score
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
        history.append(row)
        if (
            validation_loader is not None
            and epoch >= settings.minimum_epochs
            and stale >= settings.patience
        ):
            break
    if best_state is None:
        best_state = {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        }
    return best_state, best_epoch, history


@torch.no_grad()
def calibrate_policy(
    model: EvidenceDecoupledLiquidFocus,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Select a STOP threshold on development data using NLL gain minus action cost."""

    model.eval()
    batches = []
    candidate_scores = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        output = model(x, dense_teacher=True)
        table = enumerate_dense_paths(
            output.primary_logits,
            output.dense_region_logits,
            output.dense_channel_logits,
            y,
            model.encoder.region_membership,
        )
        batches.append((output, table))
        candidate_scores.append(output.region_gain[torch.isfinite(output.region_gain)].detach().cpu())
        candidate_scores.append(output.channel_gain[torch.isfinite(output.channel_gain)].detach().cpu())
    scores = torch.cat(candidate_scores).numpy()
    all_stop = float(scores.max() + max(1.0, abs(float(scores.max()))) * 0.01)
    thresholds = np.unique(np.concatenate((
        np.array([0.0, all_stop]), np.quantile(scores, np.linspace(0.05, 0.95, 19))
    )))
    best_threshold = 0.0
    best_return = -float("inf")
    for threshold in thresholds:
        returns = []
        for output, table in batches:
            batch = output.region_gain.shape[0]
            rows = torch.arange(batch, device=device)
            region = output.region_gain.argmax(dim=1)
            region_score = output.region_gain[rows, region]
            region_active = region_score > float(threshold)
            legal = model.encoder.channel_region[None] == region[:, None]
            routed = output.channel_gain.masked_fill(~legal, -torch.inf)
            channel = routed.argmax(dim=1)
            channel_score = routed[rows, channel]
            channel_active = region_active & (channel_score > float(threshold))
            action_mask = torch.stack((region_active, channel_active), dim=1)
            utility = gather_policy_utility(table, region, channel, action_mask)
            returns.append(utility - 0.01 * action_mask.sum(dim=1).to(utility.dtype))
        mean_return = float(torch.cat(returns).mean())
        if mean_return > best_return + 1e-12:
            best_return = mean_return
            best_threshold = float(threshold)
    model.set_stop_threshold(best_threshold)
    return best_threshold


@torch.no_grad()
def calibrate_residual_scale(
    model: EvidenceDecoupledLiquidFocus,
    loader: DataLoader,
    device: torch.device,
) -> float:
    candidates = (0.0, 0.25, 0.50, 1.0)
    scored = []
    for value in candidates:
        model.encoder.primary_readout.set_residual_scale(value)
        scored.append((evaluate_model(model, loader, device)["balanced_accuracy"], value))
    selected = max(scored, key=lambda item: (item[0], -item[1]))[1]
    model.encoder.primary_readout.set_residual_scale(selected)
    return float(selected)


def train_fold(
    archive: FeatureArchive,
    dataset: str,
    fold: Fold,
    output: Path,
    settings: TrainingConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Select on development data, refit on the union, and test once."""

    profile = DATASET_PROFILES[dataset]
    set_random_seed(settings.seed + fold.subject * 100 + fold.outer_fold)
    transformed = transform_features(archive.x, str(profile["input_transform"]))

    mean, scale = fit_normalisation(
        transformed, fold.train, str(profile["normalisation"]), float(profile["std_floor"])
    )
    normalized = ((transformed - mean) / scale).astype(np.float32, copy=False)
    selection_model = build_v186_model(classes=DATASET_SPECS[dataset]["classes"], config=model_config())
    selection_model.encoder.set_observation_input_transform(mean, scale)
    fit_exact_anchor_state(
        selection_model,
        stable_values=transformed,
        labels=archive.y,
        sessions=archive.session,
        fit_indices=fold.train,
        dataset=dataset,
        fit_partition="development_archive_train_only",
    )
    selection_model = selection_model.to(device)
    train_loader = make_loader(normalized, archive, fold.train, profile, settings, training=True)
    validation_loader = make_loader(
        normalized, archive, fold.validation, profile, settings, training=False
    )
    weight_array = class_weight(
        archive.y, fold.train, DATASET_SPECS[dataset]["classes"], str(profile["class_balance"])
    )
    weights = None if profile["class_balance"] == "none" else torch.as_tensor(
        weight_array, dtype=torch.float32, device=device
    )
    best_state, selected_epoch, history = _train_epochs(
        selection_model, train_loader, settings.epochs, settings,
        profile, weights, device, validation_loader
    )
    selection_model.load_state_dict(best_state)
    residual_scale = calibrate_residual_scale(selection_model, validation_loader, device)
    stop_threshold = calibrate_policy(selection_model, validation_loader, device)

    union = np.unique(np.concatenate((fold.train, fold.validation)))
    set_random_seed(settings.seed + fold.subject * 100 + fold.outer_fold)
    union_mean, union_scale = fit_normalisation(
        transformed, union, str(profile["normalisation"]), float(profile["std_floor"])
    )
    union_normalized = ((transformed - union_mean) / union_scale).astype(np.float32, copy=False)
    refit_model = build_v186_model(classes=DATASET_SPECS[dataset]["classes"], config=model_config())
    refit_model.encoder.set_observation_input_transform(union_mean, union_scale)
    fit_exact_anchor_state(
        refit_model,
        stable_values=transformed,
        labels=archive.y,
        sessions=archive.session,
        fit_indices=union,
        dataset=dataset,
        fit_partition="train+development_union",
    )
    refit_model = refit_model.to(device)
    union_loader = make_loader(union_normalized, archive, union, profile, settings, training=True)
    union_weight_array = class_weight(
        archive.y, union, DATASET_SPECS[dataset]["classes"], str(profile["class_balance"])
    )
    union_weights = None if profile["class_balance"] == "none" else torch.as_tensor(
        union_weight_array, dtype=torch.float32, device=device
    )
    final_state, _, refit_history = _train_epochs(
        refit_model, union_loader, selected_epoch, settings,
        profile, union_weights, device, validation_loader=None
    )
    refit_model.load_state_dict(final_state)
    refit_model.encoder.primary_readout.set_residual_scale(residual_scale)
    refit_model.set_stop_threshold(stop_threshold)
    fit_union_global_residual_bias(
        refit_model,
        union_normalized[union],
        device=device,
        batch_size=settings.batch_size,
    )
    test_loader = make_loader(
        union_normalized, archive, fold.test, profile, settings, training=False
    )
    test_metrics = evaluate_model(refit_model, test_loader, device, include_confusion=True)

    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "model.pt"
    checkpoint = {
        "model_state": {
            name: value.detach().cpu() for name, value in refit_model.state_dict().items()
        },
        "model_config": asdict(refit_model.config),
        "training_config": asdict(settings),
        "dataset_profile": dict(profile),
        "dataset": dataset,
        "classes": DATASET_SPECS[dataset]["classes"],
        "subject": int(fold.subject),
        "fold": int(fold.outer_fold),
        "selected_epoch": int(selected_epoch),
        "residual_scale": residual_scale,
        "stop_threshold": stop_threshold,
        "input_mean": union_mean,
        "input_scale": union_scale,
        "test_metrics": test_metrics,
    }
    torch.save(checkpoint, checkpoint_path)
    report = {
        "dataset": dataset,
        "subject": int(fold.subject),
        "fold": int(fold.outer_fold),
        "selected_epoch": int(selected_epoch),
        "residual_scale": residual_scale,
        "stop_threshold": stop_threshold,
        "test_metrics": test_metrics,
        "selection_history": history,
        "refit_history": refit_history,
        "checkpoint": checkpoint_path.name,
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, float]:
    """Compute metrics after pooling disjoint test folds of one participant."""
    counts = np.asarray(matrix, dtype=np.float64)
    if (counts.ndim != 2 or counts.shape[0] != counts.shape[1]
            or not np.isfinite(counts).all() or np.any(counts < 0)
            or np.any(counts != np.floor(counts)) or counts.sum() <= 0):
        raise ValueError("confusion matrix must contain non-negative integer counts")
    correct = np.diag(counts)
    support = counts.sum(axis=1)
    predicted = counts.sum(axis=0)
    present = support > 0
    active = support + predicted > 0
    return {
        "accuracy": float(correct.sum() / counts.sum()),
        "balanced_accuracy": float((correct[present] / support[present]).mean()),
        "macro_f1": float((2 * correct[active] / (support + predicted)[active]).mean()),
    }


def aggregate_reports(reports: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Merge each participant's folds before participant-equal aggregation."""
    rows = list(reports)
    if not rows:
        raise ValueError("at least one completed report is required")
    if len({row["dataset"] for row in rows}) != 1:
        raise ValueError("summarise each dataset separately")
    groups: dict[int, list[dict[str, Any]]] = {}
    seen = set()
    for row in rows:
        identity = (int(row["subject"]), int(row["fold"]))
        if identity in seen:
            raise ValueError("duplicate participant/fold report")
        seen.add(identity)
        groups.setdefault(identity[0], []).append(row)
    participant_metrics = []
    for subject, folds in sorted(groups.items()):
        matrices = [row["test_metrics"].get("confusion_matrix") for row in folds]
        if all(matrix is not None for matrix in matrices):
            metrics = metrics_from_confusion(np.stack(matrices).sum(axis=0))
        elif len(folds) == 1:
            metrics = {key: float(folds[0]["test_metrics"][key])
                       for key in ("accuracy", "balanced_accuracy", "macro_f1")}
        else:
            raise ValueError("multi-fold aggregation requires saved confusion matrices")
        participant_metrics.append({"subject": subject, "folds": len(folds), **metrics})
    result: dict[str, Any] = {
        "dataset": rows[0]["dataset"],
        "runs": len(rows),
        "participants": len(participant_metrics),
        "aggregation": "participant-equal after pooling test-fold predictions",
        "participant_metrics": participant_metrics,
    }
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        values = np.asarray([row[metric] for row in participant_metrics], dtype=np.float64)
        result[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return result
