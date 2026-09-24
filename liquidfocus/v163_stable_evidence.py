"""Causal stable-evidence features and development-only diagnostics.

Every public fitting entry point accepts arrays already loaded through
:class:`DevelopmentArchive`; test partitions are outside this module's API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch

from .data import DevelopmentArchive
from .geometry import SEED_CHANNELS


FEATURE_KINDS = ("channel_only", "relation_only", "combined_anchor_only")

# SEED-family anchor parameters used by the reported participant protocol.
# MPED uses the ridge anchor defined in ``v185_anchor_fit.py``.
LOCKED_SEED_FAMILY_PROFILES: dict[str, dict[str, Any]] = {
    "seed": {"C": 0.2, "session3_weight": 20.0, "causal_decay": 0.95},
    "seediv": {"C": 1.0, "session3_weight": 40.0, "causal_decay": 0.90},
    "seedv": {"C": 0.5, "session3_weight": 40.0, "causal_decay": 0.99},
}


@dataclass(frozen=True)
class StableEvidence:
    """The two disjoint stable-evidence blocks and their exact concatenation."""

    channel: np.ndarray
    relation: np.ndarray

    @property
    def combined(self) -> np.ndarray:
        return np.concatenate((self.channel, self.relation), axis=1)

    def view(self, kind: str) -> np.ndarray:
        if kind == "channel_only":
            return self.channel
        if kind == "relation_only":
            return self.relation
        if kind == "combined_anchor_only":
            return self.combined
        raise ValueError(f"unknown stable-evidence view: {kind}")


def anatomical_membership_exact(
    channel_names: tuple[str, ...] = SEED_CHANNELS,
) -> np.ndarray:
    """Return the seven-region mapping used in the reported experiments."""

    membership = np.zeros((7, len(channel_names)), dtype=np.float32)
    for channel, name in enumerate(channel_names):
        if name.endswith("z"):
            side = "mid"
        else:
            digits = "".join(character for character in name if character.isdigit())
            side = "mid" if not digits else ("left" if int(digits) % 2 else "right")
        if name.startswith(("Fp", "AF")) or (
            name.startswith("F") and not name.startswith(("FC", "FT"))
        ):
            if side == "left":
                membership[0, channel] = 1.0
            elif side == "right":
                membership[1, channel] = 1.0
            else:
                membership[0:2, channel] = 0.5
        elif name.startswith(("FT", "T", "TP")):
            membership[2 if side == "left" else 3, channel] = 1.0
        elif name.startswith(("FC", "C", "CP")):
            membership[4, channel] = 1.0
        elif name.startswith(("P", "PO")):
            membership[5, channel] = 1.0
        elif name.startswith(("O", "CB")):
            membership[6, channel] = 1.0
        else:
            raise ValueError(f"unmapped channel {name}")
    np.testing.assert_allclose(membership.sum(axis=0), 1.0, rtol=0.0, atol=0.0)
    return membership


def _correlation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Population correlation with the fixed numerical stabilizer."""

    left = np.asarray(first, dtype=np.float32)
    right = np.asarray(second, dtype=np.float32)
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    numerator = (left * right).mean(axis=1)
    denominator = np.sqrt(
        ((left * left).mean(axis=1) + 1e-4)
        * ((right * right).mean(axis=1) + 1e-4)
    )
    return np.clip(numerator / denominator, -1.0, 1.0)


def exact_stable_evidence(x: np.ndarray) -> StableEvidence:
    """Extract channel and anatomical-relation evidence.

    ``x`` is the unnormalised, value-preserving development archive view with
    shape ``[N,62,10,5]``.  Pair order is lexicographic; each pair contributes
    zero, forward, then backward lag, each in five-band order.  No Fisher-z
    transform is applied.
    """

    values = np.asarray(x)
    if values.ndim != 4 or values.shape[1:] != (62, 10, 5):
        raise ValueError("stable evidence input must be [N,62,10,5]")
    values = values.astype(np.float32, copy=False)
    channel = np.concatenate(
        (values.mean(axis=2), values.std(axis=2, ddof=0)), axis=-1
    ).reshape(len(values), -1)

    membership = anatomical_membership_exact()
    membership = membership / membership.sum(axis=1, keepdims=True)
    region = np.einsum("rc,nctf->nrtf", membership, values, optimize=True)
    zero: list[np.ndarray] = []
    forward: list[np.ndarray] = []
    backward: list[np.ndarray] = []
    for left in range(7):
        for right in range(left + 1, 7):
            zero.append(_correlation(region[:, left], region[:, right]))
            forward.append(_correlation(region[:, left, :-1], region[:, right, 1:]))
            backward.append(_correlation(region[:, right, :-1], region[:, left, 1:]))
    # The feature vector uses direction-major layout:
    # all 21 zero-lag pairs, all 21 forward pairs, then all 21 backward pairs.
    relation = np.concatenate((*zero, *forward, *backward), axis=1)
    if channel.shape[1] != 620 or relation.shape[1] != 315:
        raise AssertionError("stable-evidence dimension contract changed")
    if not np.isfinite(channel).all() or not np.isfinite(relation).all():
        raise FloatingPointError("non-finite exact stable evidence")
    return StableEvidence(channel=channel, relation=relation)


@torch.no_grad()
def exact_stable_evidence_torch(
    normalized: np.ndarray,
    input_mean: np.ndarray,
    input_scale: np.ndarray,
    device: str | torch.device = "cpu",
    batch_size: int = 128,
) -> StableEvidence:
    """Execute the feature formula with Torch float32 operations."""

    values = np.asarray(normalized, dtype=np.float32)
    if values.ndim != 4 or values.shape[1:] != (62, 10, 5):
        raise ValueError("normalized stable evidence input must be [N,62,10,5]")
    if batch_size <= 0:
        raise ValueError("feature batch size must be positive")
    target = torch.device(device)
    mean = torch.as_tensor(input_mean, dtype=torch.float32, device=target)
    scale = torch.as_tensor(input_scale, dtype=torch.float32, device=target)
    membership = torch.as_tensor(
        anatomical_membership_exact(), dtype=torch.float32, device=target
    )
    membership = membership / membership.sum(dim=-1, keepdim=True).clamp_min(1.0)
    pair = torch.triu_indices(7, 7, offset=1, device=target)
    channel_blocks = []
    relation_blocks = []

    def correlation(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        first = first - first.mean(dim=1, keepdim=True)
        second = second - second.mean(dim=1, keepdim=True)
        numerator = (first * second).mean(dim=1)
        denominator = torch.sqrt(
            (first.square().mean(dim=1) + 1e-4)
            * (second.square().mean(dim=1) + 1e-4)
        )
        return (numerator / denominator).clamp(-1.0, 1.0)

    for start in range(0, len(values), batch_size):
        sequence = torch.as_tensor(
            values[start:start + batch_size], dtype=torch.float32, device=target
        )
        sequence = sequence * scale + mean
        channel = torch.cat((
            sequence.mean(dim=2), sequence.std(dim=2, unbiased=False)
        ), dim=-1).reshape(sequence.shape[0], -1)
        region = torch.einsum("rc,bctf->btrf", membership, sequence)
        left, right = pair[0], pair[1]
        zero = correlation(region[:, :, left], region[:, :, right])
        forward = correlation(region[:, :-1, left], region[:, 1:, right])
        backward = correlation(region[:, :-1, right], region[:, 1:, left])
        connection = torch.cat(
            (zero.flatten(1), forward.flatten(1), backward.flatten(1)), dim=1
        )
        channel_blocks.append(channel.cpu().numpy())
        relation_blocks.append(connection.cpu().numpy())
    return StableEvidence(
        channel=np.concatenate(channel_blocks).astype(np.float32, copy=False),
        relation=np.concatenate(relation_blocks).astype(np.float32, copy=False),
    )


def cumulative_stable_evidence(x: np.ndarray) -> tuple[StableEvidence, ...]:
    """Return causal prefix evidence for every observed time step."""

    values = np.asarray(x)
    if values.ndim != 4 or values.shape[1:] != (62, 10, 5):
        raise ValueError("cumulative observation input must be [N,62,10,5]")
    result: list[StableEvidence] = []
    for stop in range(1, values.shape[2] + 1):
        prefix = values[:, :, :stop]
        # Reuse the exact formulas with a variable temporal length.
        raw = prefix.astype(np.float32, copy=False)
        channel = np.concatenate(
            (raw.mean(axis=2), raw.std(axis=2, ddof=0)), axis=-1
        ).reshape(len(raw), -1)
        membership = anatomical_membership_exact()
        membership /= membership.sum(axis=1, keepdims=True)
        region = np.einsum("rc,nctf->nrtf", membership, raw, optimize=True)
        zero_blocks: list[np.ndarray] = []
        forward_blocks: list[np.ndarray] = []
        backward_blocks: list[np.ndarray] = []
        for left in range(7):
            for right in range(left + 1, 7):
                zero = _correlation(region[:, left], region[:, right])
                if stop == 1:
                    forward = np.zeros_like(zero)
                    backward = np.zeros_like(zero)
                else:
                    forward = _correlation(region[:, left, :-1], region[:, right, 1:])
                    backward = _correlation(region[:, right, :-1], region[:, left, 1:])
                zero_blocks.append(zero)
                forward_blocks.append(forward)
                backward_blocks.append(backward)
        result.append(StableEvidence(
            channel,
            np.concatenate((*zero_blocks, *forward_blocks, *backward_blocks), axis=1),
        ))
    return tuple(result)


def session_sample_weights(
    sessions: np.ndarray,
    indices: np.ndarray,
    session3_weight: float,
) -> np.ndarray:
    """Training-only session weights; labels never enter this function."""

    if session3_weight <= 0:
        raise ValueError("session3 weight must be positive")
    selected = np.asarray(sessions)[np.asarray(indices, dtype=np.int64)]
    return np.where(selected == 2, float(session3_weight), 1.0).astype(np.float64)


def fit_weighted_normalisation(
    x: np.ndarray,
    indices: np.ndarray,
    sessions: np.ndarray,
    mode: str,
    std_floor: float,
    session3_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit train-only weighted neural normalisation."""

    index = np.asarray(indices, dtype=np.int64)
    values = np.asarray(x, dtype=np.float64)[index]
    sample_weight = session_sample_weights(sessions, index, session3_weight)
    if mode == "channel_band":
        # Each sample weight is repeated for every time point.
        expanded = sample_weight[:, None, None, None]
        denominator = float(sample_weight.sum() * values.shape[2])
        mean = (values * expanded).sum(axis=(0, 2), keepdims=True) / denominator
        variance = (
            ((values - mean) ** 2) * expanded
        ).sum(axis=(0, 2), keepdims=True) / denominator
    elif mode == "band_global":
        expanded = sample_weight[:, None, None, None]
        denominator = float(
            sample_weight.sum() * values.shape[1] * values.shape[2]
        )
        mean = (values * expanded).sum(axis=(0, 1, 2), keepdims=True) / denominator
        variance = (
            ((values - mean) ** 2) * expanded
        ).sum(axis=(0, 1, 2), keepdims=True) / denominator
    else:
        raise ValueError(f"unknown normalisation mode: {mode}")
    scale = np.maximum(np.sqrt(variance), float(std_floor))
    return mean.astype(np.float32), scale.astype(np.float32)


def _validated_profile(
    dataset: str, overrides: Mapping[str, Mapping[str, Any]] | None
) -> dict[str, Any]:
    profiles: dict[str, Mapping[str, Any]] = dict(LOCKED_SEED_FAMILY_PROFILES)
    if overrides:
        profiles.update(overrides)
    if dataset not in profiles:
        raise ValueError(
            f"no verified exact stable-evidence profile for {dataset}; "
            "provide a source-audited override"
        )
    profile = dict(profiles[dataset])
    required = {"C", "session3_weight", "causal_decay"}
    missing = required - set(profile)
    if missing:
        raise ValueError(f"incomplete {dataset} profile: {sorted(missing)}")
    return profile


def fit_exact_logistic_view(
    archive: DevelopmentArchive,
    fit_evidence: StableEvidence,
    kind: str,
    seed: int,
    profile_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    inference_evidence: StableEvidence | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one diagnostic anchor view on selection-train only.

    The returned probability is for the archive development partition only.
    It is diagnostic output and no function in ``compact.py`` consumes it.
    """

    if archive.metadata.get("contains_outer_test_features") is not False:
        raise PermissionError("V163 exact decomposition refuses outer features")
    dataset = str(archive.metadata["dataset"])
    profile = _validated_profile(dataset, profile_overrides)
    source_feature = fit_evidence.view(kind)
    source_inference_feature = (
        source_feature
        if inference_evidence is None
        else inference_evidence.view(kind)
    )
    if source_inference_feature.shape != source_feature.shape:
        raise ValueError("fit/inference stable-evidence shape mismatch")
    # Feature extraction is Torch float32, then
    # the concatenated feature matrix is promoted to float64 before sklearn.
    feature = np.asarray(source_feature, dtype=np.float64)
    inference_feature = np.asarray(source_inference_feature, dtype=np.float64)
    train = archive.train_indices
    development = archive.validation_indices
    weight = session_sample_weights(
        archive.session, train, float(profile["session3_weight"])
    )
    scaler = StandardScaler().fit(feature[train], sample_weight=weight)
    classifier = LogisticRegression(
        C=float(profile["C"]),
        solver="lbfgs",
        max_iter=400,
        class_weight="balanced",
        random_state=int(seed),
    ).fit(
        scaler.transform(feature[train]), archive.y[train], sample_weight=weight
    )
    probability = classifier.predict_proba(
        scaler.transform(inference_feature[development])
    )
    contract = {
        "dataset": dataset,
        "feature_kind": kind,
        "feature_dimension": int(feature.shape[1]),
        "C": float(profile["C"]),
        "session3_weight": float(profile["session3_weight"]),
        "causal_decay": float(profile["causal_decay"]),
        "scaler": "StandardScaler(sample_weight=train_session_weight)",
        "estimator": "multinomial LogisticRegression",
        "solver": "lbfgs",
        "max_iter": 400,
        "class_weight": "balanced",
        "fit_partition": "selection-train-only",
        "fit_samples": int(len(train)),
        "development_samples": int(len(development)),
        "sample_weight_sum": float(weight.sum()),
        "scaler_n_samples_seen": float(scaler.n_samples_seen_),
        "sklearn_fit_dtype": str(feature.dtype),
        "n_iter": np.asarray(classifier.n_iter_).astype(int).tolist(),
    }
    return np.asarray(probability, dtype=np.float64), contract
