"""Canonical feature loading and participant-level dataset protocols."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np


DATASET_SPECS = {
    "seed": {"classes": 3, "subjects": 15, "channels": 62},
    "seediv": {"classes": 4, "subjects": 15, "channels": 62},
    "seedv": {"classes": 5, "subjects": 16, "channels": 62},
    "mped": {"classes": 3, "subjects": 23, "channels": 62},
}


@dataclass(frozen=True)
class Fold:
    subject: int
    outer_fold: int
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


@dataclass
class FeatureArchive:
    x: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    session: np.ndarray
    trial: np.ndarray
    emotion: np.ndarray | None
    metadata: dict

    @classmethod
    def load(cls, path: Path | str) -> "FeatureArchive":
        with np.load(Path(path), allow_pickle=False) as value:
            required = {"x", "y", "subject", "session", "trial"}
            missing = required - set(value.files)
            if missing:
                raise ValueError(f"archive is missing arrays: {sorted(missing)}")
            x = canonical_features(value["x"])
            metadata = {}
            if "metadata" in value.files:
                raw_metadata = value["metadata"]
                metadata = json.loads(str(raw_metadata.item() if raw_metadata.ndim == 0 else raw_metadata))
            emotion = value["emotion"].astype(np.int64, copy=False) if "emotion" in value.files else None
            result = cls(
                x=x,
                y=value["y"].astype(np.int64, copy=False),
                subject=value["subject"].astype(np.int64, copy=False),
                session=value["session"].astype(np.int64, copy=False),
                trial=value["trial"].astype(np.int64, copy=False),
                emotion=emotion,
                metadata=metadata,
            )
        result.validate()
        return result

    def validate(self) -> None:
        arrays = [self.y, self.subject, self.session, self.trial]
        if self.emotion is not None:
            arrays.append(self.emotion)
        if any(len(value) != len(self.x) for value in arrays):
            raise ValueError("archive arrays have inconsistent lengths")
        labels = sorted(int(value) for value in np.unique(self.y))
        if labels != list(range(len(labels))):
            raise ValueError("class labels must be contiguous and zero based")
        if not np.isfinite(self.x).all():
            raise ValueError("features contain non-finite values")


class DevelopmentArchive:
    """Loader retained for the stable-evidence diagnostic API."""

    def __init__(self, path: Path | str):
        with np.load(Path(path), allow_pickle=False) as value:
            self.x = canonical_features(value["x"])
            self.y = value["y"].astype(np.int64, copy=False)
            self.partition = value["partition"].astype(np.int8, copy=False)
            self.session = value["session"].astype(np.int64, copy=False)
            self.trial = value["trial"].astype(np.int64, copy=False)
            raw_metadata = value["metadata"]
            self.metadata = json.loads(str(raw_metadata.item() if raw_metadata.ndim == 0 else raw_metadata))
        if not np.all(np.isin(self.partition, [0, 1])):
            raise ValueError("development partition must contain only train and validation rows")
        self.classes = len(np.unique(self.y))

    @property
    def train_indices(self) -> np.ndarray:
        return np.flatnonzero(self.partition == 0)

    @property
    def validation_indices(self) -> np.ndarray:
        return np.flatnonzero(self.partition == 1)


def canonical_features(values: np.ndarray) -> np.ndarray:
    """Return float32 features in [sample, channel, prefix, band] order."""
    x = np.asarray(values)
    if x.ndim == 3 and x.shape[1:] == (62, 50):
        x = x.reshape(-1, 62, 10, 5)
    if x.ndim != 4 or x.shape[1:] != (62, 10, 5):
        raise ValueError("expected feature shape [N,62,50] or [N,62,10,5]")
    return x.astype(np.float32, copy=False)


def save_feature_archive(source: Path | str, target: Path | str, dataset: str) -> None:
    """Validate and convert a provider-derived NPZ file to the canonical layout."""
    if dataset not in DATASET_SPECS:
        raise ValueError(f"unsupported dataset: {dataset}")
    archive = FeatureArchive.load(source)
    classes = len(np.unique(archive.y))
    if classes != DATASET_SPECS[dataset]["classes"]:
        raise ValueError(f"{dataset} requires {DATASET_SPECS[dataset]['classes']} classes")
    metadata = {
        "dataset": dataset,
        "classes": classes,
        "channels": 62,
        "prefixes": 10,
        "bands": 5,
        "feature_layout": "sample,channel,prefix,band",
    }
    payload = {
        "x": archive.x,
        "y": archive.y,
        "subject": archive.subject,
        "session": archive.session,
        "trial": archive.trial,
        "metadata": json.dumps(metadata, sort_keys=True),
    }
    if archive.emotion is not None:
        payload["emotion"] = archive.emotion
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target_path, **payload)


def _trial_label(data: FeatureArchive, subject: int, session: int, trial: int) -> int:
    members = (
        (data.subject == subject)
        & (data.session == session)
        & (data.trial == trial)
    )
    labels = np.unique(data.y[members])
    if len(labels) != 1:
        raise ValueError("a trial must have one emotion label")
    return int(labels[0])


def _seed_family_fold(data: FeatureArchive, subject: int) -> Fold:
    subject_mask = data.subject == subject
    classes = sorted(int(value) for value in np.unique(data.y[subject_mask]))
    labels_by_session = {}
    for session in range(3):
        trials = sorted(int(value) for value in np.unique(
            data.trial[subject_mask & (data.session == session)]
        ))
        labels_by_session[session] = {
            trial: _trial_label(data, subject, session, trial) for trial in trials
        }
    development_trials = {
        max(trial for trial, label in labels_by_session[1].items() if label == class_index)
        for class_index in classes
    }
    calibration_trials = {
        min(trial for trial, label in labels_by_session[2].items() if label == class_index)
        for class_index in classes
    }
    train = np.flatnonzero(subject_mask & (
        (data.session == 0)
        | ((data.session == 1) & ~np.isin(data.trial, sorted(development_trials)))
        | ((data.session == 2) & np.isin(data.trial, sorted(calibration_trials)))
    ))
    validation = np.flatnonzero(
        subject_mask & (data.session == 1) & np.isin(data.trial, sorted(development_trials))
    )
    test = np.flatnonzero(
        subject_mask & (data.session == 2) & ~np.isin(data.trial, sorted(calibration_trials))
    )
    return Fold(subject, 0, train, validation, test)


def _mped_folds(data: FeatureArchive, subject: int) -> list[Fold]:
    if data.emotion is None:
        raise ValueError("MPED requires the original seven-emotion trial labels")
    subject_mask = data.subject == subject
    trial_emotion = {}
    for trial in np.unique(data.trial[subject_mask]):
        labels = np.unique(data.emotion[subject_mask & (data.trial == trial)])
        if len(labels) != 1:
            raise ValueError("an MPED trial must have one source emotion label")
        trial_emotion[int(trial)] = int(labels[0])
    result = []
    for outer_fold in range(4):
        groups = {"train": [], "validation": [], "test": []}
        for emotion in range(7):
            trials = sorted(trial for trial, label in trial_emotion.items() if label == emotion)
            if len(trials) != 4:
                raise ValueError("MPED requires four trials per source emotion")
            groups["test"].append(trials[outer_fold])
            groups["validation"].append(trials[(outer_fold + 1) % 4])
            groups["train"].extend(
                trial for trial in trials
                if trial not in {trials[outer_fold], trials[(outer_fold + 1) % 4]}
            )
        indices = {
            name: np.flatnonzero(subject_mask & np.isin(data.trial, trials))
            for name, trials in groups.items()
        }
        result.append(Fold(subject, outer_fold, indices["train"], indices["validation"], indices["test"]))
    return result


def protocol_folds(data: FeatureArchive, dataset: str, subjects: Iterable[int]) -> list[Fold]:
    """Create the participant-adaptive folds reported in the manuscript."""
    result = []
    for subject in subjects:
        if dataset == "seed":
            subject_mask = data.subject == subject
            train = np.flatnonzero(subject_mask & (
                (data.session == 0)
                | ((data.session == 1) & (data.trial < 12))
                | ((data.session == 2) & (data.trial < 3))
            ))
            validation = np.flatnonzero(subject_mask & (data.session == 1) & (data.trial >= 12))
            test = np.flatnonzero(subject_mask & (data.session == 2) & (data.trial >= 3))
            result.append(Fold(subject, 0, train, validation, test))
        elif dataset in {"seediv", "seedv"}:
            result.append(_seed_family_fold(data, subject))
        elif dataset == "mped":
            result.extend(_mped_folds(data, subject))
        else:
            raise ValueError(f"unsupported dataset: {dataset}")
    for fold in result:
        validate_fold(fold)
    return result


def validate_fold(fold: Fold) -> None:
    partitions = [fold.train, fold.validation, fold.test]
    if any(len(values) == 0 for values in partitions):
        raise ValueError("fold contains an empty partition")
    combined = np.concatenate(partitions)
    if len(np.unique(combined)) != len(combined):
        raise ValueError("samples overlap across train, validation, and test")
