#!/usr/bin/env python3
"""Train participant-level LiquidFocus models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from liquidfocus.data import DATASET_SPECS, FeatureArchive, protocol_folds
from liquidfocus.training import TrainingConfig, aggregate_reports, train_fold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subject", type=int)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--minimum-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=8635)
    arguments = parser.parse_args()

    archive = FeatureArchive.load(arguments.archive)
    if archive.metadata.get("dataset") not in {None, arguments.dataset}:
        raise ValueError("archive dataset metadata does not match --dataset")
    subjects = (
        [arguments.subject]
        if arguments.subject is not None
        else sorted(int(value) for value in np.unique(archive.subject))
    )
    folds = protocol_folds(archive, arguments.dataset, subjects)
    if arguments.fold is not None:
        folds = [fold for fold in folds if fold.outer_fold == arguments.fold]
    if not folds:
        raise ValueError("no fold matches the requested subject/fold")
    settings = TrainingConfig(
        epochs=arguments.epochs,
        minimum_epochs=arguments.minimum_epochs,
        patience=arguments.patience,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        seed=arguments.seed,
    )
    device = torch.device(arguments.device)
    reports = []
    for fold in folds:
        target = arguments.output / f"subject_{fold.subject:02d}" / f"fold_{fold.outer_fold:02d}"
        reports.append(train_fold(archive, arguments.dataset, fold, target, settings, device))
    summary = aggregate_reports(reports)
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
