#!/usr/bin/env python3
"""Evaluate one released checkpoint on its participant-level test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from liquidfocus.compact import CompactConfig
from liquidfocus.data import FeatureArchive, protocol_folds
from liquidfocus.training import (
    DATASET_PROFILES, TrainingConfig, evaluate_model, make_loader, transform_features,
)
from liquidfocus.v186_evidence_decoupled import build_v186_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args()

    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    dataset = str(checkpoint["dataset"])
    subject = int(checkpoint["subject"])
    fold_index = int(checkpoint["fold"])
    archive = FeatureArchive.load(arguments.archive)
    folds = protocol_folds(archive, dataset, [subject])
    fold = next(value for value in folds if value.outer_fold == fold_index)
    transformed = transform_features(
        archive.x, str(checkpoint["dataset_profile"]["input_transform"])
    )
    normalized = (
        (transformed - np.asarray(checkpoint["input_mean"]))
        / np.asarray(checkpoint["input_scale"])
    ).astype(np.float32, copy=False)
    model = build_v186_model(
        classes=int(checkpoint["classes"]),
        config=CompactConfig(**checkpoint["model_config"]).validate(),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device(arguments.device)
    model = model.to(device)
    settings = TrainingConfig(**checkpoint["training_config"])
    loader = make_loader(
        normalized, archive, fold.test, DATASET_PROFILES[dataset], settings, training=False
    )
    metrics = evaluate_model(model, loader, device)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
