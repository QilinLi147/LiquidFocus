"""Exact current-train anchor fitting for V185."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler

from .v163_stable_evidence import exact_stable_evidence


ANCHOR_CONTRACT = {
    "seed": {"estimator": "logistic", "C": 0.2, "session3_weight": 20.0},
    "seediv": {"estimator": "logistic", "C": 1.0, "session3_weight": 40.0},
    "seedv": {"estimator": "logistic", "C": 0.5, "session3_weight": 40.0},
    "mped": {"estimator": "ridge", "alpha": 1000.0, "session3_weight": 1.0},
}


def fit_exact_anchor_state(
    model: Any, *, stable_values: np.ndarray, labels: np.ndarray,
    sessions: np.ndarray, fit_indices: np.ndarray, dataset: str,
    fit_partition: str, **_unused: Any,
) -> dict[str, Any]:
    if fit_partition not in {"development_archive_train_only", "train+development_union"}:
        raise PermissionError("V185 anchor fit partition changed")
    index = np.asarray(fit_indices, dtype=np.int64)
    if index.ndim != 1 or not len(index):
        raise ValueError("V185 exact anchor needs a non-empty train partition")
    configuration = dict(ANCHOR_CONTRACT[dataset])
    evidence = exact_stable_evidence(np.asarray(stable_values, dtype=np.float32)[index])
    if dataset == "mped":
        raw_feature = evidence.channel.astype(np.float64, copy=False)
        sample_weight = np.ones(len(index), dtype=np.float64)
    else:
        raw_feature = evidence.combined.astype(np.float64, copy=False)
        session = np.asarray(sessions, dtype=np.int64)[index]
        sample_weight = np.where(
            session == 2, float(configuration["session3_weight"]), 1.0
        ).astype(np.float64)
    scaler = StandardScaler().fit(raw_feature, sample_weight=sample_weight)
    feature = scaler.transform(raw_feature)
    label = np.asarray(labels, dtype=np.int64)[index]
    if dataset == "mped":
        estimator = RidgeClassifier(
            alpha=float(configuration["alpha"]), class_weight="balanced"
        ).fit(feature, label)
        raw_coefficient = np.asarray(estimator.coef_, dtype=np.float64)
        intercept = np.asarray(estimator.intercept_, dtype=np.float64)
        if model.classes == 2 and raw_coefficient.shape[0] == 1:
            raw_coefficient = np.concatenate((-0.5 * raw_coefficient, 0.5 * raw_coefficient))
            intercept = np.concatenate((-0.5 * intercept, 0.5 * intercept))
        coefficient = np.zeros((model.classes, 935), dtype=np.float64)
        coefficient[:, :620] = raw_coefficient
        feature_mean = np.zeros(935, dtype=np.float64); feature_mean[:620] = scaler.mean_
        feature_scale = np.ones(935, dtype=np.float64); feature_scale[:620] = scaler.scale_
        feature_scope = "channel_mean_std_620"
    else:
        estimator = LogisticRegression(
            C=float(configuration["C"]), max_iter=400,
            class_weight="balanced", random_state=8635,
        ).fit(feature, label, sample_weight=sample_weight)
        coefficient = np.asarray(estimator.coef_, dtype=np.float64)
        intercept = np.asarray(estimator.intercept_, dtype=np.float64)
        if model.classes == 2 and coefficient.shape[0] == 1:
            coefficient = np.concatenate((-0.5 * coefficient, 0.5 * coefficient))
            intercept = np.concatenate((-0.5 * intercept, 0.5 * intercept))
        feature_mean = np.asarray(scaler.mean_, dtype=np.float64)
        feature_scale = np.asarray(scaler.scale_, dtype=np.float64)
        feature_scope = "combined_935"
    if coefficient.shape != (model.classes, 935) or intercept.shape != (model.classes,):
        raise ValueError("V185 fitted anchor dimension mismatch")
    model.encoder.set_stable_feature_transform(feature_mean, feature_scale)
    model.encoder.set_anchor_state(coefficient, intercept)
    contract = {
        "schema": "liquidfocus_v185_exact_train_anchor_v1",
        "dataset": dataset, "fit_partition": fit_partition,
        "fit_samples": int(len(index)),
        "labels_used": True, "outer_used": False,
        "estimator": configuration["estimator"], "feature_scope": feature_scope,
        "configuration": configuration, "class_weight": "balanced",
        "scaler": "weighted StandardScaler" if dataset != "mped" else "unweighted StandardScaler",
        "coefficient_persisted_only_as_encoder_liquid_state_projection": True,
        "parallel_classifier": False, "raw_feature_readout": False,
    }
    return contract
