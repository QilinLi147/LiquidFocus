#!/usr/bin/env python3
"""Convert provider-derived DE features to the release archive format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from liquidfocus.data import DATASET_SPECS, FeatureArchive, canonical_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subject-offset", type=int, default=0)
    parser.add_argument("--session-offset", type=int, default=0)
    parser.add_argument("--trial-offset", type=int, default=0)
    arguments = parser.parse_args()

    with np.load(arguments.input, allow_pickle=False) as source:
        required = {"x", "y", "subject", "session", "trial"}
        missing = required - set(source.files)
        if missing:
            raise ValueError(f"input is missing arrays: {sorted(missing)}")
        x = canonical_features(source["x"])
        raw_y = np.asarray(source["y"], dtype=np.int64)
        classes = sorted(int(value) for value in np.unique(raw_y))
        label_map = {value: index for index, value in enumerate(classes)}
        y = np.asarray([label_map[int(value)] for value in raw_y], dtype=np.int64)
        subject = np.asarray(source["subject"], dtype=np.int64) - arguments.subject_offset
        session = np.asarray(source["session"], dtype=np.int64) - arguments.session_offset
        trial = np.asarray(source["trial"], dtype=np.int64) - arguments.trial_offset
        emotion = (
            np.asarray(source["emotion"], dtype=np.int64)
            if "emotion" in source.files else None
        )

    expected = DATASET_SPECS[arguments.dataset]
    if len(classes) != expected["classes"]:
        raise ValueError(
            f"{arguments.dataset} requires {expected['classes']} target classes; "
            f"found {len(classes)}"
        )
    metadata = {
        "dataset": arguments.dataset,
        "classes": expected["classes"],
        "channels": expected["channels"],
        "prefixes": 10,
        "bands": 5,
        "feature_layout": "sample,channel,prefix,band",
        "original_label_values": classes,
        "indexing": "subject, session, and trial are zero based",
    }
    payload = {
        "x": x, "y": y, "subject": subject, "session": session, "trial": trial,
        "metadata": json.dumps(metadata, sort_keys=True),
    }
    if emotion is not None:
        payload["emotion"] = emotion
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.output, **payload)
    archive = FeatureArchive.load(arguments.output)
    print(json.dumps({
        "output": str(arguments.output),
        "samples": int(len(archive.x)),
        "shape": list(archive.x.shape),
        "subjects": int(len(np.unique(archive.subject))),
        "classes": int(len(np.unique(archive.y))),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
