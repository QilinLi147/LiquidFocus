#!/usr/bin/env python3
"""Convert FACED BIDS BDF recordings to 30-channel five-band sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt, welch
from scipy.integrate import trapezoid


CHANNELS = (
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6",
    "Cz", "C3", "C4", "T7", "T8", "CP1", "CP2", "CP5", "CP6",
    "Pz", "P3", "P4", "P7", "P8", "PO3", "PO4", "Oz", "O1", "O2",
)
MODERN_TO_LEGACY = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
LABELS = {"negative": 0, "neutral": 1, "positive": 2}
BANDS = ((1.0, 4.0), (4.0, 8.0), (8.0, 14.0), (14.0, 31.0), (31.0, 50.0))


def feature_trial(signal: np.ndarray, source_hz: float) -> np.ndarray:
    signal = signal - signal.mean(axis=0, keepdims=True)
    rounded_hz = int(round(source_hz))
    if rounded_hz != 250:
        signal = resample_poly(signal, 250, rounded_hz, axis=1)
    signal = sosfiltfilt(
        butter(4, [1.0, 50.0], btype="bandpass", fs=250, output="sos"),
        signal,
        axis=1,
    )
    windows = signal.shape[1] // 250
    if windows < 10:
        return np.empty((0, signal.shape[0], 10, 5), dtype=np.float32)
    samples = signal[:, : windows * 250].reshape(
        signal.shape[0], windows, 250
    ).transpose(1, 0, 2)
    frequency, power = welch(
        samples, fs=250, nperseg=250, noverlap=0, axis=-1, scaling="density"
    )
    features = []
    for low, high in BANDS:
        selected = (frequency >= low) & (frequency < (high if high < 50 else high + 1e-9))
        band_power = trapezoid(power[..., selected], frequency[selected], axis=-1)
        features.append(np.log(band_power + 1e-12))
    feature = np.stack(features, axis=-1).astype(np.float32)
    sequences = len(feature) // 10
    return feature[: sequences * 10].reshape(
        sequences, 10, signal.shape[0], 5
    ).transpose(0, 2, 1, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--participant", required=True, help="BIDS identifier, for example sub-001")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    eeg = arguments.bids_root / arguments.participant / "eeg"
    stem = f"{arguments.participant}_task-watchingVideoClips"
    bdf = eeg / f"{stem}_eeg.bdf"
    events_path = eeg / f"{stem}_events.tsv"
    raw = mne.io.read_raw_bdf(bdf, preload=False, verbose="ERROR")
    source_channels = [
        name if name in raw.ch_names else MODERN_TO_LEGACY[name] for name in CHANNELS
    ]
    events = pd.read_csv(events_path, sep="\t")
    events = events[events["binary_label"].isin(LABELS)]
    blocks = []
    labels = []
    videos = []
    offsets = []
    for row in events.itertuples(index=False):
        start = int(round((float(row.onset) + 1.0) * raw.info["sfreq"]))
        stop = int(round((float(row.onset) + float(row.duration) - 1.0) * raw.info["sfreq"]))
        signal = raw.get_data(picks=source_channels, start=start, stop=stop)
        sequence = feature_trial(signal, float(raw.info["sfreq"]))
        blocks.append(sequence)
        labels.extend([LABELS[str(row.binary_label)]] * len(sequence))
        videos.extend([int(row.video_index)] * len(sequence))
        offsets.extend(range(len(sequence)))
    if not blocks or sum(len(block) for block in blocks) == 0:
        raise ValueError("no valid ten-second sequence was extracted")
    x = np.concatenate(blocks).astype(np.float32, copy=False)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.output,
        x=x,
        y=np.asarray(labels, dtype=np.int64),
        video_index=np.asarray(videos, dtype=np.int64),
        sequence_offset=np.asarray(offsets, dtype=np.int64),
        participant=np.asarray(arguments.participant),
        channel_names=np.asarray(CHANNELS),
    )
    print(json.dumps({
        "output": str(arguments.output),
        "participant": arguments.participant,
        "samples": int(len(x)),
        "shape": list(x.shape),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
