"""True batch-1 path profiling with raw latency retention."""

from __future__ import annotations

from dataclasses import dataclass
import os
import resource
import time

import numpy as np
import torch
from torch import Tensor


@dataclass
class ProfileResult:
    path: str
    device: str
    latencies_ms: np.ndarray
    action_counts: np.ndarray
    peak_memory_bytes: int
    cpu_rss_bytes: int
    cold_start_ms: float

    def summary(self) -> dict[str, float | int | str]:
        values = self.latencies_ms
        return {
            "path": self.path,
            "device": self.device,
            "iterations": int(len(values)),
            "median_ms": float(np.median(values)),
            "p90_ms": float(np.quantile(values, 0.90)),
            "p95_ms": float(np.quantile(values, 0.95)),
            "p99_ms": float(np.quantile(values, 0.99)),
            "throughput_per_second": float(1000.0 / values.mean()),
            "peak_memory_bytes": int(self.peak_memory_bytes),
            "cpu_rss_bytes": int(self.cpu_rss_bytes),
            "cold_start_ms": float(self.cold_start_ms),
        }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def profile_batch1(
    model: torch.nn.Module,
    x: Tensor,
    *,
    delta_t: Tensor | None = None,
    channel_mask: Tensor | None = None,
    path: str,
    warmup: int = 200,
    iterations: int = 1000,
) -> ProfileResult:
    if x.shape[0] != 1:
        raise ValueError("profiler contract requires batch size one")
    if warmup < 1 or iterations < 1:
        raise ValueError("profiler warmup/iterations must be positive")
    if path not in {"sparse", "dense", "primary_only"}:
        raise ValueError("path must be sparse, dense, or primary_only")
    device = x.device
    model.eval()

    def invoke():
        if path == "primary_only":
            if channel_mask is None:
                mask = torch.ones(x.shape[:2], dtype=torch.bool, device=device)
            else:
                mask = channel_mask
            logits, _state, _trajectory, _sequence = model.encoder(x, delta_t, mask)
            return logits, torch.zeros(1, device=device)
        output = model(
            x,
            delta_t=delta_t,
            channel_mask=channel_mask,
            dense_teacher=path == "dense",
        )
        return output.primary_logits, output.action_mask.sum(dim=1)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    cold_started = time.perf_counter_ns()
    _logits, _actions = invoke()
    _sync(device)
    cold_ms = (time.perf_counter_ns() - cold_started) / 1e6
    for _ in range(warmup):
        invoke()
    _sync(device)
    latencies = np.empty(iterations, dtype=np.float64)
    action_counts = np.empty(iterations, dtype=np.int16)
    for index in range(iterations):
        _sync(device)
        started = time.perf_counter_ns()
        _logits, actions = invoke()
        _sync(device)
        latencies[index] = (time.perf_counter_ns() - started) / 1e6
        action_counts[index] = int(actions.item())
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    # Linux reports KiB; macOS reports bytes.  Server profiling runs on Linux.
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if os.uname().sysname == "Linux":
        rss *= 1024
    return ProfileResult(
        path=path,
        device=str(device),
        latencies_ms=latencies,
        action_counts=action_counts,
        peak_memory_bytes=peak,
        cpu_rss_bytes=rss,
        cold_start_ms=cold_ms,
    )
