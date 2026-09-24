"""V186: score-centered liquid primary with prediction-decoupled evidence."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from .compact import CompactConfig, CompactOutput
from .v185_exact_anchor_liquid import (
    ExactAnchorLiquidEncoder, ExactAnchorLiquidFocus,
    ExactAnchorLiquidReadout,
)


BRANCH = "v186_score_centered_primary_prediction_decoupled_evidence"


class ScoreCenteredLiquidReadout(ExactAnchorLiquidReadout):
    """Single readout with a train-union residual-centering buffer."""

    def __init__(self, classes: int, regions: int, width: int, dropout: float) -> None:
        super().__init__(classes, regions, width, dropout)
        self.register_buffer("residual_bias", torch.zeros(classes))

    def forward(self, flattened: Tensor) -> Tensor:
        summary = flattened.reshape(-1, self.regions, 4, self.width)
        stable = summary[:, :, 0, : self.classes].sum(dim=1)
        fast = summary[:, :, :, self.width // 2 :].reshape(summary.shape[0], -1)
        residual = self.residual_bound * torch.tanh(self.rhythm(fast))
        centered = residual - self.residual_bias.to(residual.dtype)[None]
        enabled = self.residual_enabled.to(residual.dtype)
        return stable + enabled * self.residual_scale.to(residual.dtype) * centered

    @torch.no_grad()
    def set_residual_bias(self, value: Tensor | np.ndarray) -> None:
        bias = torch.as_tensor(
            value, dtype=self.residual_bias.dtype, device=self.residual_bias.device
        )
        if bias.shape != self.residual_bias.shape or not torch.isfinite(bias).all():
            raise ValueError("V186 residual bias must be one finite value per class")
        self.residual_bias.copy_(bias)


class ScoreCenteredLiquidEncoder(ExactAnchorLiquidEncoder):
    def __init__(
        self, classes: int, channels: int, bands: int, regions: int,
        config: CompactConfig,
    ) -> None:
        super().__init__(classes, channels, bands, regions, config)
        self.primary_readout = ScoreCenteredLiquidReadout(
            classes, regions, self.width, config.dropout
        )


class EvidenceDecoupledLiquidFocus(ExactAnchorLiquidFocus):
    """Router/refiner produce evidence, while prediction remains the primary."""

    def __init__(
        self, classes: int, channels: int = 62, time: int = 10,
        bands: int = 5, regions: int = 7,
        config: Optional[CompactConfig] = None,
    ) -> None:
        super().__init__(
            classes=classes, channels=channels, time=time, bands=bands,
            regions=regions, config=config,
        )
        self.encoder = ScoreCenteredLiquidEncoder(
            classes, channels, bands, regions, self.config
        )
        if list(dict(self.named_children())) != ["encoder", "router", "refiner"]:
            raise AssertionError("V186 must expose exactly encoder/router/refiner")

    def forward(
        self, x: Tensor, delta_t: Optional[Tensor] = None,
        channel_mask: Optional[Tensor] = None,
        dense_teacher: Optional[bool] = None,
    ) -> CompactOutput:
        output = super().forward(
            x, delta_t=delta_t, channel_mask=channel_mask,
            dense_teacher=dense_teacher,
        )
        # The hierarchy remains active and auditable through its actions,
        # gains, and dense candidate evidence.  It cannot overwrite the main
        # recognition result in the evidence-only deployment contract.
        output.logits = output.primary_logits
        return output


def build_v186_model(
    *, classes: int, config: CompactConfig, channels: int = 62,
    time: int = 10, bands: int = 5,
) -> EvidenceDecoupledLiquidFocus:
    return EvidenceDecoupledLiquidFocus(
        classes=classes, channels=channels, time=time, bands=bands,
        config=config,
    )


@torch.no_grad()
def fit_union_global_residual_bias(
    model: EvidenceDecoupledLiquidFocus,
    normalized_values: np.ndarray,
    *, device: torch.device, batch_size: int = 256,
    fit_partition: str = "train+development_union",
) -> dict[str, Any]:
    if fit_partition != "train+development_union":
        raise PermissionError("V186 residual bias must use the refit union")
    values = np.asarray(normalized_values, dtype=np.float32)
    if values.ndim != 4 or values.shape[1:] != (62, 10, 5):
        raise ValueError("V186 residual-bias input must be [N,62,10,5]")
    previous_scale = float(model.encoder.primary_readout.residual_scale)
    model.eval(); residual = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values)), batch_size=int(batch_size),
        shuffle=False, num_workers=0,
    )
    for (x,) in loader:
        x = x.to(device)
        model.encoder.primary_readout.set_residual_scale(0.0)
        stable = model(x, dense_teacher=False).primary_logits
        model.encoder.primary_readout.set_residual_scale(1.0)
        full = model(x, dense_teacher=False).primary_logits
        residual.append((full - stable).cpu())
    residual_value = torch.cat(residual).mean(dim=0)
    model.encoder.primary_readout.set_residual_bias(residual_value)
    model.encoder.primary_readout.set_residual_scale(previous_scale)
    centered_mean = residual_value - model.encoder.primary_readout.residual_bias.cpu()
    if not torch.equal(centered_mean, torch.zeros_like(centered_mean)):
        raise AssertionError("V186 fitted residual bias failed exact centering")
    value = residual_value.numpy().astype(np.float32, copy=False)
    return {
        "schema": "liquidfocus_v186_union_global_residual_bias_v1",
        "fit_partition": fit_partition, "fit_samples": int(len(values)),
        "method": "global mean of scale-one liquid residual",
        "score_domain": "pre-causal primary liquid score",
        "bias_float32": value.tolist(),
        "labels_used": False, "outer_used": False,
        "parallel_classifier": False,
    }
