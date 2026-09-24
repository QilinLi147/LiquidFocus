"""Public API for LiquidFocus."""

from .compact import CompactConfig
from .v186_evidence_decoupled import (
    EvidenceDecoupledLiquidFocus,
    build_v186_model,
)

build_model = build_v186_model

__all__ = [
    "CompactConfig",
    "EvidenceDecoupledLiquidFocus",
    "build_model",
    "build_v186_model",
]
