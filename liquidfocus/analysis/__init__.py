"""Evaluation tools used by the Information Fusion experiments."""

from .state_gates import StateGateConfig, build_ifusion_v186_model
from .sparse_inference import build_ifusion_production_fused_revision_v2_model

build_analysis_model = build_ifusion_v186_model
build_sparse_model = build_ifusion_production_fused_revision_v2_model

__all__ = [
    "StateGateConfig",
    "build_analysis_model",
    "build_sparse_model",
    "build_ifusion_v186_model",
    "build_ifusion_production_fused_revision_v2_model",
]
