"""The public rename must not invalidate existing imports or parameter files."""

import importlib
import pickle

import torch

from liquidfocus import EvidenceDecoupledLiquidFocus, build_model
from liquidfocus.training import model_config


def test_legacy_api_is_canonical_api():
    legacy = importlib.import_module("liquidfocus_eeg")
    assert legacy.build_model is build_model
    assert legacy.EvidenceDecoupledLiquidFocusEEG is EvidenceDecoupledLiquidFocus


def test_legacy_submodules_are_canonical_modules():
    for suffix in ("compact", "data", "training", "v186_evidence_decoupled",
                   "analysis", "analysis.schema", "analysis.sparse_inference"):
        assert importlib.import_module("liquidfocus_eeg." + suffix) is importlib.import_module("liquidfocus." + suffix)


def test_legacy_class_reference_resolves():
    saved_reference = b"cliquidfocus_eeg.v186_evidence_decoupled\nEvidenceDecoupledLiquidFocusEEG\n."
    assert pickle.loads(saved_reference) is EvidenceDecoupledLiquidFocus


def test_parameter_state_loads_without_key_changes():
    canonical = build_model(classes=3, config=model_config())
    legacy = importlib.import_module("liquidfocus_eeg").build_model(
        classes=3, config=model_config()
    )
    state = canonical.state_dict()
    result = legacy.load_state_dict(state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert state.keys() == legacy.state_dict().keys()
    assert all(torch.equal(value, legacy.state_dict()[name]) for name, value in state.items())
