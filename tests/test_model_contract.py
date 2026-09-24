from __future__ import annotations

import torch

from liquidfocus import build_model
from liquidfocus.analysis import StateGateConfig, build_analysis_model
from liquidfocus.training import model_config


def fitted_model():
    model = build_model(classes=3, config=model_config())
    model.encoder.set_observation_input_transform(
        torch.zeros(1, 62, 1, 5), torch.ones(1, 62, 1, 5)
    )
    model.encoder.set_stable_feature_transform(torch.zeros(935), torch.ones(935))
    model.encoder.set_anchor_state(torch.zeros(3, 935), torch.tensor([0.2, 0.0, -0.2]))
    return model


def test_prediction_is_decoupled_from_evidence_path() -> None:
    model = fitted_model().eval()
    values = torch.randn(2, 62, 10, 5)
    output = model(values, dense_teacher=True)
    assert torch.equal(output.logits, output.primary_logits)
    assert output.dense_region_logits.shape == (2, 7, 3)
    assert output.dense_channel_logits.shape == (2, 62, 3)


def test_analysis_model_loads_the_training_state() -> None:
    trained = fitted_model().eval()
    analysis = build_analysis_model(classes=3, config=model_config())
    analysis.load_state_dict(trained.state_dict(), strict=True)
    analysis.set_state_gates(StateGateConfig())
    output = analysis.eval()(torch.randn(2, 62, 10, 5), dense_teacher=False)
    assert torch.equal(output.logits, output.primary_logits)
    assert output.node_prefix_features.shape == (2, 10, 620)
    assert output.relation_prefix_features.shape == (2, 10, 315)
