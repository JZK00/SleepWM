from __future__ import annotations

import pytest
import torch

from dynamic_missing_baselines import DynamicMissingOutcomeBaseline


@pytest.mark.parametrize("architecture", ("saits", "patchtst", "rssm"))
def test_new_dynamic_baseline_shapes_and_finiteness(architecture: str) -> None:
    model = DynamicMissingOutcomeBaseline(
        architecture=architecture,
        modalities=3,
        horizons=(1, 2, 4, 10, 14),
        physiology_features=11,
        feature_dim=16,
        hidden_dim=32,
        layers=1,
        heads=4,
        dropout=0.0,
        latent_dim=12,
    )
    signals = torch.randn(2, 20, 3, 384)
    present = torch.ones(2, 20, 3, dtype=torch.bool)
    present[:, -6:, 0] = False
    present[:, -3:, 1:] = False
    signals = signals * present.unsqueeze(-1)
    output = model(signals, present)

    assert output["current_logits"].shape == (2, 5)
    assert output["future_logits"].shape == (2, 5, 5)
    assert output["future_physiology"].shape == (2, 5, 11)
    assert output["trajectory"].shape[0] == 2
    assert output["trajectory"].shape[-1] == 32
    for value in output.values():
        assert torch.isfinite(value).all()
