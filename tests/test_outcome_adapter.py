from __future__ import annotations

import torch

from uniphysio_wm.outcome_adapter import TrajectoryOutcomeAdapter


def fake_output(freshness: float) -> dict[str, torch.Tensor]:
    batch, horizons, states, modalities, features = 2, 3, 8, 3, 11
    return {
        "predicted_states": torch.randn(batch, horizons, states),
        "belief_base_predicted_states": torch.randn(batch, horizons, states),
        "observation_reliability": torch.rand(batch, modalities),
        "observation_freshness": torch.full((batch, modalities), freshness),
        "observation_age_epochs": torch.full((batch, modalities), 1.0 - freshness),
        "recursive_log_variance": torch.randn(batch, horizons),
        "recursive_horizons": torch.tensor([1, 2, 4]),
        "belief_trajectory": torch.randn(batch, 20, states),
        "stage_logits": torch.randn(batch, horizons, 5),
        "future_physiology": torch.randn(batch, horizons, features),
    }


def test_adapter_is_identity_when_observations_are_fresh() -> None:
    adapter = TrajectoryOutcomeAdapter(8, 3, 5, 11)
    output = fake_output(1.0)
    adapted = adapter(output)
    assert torch.equal(adapted["stage_logits"], output["stage_logits"])
    assert torch.equal(adapted["future_physiology"], output["future_physiology"])
    assert torch.count_nonzero(adapted["stale_gate"]) == 0


def test_adapter_shapes_match_outcomes_when_stale() -> None:
    adapter = TrajectoryOutcomeAdapter(8, 3, 5, 11)
    output = fake_output(0.5)
    adapted = adapter(output)
    assert adapted["stage_logits"].shape == (2, 3, 5)
    assert adapted["future_physiology"].shape == (2, 3, 11)
    assert torch.all(adapted["stale_gate"] > 0)
