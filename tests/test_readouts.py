from __future__ import annotations

import torch

from sleepwm.readouts import TrajectoryOutcomeAdapter


def fake_output_outcome_adapter(freshness: float) -> dict[str, torch.Tensor]:
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


def test_adapter_is_identity_when_observations_are_fresh_outcome_adapter() -> None:
    adapter = TrajectoryOutcomeAdapter(8, 3, 5, 11)
    output = fake_output_outcome_adapter(1.0)
    adapted = adapter(output)
    assert torch.equal(adapted["stage_logits"], output["stage_logits"])
    assert torch.equal(adapted["future_physiology"], output["future_physiology"])
    assert torch.count_nonzero(adapted["stale_gate"]) == 0


def test_adapter_shapes_match_outcomes_when_stale_outcome_adapter() -> None:
    adapter = TrajectoryOutcomeAdapter(8, 3, 5, 11)
    output = fake_output_outcome_adapter(0.5)
    adapted = adapter(output)
    assert adapted["stage_logits"].shape == (2, 3, 5)
    assert adapted["future_physiology"].shape == (2, 3, 11)
    assert torch.all(adapted["stale_gate"] > 0)


import torch
from sleepwm.readouts import build_direct_branch
from sleepwm.readouts import LatentHazardSafeAdapter
from sleepwm.readouts import ReliabilityGatedEventCorrection


def test_final_hazards_are_monotone_and_stage_readout_is_preserved_final_adapters():
    torch.manual_seed(1)
    b, h, d, m, k, f = 2, 5, 16, 3, 5, 11
    state = {
        "predicted_states": torch.randn(b, h, d),
        "belief_base_predicted_states": torch.randn(b, h, d),
        "belief_trajectory": torch.randn(b, 20, d),
        "observation_reliability": torch.rand(b, m),
        "observation_freshness": torch.rand(b, m),
        "observation_age_epochs": torch.ones(b, m) * 4,
        "recursive_log_variance": torch.randn(b, h),
        "recursive_horizons": torch.tensor([1, 2, 4, 10, 14]),
        "current_stage_logits": torch.randn(b, k),
    }
    logits = torch.randn(b, h, k)
    latent = LatentHazardSafeAdapter(d, m, k, f).eval()
    output = latent(state, logits, torch.randn(b, h, f), torch.randn(b, h, f))
    assert torch.equal(output["stage_logits"], logits)
    correction = ReliabilityGatedEventCorrection(m, k).eval()
    result = correction(state, logits, output["interval_hazard"], torch.rand(b, h))
    risk = result["transition_risk"]
    assert torch.isfinite(risk).all()
    assert ((risk >= 0) & (risk <= 1)).all()
    assert (risk[:, 1:] >= risk[:, :-1] - 1e-7).all()


def test_direct_dependency_ignores_unavailable_signal_values_final_adapters():
    config = {"data": {"modalities": ["EEG", "ECG", "EMG"], "future_horizons": [1,2,4], "num_classes": 5},
              "physiology": {"feature_names": list(range(11))},
              "model": {"architecture": "grud", "feature_dim": 8, "hidden_dim": 16, "dropout": 0.0}}
    model = build_direct_branch(config).eval()
    x = torch.randn(2, 8, 3, 128)
    present = torch.ones(2,8,3,dtype=torch.bool)
    present[:, -4:, :] = False
    altered = x.clone()
    altered[:, -4:, :] = 1000 * torch.randn_like(altered[:, -4:, :])
    with torch.inference_mode():
        a, b = model(x, present), model(altered, present)
    for key in a:
        assert torch.allclose(a[key], b[key], atol=1e-7), key
