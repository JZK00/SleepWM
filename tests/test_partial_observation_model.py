import math

import torch
import torch.nn as nn

from uniphysio_wm.partial_observation_model import (
    FreshnessAwareCarryCorrectWorldModel,
)


def bare_model(modalities: int = 3, state_dim: int = 4):
    model = FreshnessAwareCarryCorrectWorldModel.__new__(
        FreshnessAwareCarryCorrectWorldModel
    )
    nn.Module.__init__(model)
    model.log_freshness_decay = nn.Parameter(
        torch.full((modalities,), math.log(math.expm1(0.35)))
    )
    model.partial_uncertainty_state_norm = nn.LayerNorm(state_dim)
    input_dim = state_dim + 2 * modalities + 1
    model.partial_uncertainty_head = nn.Sequential(
        nn.LayerNorm(input_dim), nn.Linear(input_dim, 1)
    )
    nn.init.zeros_(model.partial_uncertainty_head[-1].weight)
    nn.init.zeros_(model.partial_uncertainty_head[-1].bias)
    model.uncertainty_bias = nn.Parameter(torch.tensor(-2.0))
    model.uncertainty_age_log_scale = nn.Parameter(
        torch.full((modalities,), math.log(math.expm1(0.15)))
    )
    model.uncertainty_horizon_log_scale = nn.Parameter(
        torch.tensor(math.log(math.expm1(0.05)))
    )
    return model


def test_observation_age_counts_trailing_interruption():
    model = bare_model()
    signals = torch.zeros(2, 6, 3, 8)
    present = torch.ones(2, 6, 3, dtype=torch.bool)
    present[0, -2:, 0] = False
    present[0, -4:, 1] = False
    present[1, :, 2] = False

    age, freshness, availability, last = model._observation_metadata(
        signals, present
    )

    assert age[0].tolist() == [2.0, 4.0, 0.0]
    assert age[1].tolist() == [0.0, 0.0, 6.0]
    assert last[0].tolist() == [3, 1, 5]
    assert freshness[0, 0] > freshness[0, 1]
    assert freshness[1, 2] == 0.0
    assert availability[1, 2] == 0.0


def test_last_observed_state_is_gathered_per_modality():
    states = torch.arange(1 * 5 * 3 * 2).reshape(1, 5, 3, 2).float()
    indices = torch.tensor([[4, 2, 0]])
    gathered = FreshnessAwareCarryCorrectWorldModel._last_observed_states(
        states, indices
    )
    assert torch.equal(gathered[0, 0], states[0, 4, 0])
    assert torch.equal(gathered[0, 1], states[0, 2, 1])
    assert torch.equal(gathered[0, 2], states[0, 0, 2])


def test_uncertainty_increases_with_interruption_and_horizon():
    model = bare_model()
    state = torch.zeros(1, 4)
    reliability = torch.ones(1, 3)
    complete = model._partial_log_variance(
        state, torch.zeros(1, 3), reliability, (1, 2, 4)
    )
    interrupted = model._partial_log_variance(
        state, torch.full((1, 3), 4.0), reliability, (1, 2, 4)
    )
    assert torch.all(interrupted > complete)
    assert complete[0, 0] < complete[0, 1] < complete[0, 2]

