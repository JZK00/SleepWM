import math

import torch
import torch.nn as nn

from uniphysio_wm.partial_observation_model import (
    RecursiveBeliefCarryCorrectWorldModel,
)


def bare_model(state_dim: int = 4, modalities: int = 3):
    model = RecursiveBeliefCarryCorrectWorldModel.__new__(
        RecursiveBeliefCarryCorrectWorldModel
    )
    nn.Module.__init__(model)
    model.encoder = type("Encoder", (), {})()
    model.encoder.config = type("Config", (), {"modalities": tuple(range(modalities))})()
    model.belief_max_delta = 0.5
    model.belief_use_dynamics = True
    model.belief_correction_mode = "learned"
    model.belief_transition = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, state_dim))
    nn.init.zeros_(model.belief_transition[-1].weight)
    nn.init.zeros_(model.belief_transition[-1].bias)
    input_dim = 2 * state_dim + 2 * modalities
    model.belief_correction_gate = nn.Sequential(
        nn.LayerNorm(input_dim), nn.Linear(input_dim, state_dim), nn.Sigmoid()
    )
    return model


def test_full_observation_tracks_epoch_state_exactly():
    model = bare_model()
    states = torch.randn(2, 6, 4)
    present = torch.ones(2, 6, 3, dtype=torch.bool)
    belief, _, gates, _ = model._belief_trajectory(states, present)
    assert torch.allclose(belief, states)
    assert torch.allclose(gates, torch.ones_like(gates))


def test_no_observation_has_zero_correction():
    model = bare_model()
    with torch.no_grad():
        model.belief_transition[-1].bias.fill_(0.2)
    states = torch.randn(1, 5, 4)
    present = torch.ones(1, 5, 3, dtype=torch.bool)
    present[:, 2:] = False
    belief, priors, gates, corrections = model._belief_trajectory(states, present)
    assert torch.allclose(gates[:, 2:], torch.zeros_like(gates[:, 2:]))
    assert torch.allclose(corrections[:, 2:], torch.zeros_like(corrections[:, 2:]))
    assert torch.allclose(belief[:, 2:], priors[:, 2:])


def test_recovery_hard_corrects_to_observation():
    model = bare_model()
    states = torch.randn(1, 5, 4)
    present = torch.ones(1, 5, 3, dtype=torch.bool)
    present[:, 1:4] = False
    belief, _, gates, _ = model._belief_trajectory(states, present)
    assert torch.allclose(gates[:, 4], torch.ones_like(gates[:, 4]))
    assert torch.allclose(belief[:, 4], states[:, 4])


def test_dynamics_changes_state_while_persistence_does_not():
    model = bare_model()
    with torch.no_grad():
        model.belief_transition[-1].bias.fill_(0.3)
    states = torch.zeros(1, 5, 4)
    present = torch.ones(1, 5, 3, dtype=torch.bool)
    present[:, 1:] = False
    dynamic, _, _, _ = model._belief_trajectory(states, present, use_dynamics=True)
    persistence, _, _, _ = model._belief_trajectory(states, present, use_dynamics=False)
    assert dynamic[:, -1].abs().sum() > 0.0
    assert torch.allclose(persistence, torch.zeros_like(persistence))


def test_configured_no_dynamics_matches_persistence():
    model = bare_model()
    model.belief_use_dynamics = False
    with torch.no_grad():
        model.belief_transition[-1].bias.fill_(0.3)
    states = torch.zeros(1, 5, 4)
    present = torch.ones(1, 5, 3, dtype=torch.bool)
    present[:, 1:] = False
    ablated, _, _, _ = model._belief_trajectory(states, present)
    persistence, _, _, _ = model._belief_trajectory(
        states, present, use_dynamics=False
    )
    assert torch.allclose(ablated, persistence)


def test_ungated_correction_trusts_any_available_observation():
    model = bare_model()
    model.belief_correction_mode = "ungated"
    states = torch.randn(1, 4, 4)
    present = torch.ones(1, 4, 3, dtype=torch.bool)
    present[:, 2, 1:] = False
    belief, _, gates, _ = model._belief_trajectory(states, present)
    assert torch.allclose(gates[:, 2], torch.ones_like(gates[:, 2]))
    assert torch.allclose(belief[:, 2], states[:, 2])
