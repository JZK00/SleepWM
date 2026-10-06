from __future__ import annotations

import torch

from sleepwm.masking import DynamicObservationSpec, dynamic_observation_view


MODALITIES = ("EEG", "ECG", "EMG")


def tensors_partial_observation():
    signals = torch.ones(2, 20, 3, 8)
    present = torch.ones(2, 20, 3, dtype=torch.bool)
    return signals, present


def test_tail_interruption_has_exact_duration_partial_observation():
    signals, present = tensors_partial_observation()
    output, observed, quality = dynamic_observation_view(
        signals,
        present,
        MODALITIES,
        DynamicObservationSpec("ecg_tail", {"ECG": 4}),
    )
    assert observed[:, -4:, 1].sum() == 0
    assert observed[:, :-4, 1].all()
    assert torch.equal(output[:, -4:, 1], torch.zeros_like(output[:, -4:, 1]))
    assert quality[:, -4:, 1].sum() == 0


def test_recovery_restores_final_epochs_partial_observation():
    signals, present = tensors_partial_observation()
    _, observed, _ = dynamic_observation_view(
        signals,
        present,
        MODALITIES,
        DynamicObservationSpec("recovery", {name: 4 for name in MODALITIES}, 2),
    )
    assert observed[:, -2:].all()
    assert observed[:, -6:-2].sum() == 0


def test_natural_absence_is_never_restored_partial_observation():
    signals, present = tensors_partial_observation()
    present[:, 3, 0] = False
    _, observed, quality = dynamic_observation_view(
        signals,
        present,
        MODALITIES,
        DynamicObservationSpec("emg_tail", {"EMG": 1}),
    )
    assert not observed[:, 3, 0].any()
    assert quality[:, 3, 0].sum() == 0


def test_linear_decay_is_monotonic_and_causal_partial_observation():
    signals, present = tensors_partial_observation()
    output, observed, quality = dynamic_observation_view(
        signals,
        present,
        MODALITIES,
        DynamicObservationSpec("decay", {"EEG": 4}, profile="linear_decay"),
    )
    expected = torch.tensor([0.75, 0.50, 0.25, 0.00])
    assert torch.allclose(quality[0, -4:, 0], expected)
    assert torch.allclose(output[0, -4:, 0, 0], expected)
    assert observed[:, -1, 0].sum() == 0


import math

import torch
import torch.nn as nn

from sleepwm.state import RecursiveBeliefCarryCorrectWorldModel


def bare_model_recursive_belief_filter(state_dim: int = 4, modalities: int = 3):
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


def test_full_observation_tracks_epoch_state_exactly_recursive_belief_filter():
    model = bare_model_recursive_belief_filter()
    states = torch.randn(2, 6, 4)
    present = torch.ones(2, 6, 3, dtype=torch.bool)
    belief, _, gates, _ = model._belief_trajectory(states, present)
    assert torch.allclose(belief, states)
    assert torch.allclose(gates, torch.ones_like(gates))


def test_no_observation_has_zero_correction_recursive_belief_filter():
    model = bare_model_recursive_belief_filter()
    with torch.no_grad():
        model.belief_transition[-1].bias.fill_(0.2)
    states = torch.randn(1, 5, 4)
    present = torch.ones(1, 5, 3, dtype=torch.bool)
    present[:, 2:] = False
    belief, priors, gates, corrections = model._belief_trajectory(states, present)
    assert torch.allclose(gates[:, 2:], torch.zeros_like(gates[:, 2:]))
    assert torch.allclose(corrections[:, 2:], torch.zeros_like(corrections[:, 2:]))
    assert torch.allclose(belief[:, 2:], priors[:, 2:])


def test_recovery_hard_corrects_to_observation_recursive_belief_filter():
    model = bare_model_recursive_belief_filter()
    states = torch.randn(1, 5, 4)
    present = torch.ones(1, 5, 3, dtype=torch.bool)
    present[:, 1:4] = False
    belief, _, gates, _ = model._belief_trajectory(states, present)
    assert torch.allclose(gates[:, 4], torch.ones_like(gates[:, 4]))
    assert torch.allclose(belief[:, 4], states[:, 4])


def test_dynamics_changes_state_while_persistence_does_not_recursive_belief_filter():
    model = bare_model_recursive_belief_filter()
    with torch.no_grad():
        model.belief_transition[-1].bias.fill_(0.3)
    states = torch.zeros(1, 5, 4)
    present = torch.ones(1, 5, 3, dtype=torch.bool)
    present[:, 1:] = False
    dynamic, _, _, _ = model._belief_trajectory(states, present, use_dynamics=True)
    persistence, _, _, _ = model._belief_trajectory(states, present, use_dynamics=False)
    assert dynamic[:, -1].abs().sum() > 0.0
    assert torch.allclose(persistence, torch.zeros_like(persistence))


def test_configured_no_dynamics_matches_persistence_recursive_belief_filter():
    model = bare_model_recursive_belief_filter()
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


def test_ungated_correction_trusts_any_available_observation_recursive_belief_filter():
    model = bare_model_recursive_belief_filter()
    model.belief_correction_mode = "ungated"
    states = torch.randn(1, 4, 4)
    present = torch.ones(1, 4, 3, dtype=torch.bool)
    present[:, 2, 1:] = False
    belief, _, gates, _ = model._belief_trajectory(states, present)
    assert torch.allclose(gates[:, 2], torch.ones_like(gates[:, 2]))
    assert torch.allclose(belief[:, 2], states[:, 2])


import torch

from sleepwm.masking import nonempty_modality_subsets, random_full_biased_natural_modality_subset, random_modality_presence, random_natural_modality_subset, random_strict_natural_modality_subset, random_span_mask


def test_span_mask_shape_and_coverage_masking() -> None:
    mask = random_span_mask(5, 12, 0.25, min_span=2, max_span=4)
    assert mask.shape == (5, 12)
    assert mask.dtype == torch.bool
    assert torch.all(mask.sum(dim=1) >= 3)


def test_modality_dropout_never_drops_everything_masking() -> None:
    present = random_modality_presence(100, 3, 0.8, torch.device("cpu"))
    assert present.shape == (100, 3)
    assert present.any(dim=1).all()


def test_modality_dropout_is_reproducible_masking() -> None:
    first_generator = torch.Generator().manual_seed(9)
    second_generator = torch.Generator().manual_seed(9)
    first = random_modality_presence(32, 3, 0.7, torch.device("cpu"), generator=first_generator)
    second = random_modality_presence(32, 3, 0.7, torch.device("cpu"), generator=second_generator)
    assert torch.equal(first, second)


def test_uniform_subset_respects_natural_availability_masking() -> None:
    natural = torch.tensor([[True, True, True], [True, False, True], [False, True, False]])
    subset = random_natural_modality_subset(natural, torch.Generator().manual_seed(12))
    assert subset.any(dim=1).all()
    assert not (subset & ~natural).any()
    assert subset[2].tolist() == [False, True, False]


def test_strict_subset_drops_a_modality_when_possible_masking() -> None:
    natural = torch.tensor([[True, True, True], [True, False, True], [False, True, False]])
    subset = random_strict_natural_modality_subset(natural, torch.Generator().manual_seed(12))
    assert subset.any(dim=1).all()
    assert not (subset & ~natural).any()
    assert (subset[:2] != natural[:2]).any(dim=1).all()
    assert subset[2].tolist() == [False, True, False]


def test_full_biased_subset_probability_endpoints_masking() -> None:
    natural = torch.ones(16, 3, dtype=torch.bool)
    full = random_full_biased_natural_modality_subset(natural, 1.0)
    strict = random_full_biased_natural_modality_subset(
        natural,
        0.0,
        torch.Generator().manual_seed(17),
    )
    assert torch.equal(full, natural)
    assert strict.any(dim=1).all()
    assert (strict != natural).any(dim=1).all()


def test_three_modalities_have_seven_nonempty_subsets_masking() -> None:
    subsets = nonempty_modality_subsets(("EEG", "ECG", "EMG"))
    assert len(subsets) == 7
    assert ("EEG", "ECG", "EMG") in subsets
