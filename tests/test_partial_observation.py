import torch

from uniphysio_wm.partial_observation import (
    DynamicObservationSpec,
    dynamic_observation_view,
)


MODALITIES = ("EEG", "ECG", "EMG")


def tensors():
    signals = torch.ones(2, 20, 3, 8)
    present = torch.ones(2, 20, 3, dtype=torch.bool)
    return signals, present


def test_tail_interruption_has_exact_duration():
    signals, present = tensors()
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


def test_recovery_restores_final_epochs():
    signals, present = tensors()
    _, observed, _ = dynamic_observation_view(
        signals,
        present,
        MODALITIES,
        DynamicObservationSpec("recovery", {name: 4 for name in MODALITIES}, 2),
    )
    assert observed[:, -2:].all()
    assert observed[:, -6:-2].sum() == 0


def test_natural_absence_is_never_restored():
    signals, present = tensors()
    present[:, 3, 0] = False
    _, observed, quality = dynamic_observation_view(
        signals,
        present,
        MODALITIES,
        DynamicObservationSpec("emg_tail", {"EMG": 1}),
    )
    assert not observed[:, 3, 0].any()
    assert quality[:, 3, 0].sum() == 0


def test_linear_decay_is_monotonic_and_causal():
    signals, present = tensors()
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

