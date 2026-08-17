from __future__ import annotations

import copy

import pytest
import torch

from uniphysio_wm.mainline import (
    MISSING_CONDITION_PREFIXES,
    OBSERVATION_PREFIXES,
    WAVEFORM_PREFIX,
    assemble_mainline_components,
    build_mainline_model,
)


def tiny_mainline_config() -> dict:
    return {
        "data": {
            "modalities": ["EEG", "ECG", "EMG"],
            "stored_modalities": ["EEG", "ECG", "EMG"],
            "sample_rate": 32,
            "epoch_seconds": 2,
            "num_classes": 5,
            "future_horizons": [1, 2, 3],
        },
        "model": {
            "patch_samples": 8,
            "d_model": 16,
            "layers": 1,
            "heads": 4,
            "dropout": 0.0,
            "transition_layers": 1,
            "transition_heads": 4,
        },
        "physiology": {
            "feature_groups": {"EEG": ["a"], "ECG": ["b"], "EMG": ["c"]},
            "hidden_dim": 16,
            "physiology_dynamics_dim": 8,
            "physiology_dynamics_layers": 1,
            "physiology_dynamics_heads": 2,
        },
        "waveform": {
            "horizons_seconds": [1, 2],
            "max_seconds": 2,
            "patch_samples": 8,
            "decoder_dim": 16,
            "decoder_layers": 1,
            "decoder_heads": 4,
            "structured_event_heads": True,
            "probabilistic_event_heads": True,
            "ecg_refractory_event_head": True,
            "ecg_rr_bins": 8,
            "output_baseline": "none",
            "physiological_event_renderer": True,
            "missing_modality_conditioning": True,
            "missing_physiology_calibration": True,
            "missing_emg_teacher_calibration": True,
        },
        "observation_repair": {"hidden_dim": 16, "task_hidden_dim": 16},
        "recursive": {
            "direct_horizons": [1, 2],
            "anchor_direct": True,
            "hidden_dim": 16,
        },
    }


def _filled_state(state: dict, prefix: str, value: float) -> dict:
    result = {}
    for key, tensor in state.items():
        clone = tensor.clone()
        if key.startswith(prefix) and clone.is_floating_point():
            clone.fill_(value)
        result[key] = clone
    return result


def test_mainline_assembly_preserves_component_ownership() -> None:
    model = build_mainline_model(tiny_mainline_config())
    initial = model.state_dict()
    recursive = _filled_state(initial, "", 1.0)
    waveform = _filled_state(initial, WAVEFORM_PREFIX, 2.0)
    waveform = {
        key: value
        for key, value in waveform.items()
        if not key.startswith(MISSING_CONDITION_PREFIXES)
    }
    observation = copy.deepcopy(recursive)
    checks = assemble_mainline_components(
        model,
        {"model_state": observation},
        {"model_state": recursive},
        {"model_state": waveform},
    )
    loaded = model.state_dict()
    for key, tensor in loaded.items():
        if key.startswith(MISSING_CONDITION_PREFIXES):
            expected = initial[key]
        else:
            expected = waveform[key] if key.startswith(WAVEFORM_PREFIX) else recursive[key]
        assert torch.equal(tensor, expected)
    assert checks["o2_preserved_in_r3"]
    assert checks["all_parameters_frozen"]
    assert checks["w5_waveform_tensor_count"] > 0
    assert checks["zero_initialized_missing_condition_tensor_count"] > 0


def test_mainline_assembly_rejects_broken_o2_lineage() -> None:
    model = build_mainline_model(tiny_mainline_config())
    state = model.state_dict()
    observation = copy.deepcopy(state)
    key = next(key for key in state if key.startswith(OBSERVATION_PREFIXES))
    observation[key] = observation[key].clone()
    observation[key].reshape(-1)[0] += 1.0
    with pytest.raises(ValueError, match="does not preserve"):
        assemble_mainline_components(
            model,
            {"model_state": observation},
            {"model_state": state},
            {"model_state": state},
        )


def test_mainline_exposes_long_state_and_short_waveform_interfaces() -> None:
    model = build_mainline_model(tiny_mainline_config()).eval()
    history = torch.randn(2, 2, 3, 64)
    present = torch.ones(2, 2, 3, dtype=torch.bool)
    present[:, :, 0] = False
    history[:, :, 0] = 0.0
    with torch.no_grad():
        state_output = model.rollout_context_horizons(
            history, present, (1, 2, 3)
        )
        waveform_output = model.rollout(history, present)
    assert state_output["predicted_states"].shape == (2, 3, 16)
    assert state_output["stage_logits"].shape == (2, 3, 5)
    assert state_output["future_physiology"].shape == (2, 3, 3)
    assert waveform_output["future_waveforms"].shape == (2, 3, 64)
    assert set(waveform_output["future_waveform_probabilities"]) == {
        "EEG",
        "ECG",
        "EMG",
    }


def test_missing_physiology_calibration_is_gated_and_receives_gradients() -> None:
    model = build_mainline_model(tiny_mainline_config()).eval()
    history = torch.randn(2, 2, 3, 64)
    present = torch.ones(2, 2, 3, dtype=torch.bool)
    with torch.no_grad():
        full = model.rollout(history, present)
    assert torch.count_nonzero(
        full["future_waveform_probabilities"]["EEG"][
            "missing_log_spectral_energy_shift"
        ]
    ) == 0
    assert torch.count_nonzero(
        full["future_waveform_probabilities"]["EMG"]["missing_rms_log_scale"]
    ) == 0

    present[:, :, (0, 2)] = False
    history[:, :, (0, 2)] = 0.0
    missing = model.rollout(history, present)
    probability = missing["future_waveform_probabilities"]
    loss = probability["EEG"]["spectral_mean"].mean() + probability["EMG"][
        "rms_mean"
    ].mean()
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.waveform_decoder.missing_calibration_parameters()
    ]
    assert all(gradient is not None for gradient in gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_emg_teacher_rms_intercept_only_changes_missing_emg() -> None:
    model = build_mainline_model(tiny_mainline_config()).eval()
    history = torch.randn(2, 2, 3, 64)
    present = torch.ones(2, 2, 3, dtype=torch.bool)
    with torch.no_grad():
        full_before = model.rollout(history, present)["future_waveforms"]
        model.waveform_decoder.missing_emg_log_rms_bias.copy_(
            torch.tensor([2.0, 3.0, 4.0]).log()
        )
        full_after = model.rollout(history, present)["future_waveforms"]
    assert torch.equal(full_before, full_after)

    for retained, expected_scale in (((0,), 2.0), ((1,), 3.0), ((0, 1), 4.0)):
        retained_present = torch.zeros_like(present)
        retained_present[:, :, retained] = True
        retained_history = history * retained_present.unsqueeze(-1)
        with torch.no_grad():
            missing = model.rollout(retained_history, retained_present)
        teacher_scale = missing["future_waveform_probabilities"]["EMG"][
            "missing_teacher_rms_log_scale"
        ]
        assert torch.allclose(
            teacher_scale,
            torch.full_like(teacher_scale, torch.log(torch.tensor(expected_scale))),
        )
