"""Numerical and trainability checks on the shared model components, using toy input."""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
import torch
from torch.nn import functional as F

from sleepwm import SleepWM, default_config


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    return SleepWM(default_config(compact=True))


def test_missing_and_returned_sensors_are_finite(model):
    model.eval()
    signals = torch.randn(5, 20, 3, 3840)
    present = torch.ones(5, 20, 3, dtype=torch.bool)
    present[1, -4:, 0] = False       # EEG disappears.
    present[2, -4:, :] = False      # All sensors disappear.
    present[3, 10:18, :] = False    # Sensors return for the last two epochs.
    present[4, :, :] = False        # No observations in the entire history.
    signals[~present] = float("nan")
    with torch.no_grad():
        output = model(signals, present)
    assert output["stage_logits"].shape == (5, 3, 5)
    assert output["future_physiology"].shape == (5, 3, 11)
    assert output["transition_risk"].shape == (5, 3)
    assert all(torch.isfinite(value).all() for value in output.values())
    assert torch.allclose(output["stage_probabilities"].sum(-1), torch.ones(5, 3), atol=1e-6)
    risk = output["transition_risk"]
    assert ((risk >= 0) & (risk <= 1)).all()
    assert (risk[:, 1:] >= risk[:, :-1]).all()
    with torch.no_grad():
        clean_output = model(torch.nan_to_num(signals), present)
    for name in output:
        assert torch.equal(output[name], clean_output[name]), name


def test_all_six_trainable_components_receive_an_update(model):
    model.train()
    torch.manual_seed(11)
    signals = torch.randn(2, 20, 3, 3840)
    present = torch.ones(2, 20, 3, dtype=torch.bool)
    present[:, -4:, 0] = False
    present[1, -2:, :] = False
    modules = {name: getattr(model, name) for name in (
        "state_model", "outcome_adapter", "direct_model", "latent_adapter",
        "direct_event", "event_correction",
    )}
    before = {name: {key: parameter.detach().clone()
                     for key, parameter in module.named_parameters() if parameter.requires_grad}
              for name, module in modules.items()}
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    output = model(signals, present)
    target_stage = torch.tensor([[0, 1, 2], [3, 4, 0]])
    target_event = torch.tensor([[0., 0., 1.], [0., 1., 1.]])
    loss = (F.cross_entropy(output["stage_logits"].reshape(-1, 5), target_stage.reshape(-1))
            + F.mse_loss(output["future_physiology"], torch.zeros(2, 3, 11))
            + F.binary_cross_entropy(output["transition_risk"], target_event))
    assert torch.isfinite(loss)
    loss.backward()
    for name, module in modules.items():
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads), name
        assert any(torch.count_nonzero(g) > 0 for g in grads), name
    optimizer.step()
    for name, module in modules.items():
        assert any(not torch.equal(before[name][key], parameter.detach())
                   for key, parameter in module.named_parameters() if key in before[name]), name
    assert all(not p.requires_grad for p in model.state_model.waveform_decoder.parameters())


def test_available_nan_and_invalid_mask_are_rejected(model):
    signals = torch.zeros(1, 20, 3, 3840)
    present = torch.ones(1, 20, 3, dtype=torch.bool)
    signals[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(signals, present)
    with pytest.raises(ValueError, match="boolean"):
        model(torch.nan_to_num(signals), present.float())
