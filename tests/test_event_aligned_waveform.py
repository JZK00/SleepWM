from __future__ import annotations

import torch

from uniphysio_wm.event_aligned_waveform import (
    ecg_event_alignment_loss,
    shift_aware_shape_time_loss,
)


def test_shift_aware_loss_separates_shape_and_time() -> None:
    target = torch.zeros(1, 128)
    target[0, 48:53] = torch.tensor([0.0, 1.0, 3.0, 1.0, 0.0])
    shifted = torch.zeros_like(target)
    shifted[0, 56:61] = target[0, 48:53]
    aligned_shape, aligned_time = shift_aware_shape_time_loss(
        target, target, sample_rate=64
    )
    shifted_shape, shifted_time = shift_aware_shape_time_loss(
        shifted, target, sample_rate=64
    )
    assert shifted_shape < 0.01
    assert shifted_time > aligned_time
    distorted = shifted.clone()
    distorted[0, 56:61] = torch.tensor([0.0, -1.0, 3.0, -1.0, 0.0])
    distorted_shape, _ = shift_aware_shape_time_loss(
        distorted, target, sample_rate=64
    )
    assert distorted_shape > shifted_shape


def test_late_weighted_event_loss_is_finite_and_differentiable() -> None:
    samples = 640
    slots = 10
    target = torch.zeros(2, 1, samples)
    recent = torch.zeros_like(target)
    for position in range(48, samples, 64):
        target[:, 0, position] = -2.0
        recent[:, 0, position] = -2.0
    prediction = {
        "qrs_logits": torch.zeros(2, samples, requires_grad=True),
        "event_hazard_logits": torch.zeros(2, slots, requires_grad=True),
        "event_offset_samples": torch.zeros(2, slots, requires_grad=True),
        "rr_logits": torch.zeros(2, slots, 48, requires_grad=True),
        "rr_residual_gate_logits": torch.zeros(2, slots, requires_grad=True),
    }
    loss = ecg_event_alignment_loss(
        prediction,
        target,
        target,
        torch.ones(2, 1, dtype=torch.bool),
        ("ECG",),
        sample_rate=128,
        horizons_seconds=(1, 5),
        patch_samples=64,
        recent_waveform=recent,
        recursive_events=True,
        horizon_weights=(0.5, 1.0),
        late_event_weight=2.0,
    )["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction["rr_logits"].grad is not None
