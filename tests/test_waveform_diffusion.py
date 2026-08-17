from __future__ import annotations

import torch

from uniphysio_wm.waveform_diffusion import (
    StructuralResidualDiffusion,
    align_ecg_events_to_probability_mode,
    align_ecg_to_causal_rr_phase,
    causal_ar_phase_anchor,
    event_centered_beat_morphology_loss,
    project_emg_rms_and_peaks,
    project_emg_probability_bursts,
    project_event_centered_beat_morphology,
    structural_diffusion_loss,
    warp_ecg_to_event_sequence,
    _eeg_autocorrelation_loss,
    _emg_burst_distribution_loss,
)


def build_refiner() -> StructuralResidualDiffusion:
    return StructuralResidualDiffusion(
        ("EEG", "ECG", "EMG"),
        shared_state_dim=8,
        dynamics_state_dim=4,
        diffusion_steps=8,
        channels=16,
        blocks=3,
    )


def test_diffusion_schedule_and_zero_initialized_anchor() -> None:
    refiner = build_refiner()
    assert torch.all(refiner.alpha_bar[1:] < refiner.alpha_bar[:-1])
    base = torch.randn(2, 3, 32)
    recent = torch.randn_like(base)
    target = torch.randn_like(base)
    condition = torch.randn(2, 3, 2, 32)
    shared = torch.randn(2, 8)
    dynamics = torch.randn(2, 3, 4)
    output = refiner.training_prediction(
        base,
        recent,
        target,
        condition,
        shared,
        dynamics,
        steps=torch.tensor([0, 7]),
        noise=torch.zeros_like(base),
    )
    assert torch.equal(output["waveform"], base)


def test_structural_diffusion_loss_reaches_all_denoisers() -> None:
    refiner = build_refiner()
    base = torch.randn(3, 3, 32)
    recent = torch.randn_like(base)
    target = torch.randn_like(base)
    condition = torch.randn(3, 3, 2, 32)
    shared = torch.randn(3, 8)
    dynamics = torch.randn(3, 3, 4)
    output = refiner.training_prediction(
        base, recent, target, condition, shared, dynamics
    )
    losses = structural_diffusion_loss(
        output,
        target,
        torch.ones(3, 3, dtype=torch.bool),
        sample_rate=16,
        patch_samples=8,
        fft_sizes=(8, 16),
    )
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    for denoiser in refiner.denoisers.values():
        gradient = denoiser.output[-1].weight.grad
        assert gradient is not None
        assert torch.count_nonzero(gradient) > 0


def test_diffusion_sampling_is_stochastic_and_shape_stable() -> None:
    refiner = build_refiner().eval()
    with torch.no_grad():
        for denoiser in refiner.denoisers.values():
            denoiser.output[-1].weight.normal_(mean=0.0, std=0.01)
    base = torch.randn(2, 3, 32)
    recent = torch.randn_like(base)
    condition = torch.randn(2, 3, 2, 32)
    shared = torch.randn(2, 8)
    dynamics = torch.randn(2, 3, 4)
    first = refiner.sample(
        base,
        recent,
        condition,
        shared,
        dynamics,
        sampling_steps=4,
        generator=torch.Generator().manual_seed(1),
    )
    second = refiner.sample(
        base,
        recent,
        condition,
        shared,
        dynamics,
        sampling_steps=4,
        generator=torch.Generator().manual_seed(2),
    )
    assert first.shape == base.shape
    assert not torch.equal(first, second)


def test_inactive_modality_is_preserved_exactly() -> None:
    refiner = build_refiner().eval()
    with torch.no_grad():
        for denoiser in refiner.denoisers.values():
            denoiser.output[-1].weight.normal_(mean=0.0, std=0.01)
    base = torch.randn(2, 3, 32)
    recent = torch.randn_like(base)
    condition = torch.randn(2, 3, 2, 32)
    shared = torch.randn(2, 8)
    dynamics = torch.randn(2, 3, 4)
    generated = refiner.sample(
        base,
        recent,
        condition,
        shared,
        dynamics,
        sampling_steps=4,
        generator=torch.Generator().manual_seed(3),
        active_modalities=("EEG", "EMG"),
    )
    assert torch.equal(generated[:, 1], base[:, 1])


def test_residual_mask_preserves_locked_ecg_samples() -> None:
    refiner = build_refiner().eval()
    with torch.no_grad():
        for denoiser in refiner.denoisers.values():
            denoiser.output[-1].weight.normal_(mean=0.0, std=0.01)
    base = torch.randn(2, 3, 32)
    recent = torch.randn_like(base)
    condition = torch.randn(2, 3, 2, 32)
    shared = torch.randn(2, 8)
    dynamics = torch.randn(2, 3, 4)
    residual_mask = torch.ones_like(base)
    residual_mask[:, 1, 12:20] = 0.0
    generated = refiner.sample(
        base,
        recent,
        condition,
        shared,
        dynamics,
        sampling_steps=4,
        generator=torch.Generator().manual_seed(4),
        residual_mask=residual_mask,
    )
    assert torch.equal(generated[:, 1, 12:20], base[:, 1, 12:20])


def test_masked_training_target_has_no_locked_residual() -> None:
    refiner = build_refiner()
    base = torch.randn(2, 3, 32)
    recent = torch.randn_like(base)
    target = torch.randn_like(base)
    condition = torch.randn(2, 3, 2, 32)
    shared = torch.randn(2, 8)
    dynamics = torch.randn(2, 3, 4)
    residual_mask = torch.ones_like(base)
    residual_mask[:, 1, 10:22] = 0.0
    output = refiner.training_prediction(
        base,
        recent,
        target,
        condition,
        shared,
        dynamics,
        residual_mask=residual_mask,
    )
    assert torch.count_nonzero(output["target_residual"][:, 1, 10:22]) == 0
    assert torch.equal(output["waveform"][:, 1, 10:22], base[:, 1, 10:22])


def test_event_centered_morphology_loss_handles_shifted_beats() -> None:
    target = torch.zeros(1, 128)
    prediction = torch.zeros_like(target)
    target[0, 48:53] = torch.tensor([0.0, 1.0, 3.0, 1.0, 0.0])
    prediction[0, 58:63] = target[0, 48:53]
    target_events = torch.zeros_like(target, dtype=torch.bool)
    prediction_events = torch.zeros_like(target, dtype=torch.bool)
    target_events[0, 50] = True
    prediction_events[0, 60] = True
    aligned = event_centered_beat_morphology_loss(
        prediction, target, prediction_events, target_events, sample_rate=64
    )
    distorted = prediction.clone()
    distorted[0, 58:63] = torch.tensor([0.0, -1.0, 3.0, -1.0, 0.0])
    misaligned = event_centered_beat_morphology_loss(
        distorted, target, prediction_events, target_events, sample_rate=64
    )
    assert aligned < 1e-5
    assert misaligned > aligned


def test_event_centered_projection_preserves_base_beat_shape() -> None:
    base = torch.zeros(1, 128)
    base[0, 48:53] = torch.tensor([0.0, 1.0, 3.0, 1.0, 0.0])
    candidate = torch.randn_like(base) * 0.2
    events = torch.zeros_like(base, dtype=torch.bool)
    events[0, 50] = True
    projected = project_event_centered_beat_morphology(
        candidate, base, events, sample_rate=64
    )
    start, stop = 50 - 13, 50 + 26
    base_segment = base[0, start:stop] - base[0, start:stop].mean()
    projected_segment = projected[0, start:stop] - projected[0, start:stop].mean()
    correlation = (base_segment * projected_segment).sum() / (
        base_segment.norm() * projected_segment.norm()
    ).clamp_min(1e-6)
    assert correlation > 0.999


def test_causal_ar_phase_anchor_is_finite_and_short_term() -> None:
    time = torch.arange(256).float()
    recent = torch.sin(2.0 * torch.pi * time / 16.0).unsqueeze(0)
    candidate = torch.ones(1, 128)
    anchored = causal_ar_phase_anchor(candidate, recent, sample_rate=64)
    assert torch.isfinite(anchored).all()
    assert torch.count_nonzero(anchored[:, :64]) > 0
    assert torch.equal(anchored[:, 64:], candidate[:, 64:])


def test_ecg_causal_rr_phase_moves_late_first_beat() -> None:
    candidate = torch.zeros(1, 256)
    candidate[0, 108] = 1.0
    predicted_events = torch.zeros_like(candidate, dtype=torch.bool)
    predicted_events[0, 108] = True
    recent_events = torch.zeros(1, 640, dtype=torch.bool)
    recent_events[0, [216, 344, 472, 600]] = True
    aligned = align_ecg_to_causal_rr_phase(
        candidate,
        predicted_events,
        recent_events,
        torch.ones(1, 256),
        sample_rate=128,
    )
    assert int(aligned[0].argmax()) == 88


def test_ecg_probability_mode_alignment_moves_soft_center() -> None:
    candidate = torch.zeros(1, 256)
    candidate[0, 100] = 1.0
    events = torch.zeros_like(candidate, dtype=torch.bool)
    events[0, 100] = True
    qrs_logits = torch.zeros_like(candidate)
    qrs_logits[0, 91] = 5.0
    aligned = align_ecg_events_to_probability_mode(
        candidate, events, qrs_logits, sample_rate=128
    )
    assert int(aligned[0].argmax()) == 91


def test_ecg_event_sequence_warp_preserves_shape_and_moves_peak() -> None:
    waveform = torch.zeros(1, 256)
    waveform[0, 96:105] = torch.tensor(
        [0.0, -0.5, -1.0, 0.0, 2.0, 0.0, -0.5, 0.0, 0.0]
    )
    source = torch.zeros_like(waveform, dtype=torch.bool)
    desired = torch.zeros_like(waveform, dtype=torch.bool)
    source[0, 100] = True
    desired[0, 88] = True
    warped = warp_ecg_to_event_sequence(waveform, source, desired)
    assert int(warped[0].argmax()) == 88
    assert torch.allclose(warped[0, 84:93], waveform[0, 96:105])
    assert torch.allclose(warped[0, 80:96], waveform[0, 92:108])
    assert torch.isfinite(warped).all()


def test_emg_projection_restores_rms_and_peak_contrast() -> None:
    candidate = torch.linspace(-0.1, 0.1, 64).repeat(2, 1)
    desired_rms = torch.full_like(candidate, 0.15)
    projected = project_emg_rms_and_peaks(candidate, desired_rms, patch_samples=32)
    assert projected.square().mean().sqrt() > candidate.square().mean().sqrt()
    assert projected.abs().max() / projected.square().mean().sqrt() > candidate.abs().max() / candidate.square().mean().sqrt()


def test_eeg_autocorrelation_loss_is_phase_invariant() -> None:
    time = torch.arange(512).float()
    target = torch.sin(2.0 * torch.pi * time / 32.0).unsqueeze(0)
    shifted = torch.roll(target, shifts=9, dims=-1)
    noise = torch.randn_like(target)
    assert _eeg_autocorrelation_loss(shifted, target, 128) < _eeg_autocorrelation_loss(
        noise, target, 128
    )


def test_emg_burst_distribution_penalizes_flattened_signal() -> None:
    target = torch.zeros(1, 256)
    target[:, 60:76] = torch.linspace(-1.0, 1.0, 16)
    target[:, 170:186] = torch.linspace(0.8, -0.8, 16)
    shifted = torch.roll(target, shifts=24, dims=-1)
    flattened = torch.full_like(target, target.abs().mean())
    assert _emg_burst_distribution_loss(
        shifted, target
    ) < _emg_burst_distribution_loss(flattened, target)


def test_probability_burst_projection_redistributes_energy() -> None:
    candidate = torch.randn(2, 128) * 0.1
    target_rms = torch.full_like(candidate, 0.2)
    probability = torch.tensor([[0.1, 0.9], [0.2, 0.8]])
    projected = project_emg_probability_bursts(
        candidate,
        target_rms,
        probability,
        patch_samples=64,
        decision_threshold=0.5,
    )
    patch_rms = projected.reshape(2, 2, 64).square().mean(dim=-1).sqrt()
    assert torch.all(patch_rms[:, 1] > patch_rms[:, 0])
    patch_envelope = projected.reshape(2, 2, 64).abs().mean(dim=-1)
    assert torch.all(patch_envelope[:, 1] > patch_envelope[:, 0])
    assert torch.allclose(
        patch_rms.square().mean(dim=-1).sqrt(),
        torch.full((2,), 0.2),
        atol=0.02,
    )


def test_hard_probability_burst_projection_locks_event_patch() -> None:
    candidate = torch.randn(2, 256) * 0.1
    target_rms = torch.full_like(candidate, 0.2)
    probability = torch.tensor([[0.1, 0.2, 0.9, 0.1], [0.8, 0.1, 0.2, 0.1]])
    projected = project_emg_probability_bursts(
        candidate,
        target_rms,
        probability,
        patch_samples=64,
        decision_threshold=0.5,
        redistribution_strength=0.55,
        hard_event_lock=True,
    )
    patch_rms = projected.reshape(2, 4, 64).square().mean(dim=-1).sqrt()
    assert patch_rms[0, 2] > patch_rms[0, [0, 1, 3]].max()
    assert patch_rms[1, 0] > patch_rms[1, 1:].max()
    assert torch.allclose(
        patch_rms.square().mean(dim=-1).sqrt(),
        torch.full((2,), 0.2),
        atol=0.02,
    )
