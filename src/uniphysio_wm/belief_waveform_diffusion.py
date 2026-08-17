from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F

from uniphysio_wm.waveform_diffusion import (
    StructuralResidualDiffusion,
    project_emg_probability_bursts,
    project_emg_rms_and_peaks,
    structural_diffusion_loss,
)


def _waveform_patches(values: torch.Tensor, patch_samples: int) -> torch.Tensor:
    usable = values.shape[-1] // int(patch_samples) * int(patch_samples)
    if usable == 0:
        raise ValueError("waveform is shorter than one structure patch")
    return values[:, :usable].reshape(len(values), -1, int(patch_samples))


def eeg_multiband_power_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int,
    patch_samples: int,
) -> torch.Tensor:
    """Match patch-level delta, theta, alpha, and beta log power."""

    predicted_patches = _waveform_patches(prediction, patch_samples)
    target_patches = _waveform_patches(target, patch_samples)
    predicted_patches = predicted_patches - predicted_patches.mean(-1, keepdim=True)
    target_patches = target_patches - target_patches.mean(-1, keepdim=True)
    predicted_spectrum = torch.fft.rfft(
        predicted_patches, dim=-1, norm="ortho"
    )
    target_spectrum = torch.fft.rfft(target_patches, dim=-1, norm="ortho")
    predicted_power = predicted_spectrum.real.square() + predicted_spectrum.imag.square()
    target_power = target_spectrum.real.square() + target_spectrum.imag.square()
    frequencies = torch.fft.rfftfreq(
        int(patch_samples),
        d=1.0 / float(sample_rate),
        device=prediction.device,
    )
    predicted_bands = []
    target_bands = []
    for low, high in ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0)):
        selected = (frequencies >= low) & (frequencies < high)
        if not selected.any():
            continue
        predicted_bands.append(
            torch.log1p(predicted_power[..., selected].mean(dim=-1))
        )
        target_bands.append(torch.log1p(target_power[..., selected].mean(dim=-1)))
    if not predicted_bands:
        return prediction.sum() * 0.0
    predicted_values = torch.stack(predicted_bands, dim=-1)
    target_values = torch.stack(target_bands, dim=-1)
    absolute = F.smooth_l1_loss(predicted_values, target_values)
    relative = F.smooth_l1_loss(
        torch.log_softmax(predicted_values, dim=-1),
        torch.log_softmax(target_values, dim=-1),
    )
    return 0.75 * absolute + 0.25 * relative


def emg_joint_structure_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int,
    patch_samples: int,
) -> Dict[str, torch.Tensor]:
    """Return differentiable envelope, RMS, and soft burst constraints."""

    predicted_patches = _waveform_patches(prediction, patch_samples)
    target_patches = _waveform_patches(target, patch_samples)
    usable = predicted_patches.shape[1] * int(patch_samples)
    kernel = max(1, min(round(0.25 * float(sample_rate)), usable))

    def descriptors(values: torch.Tensor, patches: torch.Tensor):
        envelope_samples = F.avg_pool1d(
            values[:, :usable].abs().unsqueeze(1),
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        )[..., :usable].squeeze(1)
        envelope = envelope_samples.reshape_as(patches).mean(dim=-1)
        rms = patches.square().mean(dim=-1).clamp_min(1e-8).sqrt()
        normalized = (envelope - envelope.mean(-1, keepdim=True)) / envelope.std(
            -1, keepdim=True, unbiased=False
        ).clamp_min(1e-4)
        return envelope, rms, (normalized - 1.0) / 0.35

    predicted_envelope, predicted_rms, predicted_burst_logits = descriptors(
        prediction, predicted_patches
    )
    target_envelope, target_rms, target_burst_logits = descriptors(
        target, target_patches
    )
    target_bursts = target_burst_logits >= 0.0
    positives = target_bursts.sum().to(prediction.dtype).clamp_min(1.0)
    negatives = target_bursts.numel() - positives
    positive_weight = (negatives / positives).clamp(1.0, 12.0)
    burst_bce = F.binary_cross_entropy_with_logits(
        predicted_burst_logits,
        target_bursts.to(prediction.dtype),
        pos_weight=positive_weight,
    )
    predicted_probability = predicted_burst_logits.sigmoid()
    target_probability = target_bursts.to(prediction.dtype)
    intersection = (predicted_probability * target_probability).sum()
    burst_dice = 1.0 - (2.0 * intersection + 1.0) / (
        predicted_probability.sum() + target_probability.sum() + 1.0
    )
    burst_occupancy = F.smooth_l1_loss(
        predicted_probability.mean(-1), target_probability.mean(-1)
    )
    envelope_scale = target_envelope.detach().mean(-1, keepdim=True).clamp_min(0.05)
    rms_scale = target_rms.detach().mean(-1, keepdim=True).clamp_min(0.05)
    return {
        "emg_envelope_loss": F.smooth_l1_loss(
            predicted_envelope / envelope_scale, target_envelope / envelope_scale
        ),
        "emg_rms_loss": F.smooth_l1_loss(
            predicted_rms / rms_scale, target_rms / rms_scale
        ),
        "emg_burst_loss": 0.55 * burst_bce
        + 0.35 * burst_dice
        + 0.10 * burst_occupancy,
    }


def project3_joint_structure_loss(
    output: Dict[str, torch.Tensor],
    target: torch.Tensor,
    valid: torch.Tensor,
    modalities: Sequence[str],
    sample_rate: int,
    patch_samples: int,
    fft_sizes: Sequence[int],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    """Augment diffusion training with Project3 modality-specific structure losses."""

    weights = dict(weights or {})
    losses = structural_diffusion_loss(
        output,
        target,
        valid,
        sample_rate,
        patch_samples,
        fft_sizes,
        weights,
    )
    prediction = output["waveform"]
    zero = prediction.sum() * 0.0
    eeg_loss = zero
    emg_losses = {
        "emg_envelope_loss": zero,
        "emg_rms_loss": zero,
        "emg_burst_loss": zero,
    }
    if "EEG" in modalities:
        eeg_index = tuple(modalities).index("EEG")
        selected = valid[:, eeg_index]
        if selected.any():
            eeg_loss = eeg_multiband_power_loss(
                prediction[selected, eeg_index],
                target[selected, eeg_index],
                sample_rate,
                patch_samples,
            )
    if "EMG" in modalities:
        emg_index = tuple(modalities).index("EMG")
        selected = valid[:, emg_index]
        if selected.any():
            emg_losses = emg_joint_structure_losses(
                prediction[selected, emg_index],
                target[selected, emg_index],
                sample_rate,
                patch_samples,
            )
    losses["eeg_multiband_loss"] = eeg_loss
    losses.update(emg_losses)
    losses["loss"] = (
        losses["loss"]
        + float(weights.get("eeg_multiband", 0.0)) * eeg_loss
        + float(weights.get("emg_envelope", 0.0)) * emg_losses["emg_envelope_loss"]
        + float(weights.get("emg_rms", 0.0)) * emg_losses["emg_rms_loss"]
        + float(weights.get("emg_burst", 0.0)) * emg_losses["emg_burst_loss"]
    )
    return losses


def build_frozen_structural_refiner(
    checkpoint: dict,
    shared_state_dim: int,
    dynamics_state_dim: int,
    structural_condition_channels: Optional[int] = None,
) -> StructuralResidualDiffusion:
    config = checkpoint["config"]
    diffusion = config["diffusion"]
    checkpoint_condition_channels = int(
        diffusion.get("structural_condition_channels", 2)
    )
    requested_condition_channels = int(
        structural_condition_channels
        if structural_condition_channels is not None
        else checkpoint_condition_channels
    )
    model = StructuralResidualDiffusion(
        tuple(config["data"]["modalities"]),
        int(shared_state_dim),
        int(dynamics_state_dim),
        diffusion_steps=int(diffusion.get("steps", 32)),
        channels=int(diffusion.get("channels", 48)),
        blocks=int(diffusion.get("blocks", 6)),
        residual_clip=float(diffusion.get("residual_clip", 6.0)),
        structural_condition_channels=requested_condition_channels,
    )
    state = dict(checkpoint["refiner_state"])
    if requested_condition_channels != checkpoint_condition_channels:
        if requested_condition_channels < checkpoint_condition_channels:
            raise ValueError("cannot remove pretrained structural condition channels")
        target_state = model.state_dict()
        for name, value in tuple(state.items()):
            if name.endswith(".input.weight") and value.shape != target_state[name].shape:
                expanded = torch.zeros_like(target_state[name])
                expanded[:, : value.shape[1]] = value
                state[name] = expanded
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.eval()
    return model


def append_emg_burst_condition(
    structural_condition: torch.Tensor,
    burst_probability: torch.Tensor,
    modalities: Sequence[str],
    patch_samples: int,
    required_channels: int,
) -> torch.Tensor:
    """Append calibrated EMG burst probability as a sample-level condition."""

    if structural_condition.shape[2] == int(required_channels):
        return structural_condition
    if structural_condition.shape[2] + 1 != int(required_channels):
        raise ValueError("unsupported structural condition channel expansion")
    samples = structural_condition.shape[-1]
    burst_samples = burst_probability.repeat_interleave(
        int(patch_samples), dim=-1
    )[:, :samples]
    if burst_samples.shape[-1] < samples:
        burst_samples = F.pad(
            burst_samples, (0, samples - burst_samples.shape[-1]), mode="replicate"
        )
    extra = structural_condition.new_zeros(
        structural_condition.shape[0], structural_condition.shape[1], 1, samples
    )
    emg_index = tuple(modalities).index("EMG")
    extra[:, emg_index, 0] = burst_samples.to(extra.dtype)
    return torch.cat((structural_condition, extra), dim=2)


def residual_strength_mask(
    base: torch.Tensor,
    modalities: Sequence[str],
    strengths: Dict[str, float],
) -> torch.Tensor:
    return torch.stack(
        [
            torch.full_like(base[:, index], float(strengths.get(modality, 0.0)))
            for index, modality in enumerate(modalities)
        ],
        dim=1,
    )


def eeg_spectral_projection(
    waveform: torch.Tensor,
    spectral_mean: torch.Tensor,
    patch_samples: int,
    blend: float,
) -> torch.Tensor:
    """Project EEG magnitude toward its predicted spectrum while retaining sampled phase."""

    if waveform.ndim != 2:
        raise ValueError("EEG waveform must be [batch, samples]")
    patch_samples = int(patch_samples)
    patches = waveform.reshape(waveform.shape[0], -1, patch_samples)
    centered = patches - patches.mean(dim=-1, keepdim=True)
    spectrum = torch.fft.rfft(centered, dim=-1, norm="ortho")
    bins = spectrum.shape[-1]
    target_log_magnitude = spectral_mean[:, : patches.shape[1], :bins]
    target_magnitude = torch.expm1(target_log_magnitude.clamp(-2.0, 3.0)).clamp_min(0.0)
    current_magnitude = spectrum.abs()
    mixed_magnitude = (1.0 - float(blend)) * current_magnitude + float(
        blend
    ) * target_magnitude
    phase = spectrum / current_magnitude.clamp_min(1e-6)
    projected = torch.fft.irfft(
        phase * mixed_magnitude,
        n=patch_samples,
        dim=-1,
        norm="ortho",
    )
    return (projected + patches.mean(dim=-1, keepdim=True)).flatten(1)


def _patch_descriptors(
    waveform: torch.Tensor,
    modality: str,
    patch_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    patches = waveform.reshape(waveform.shape[0], -1, int(patch_samples))
    if modality == "EEG":
        centered = patches - patches.mean(dim=-1, keepdim=True)
        log_spectrum = torch.log1p(
            torch.fft.rfft(centered, dim=-1, norm="ortho").abs()
        )
        return log_spectrum.mean(dim=-1), log_spectrum[..., 1:].mean(dim=-1)
    envelope = patches.abs().mean(dim=-1)
    rms = patches.square().mean(dim=-1).clamp_min(1e-8).sqrt()
    return envelope, rms


def emg_high_frequency_phase_randomization(
    waveform: torch.Tensor,
    sample_rate: int,
    generator: torch.Generator,
    blend: float,
    cutoff_hz: float = 20.0,
) -> torch.Tensor:
    """Vary EMG high-frequency phase while preserving its Fourier magnitude."""

    spectrum = torch.fft.rfft(waveform, dim=-1, norm="ortho")
    magnitude = spectrum.abs()
    current_phase = spectrum / magnitude.clamp_min(1e-6)
    random_angles = 2.0 * torch.pi * torch.rand(
        spectrum.shape,
        device=waveform.device,
        dtype=waveform.dtype,
        generator=generator,
    )
    random_phase = torch.complex(random_angles.cos(), random_angles.sin())
    frequency = torch.fft.rfftfreq(
        waveform.shape[-1], d=1.0 / float(sample_rate)
    ).to(waveform.device)
    high_frequency = frequency >= float(cutoff_hz)
    mixed_phase = (1.0 - float(blend)) * current_phase + float(blend) * random_phase
    mixed_phase = mixed_phase / mixed_phase.abs().clamp_min(1e-6)
    phase = torch.where(high_frequency.reshape(1, -1), mixed_phase, current_phase)
    return torch.fft.irfft(
        magnitude * phase,
        n=waveform.shape[-1],
        dim=-1,
        norm="ortho",
    )


def condition_consistency_score(
    members: torch.Tensor,
    probabilities: dict,
    structural_condition: torch.Tensor,
    modalities: Sequence[str],
    patch_samples: int,
    emg_burst_probability: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Return modality-specific target-free scores shaped [members, batch]."""

    eeg_scores = []
    emg_scores = []
    eeg_index = tuple(modalities).index("EEG")
    emg_index = tuple(modalities).index("EMG")
    patch_count = members.shape[-1] // int(patch_samples)
    eeg_target = probabilities["EEG"]["spectral_mean"][:, :patch_count]
    eeg_target_primary = eeg_target[..., : int(patch_samples) // 2 + 1].mean(-1)
    eeg_target_secondary = eeg_target[..., 1 : int(patch_samples) // 2 + 1].mean(-1)
    emg_envelope_target = structural_condition[:, emg_index, 0].reshape(
        members.shape[1], patch_count, int(patch_samples)
    ).mean(-1)
    emg_rms_target = structural_condition[:, emg_index, 1].reshape(
        members.shape[1], patch_count, int(patch_samples)
    ).mean(-1)
    emg_burst_target = (
        probabilities["EMG"]["burst_logits"][:, :patch_count].sigmoid()
        if emg_burst_probability is None
        else emg_burst_probability[:, :patch_count]
    )
    for member in members:
        eeg_primary, eeg_secondary = _patch_descriptors(
            member[:, eeg_index], "EEG", patch_samples
        )
        emg_envelope, emg_rms = _patch_descriptors(
            member[:, emg_index], "EMG", patch_samples
        )
        emg_burst_score = torch.sigmoid(
            (emg_envelope - emg_envelope.mean(-1, keepdim=True))
            / emg_envelope.std(-1, keepdim=True).clamp_min(1e-3)
        )
        eeg_scale = eeg_target_primary.abs().mean(-1).clamp_min(0.1)
        emg_scale = emg_rms_target.abs().mean(-1).clamp_min(0.05)
        eeg_score = (
            (eeg_primary - eeg_target_primary).abs().mean(-1) / eeg_scale
            + 0.5
            * (eeg_secondary - eeg_target_secondary).abs().mean(-1)
            / eeg_scale
        )
        emg_score = (
            (emg_envelope - emg_envelope_target).abs().mean(-1) / emg_scale
            + (emg_rms - emg_rms_target).abs().mean(-1) / emg_scale
            + (emg_burst_score - emg_burst_target).abs().mean(-1)
        )
        eeg_scores.append(eeg_score)
        emg_scores.append(emg_score)
    return {"EEG": torch.stack(eeg_scores), "EMG": torch.stack(emg_scores)}


@torch.no_grad()
def sample_belief_conditioned_waveforms(
    refiner: StructuralResidualDiffusion,
    base: torch.Tensor,
    recent: torch.Tensor,
    structural_condition: torch.Tensor,
    probabilities: dict,
    shared_state: torch.Tensor,
    dynamics_state: torch.Tensor,
    modalities: Sequence[str],
    patch_samples: int,
    sample_rate: int,
    ensemble_samples: int,
    sampling_steps: int,
    seed: int,
    residual_strengths: Dict[str, float],
    eeg_spectral_blend: float,
    emg_peak_exponent: float,
    emg_maximum_gain: float,
    emg_phase_randomization: float,
    emg_burst_probability: Optional[torch.Tensor] = None,
    emg_burst_threshold: float = 0.5,
    emg_burst_anchor: float = 0.85,
    emg_burst_redistribution: float = 0.0,
    preserve_base_emg_bursts: bool = False,
    emg_rms_projection_blend: float = 1.0,
    residual_output_gains: Optional[Dict[str, float]] = None,
) -> dict[str, torch.Tensor]:
    mask = residual_strength_mask(base, modalities, residual_strengths)
    active_modalities = tuple(
        modality
        for modality in modalities
        if float(residual_strengths.get(modality, 0.0)) > 0.0
    )
    members = []
    eeg_index = tuple(modalities).index("EEG")
    emg_index = tuple(modalities).index("EMG")
    for member_index in range(int(ensemble_samples)):
        generator = torch.Generator(device=base.device).manual_seed(
            int(seed) + 100003 * member_index
        )
        member = refiner.sample(
            base,
            recent,
            structural_condition,
            shared_state,
            dynamics_state,
            sampling_steps=int(sampling_steps),
            generator=generator,
            active_modalities=active_modalities,
            residual_mask=mask,
        )
        for modality, gain in dict(residual_output_gains or {}).items():
            if modality not in active_modalities:
                continue
            modality_index = tuple(modalities).index(modality)
            member[:, modality_index] = base[:, modality_index] + float(gain) * (
                member[:, modality_index] - base[:, modality_index]
            )
        if "EEG" in active_modalities:
            member[:, eeg_index] = eeg_spectral_projection(
                member[:, eeg_index],
                probabilities["EEG"]["spectral_mean"],
                int(patch_samples),
                float(eeg_spectral_blend),
            )
        if "EMG" in active_modalities:
            projected_emg = project_emg_rms_and_peaks(
                member[:, emg_index],
                structural_condition[:, emg_index, 1],
                int(patch_samples),
                peak_exponent=float(emg_peak_exponent),
                maximum_gain=float(emg_maximum_gain),
            )
            projection_blend = min(1.0, max(0.0, float(emg_rms_projection_blend)))
            member[:, emg_index] = (
                (1.0 - projection_blend) * member[:, emg_index]
                + projection_blend * projected_emg
            )
            member[:, emg_index] = emg_high_frequency_phase_randomization(
                member[:, emg_index],
                int(sample_rate),
                generator,
                float(emg_phase_randomization),
            )
            if emg_burst_probability is not None:
                if float(emg_burst_redistribution) > 0.0:
                    member[:, emg_index] = project_emg_probability_bursts(
                        member[:, emg_index],
                        structural_condition[:, emg_index, 1],
                        emg_burst_probability,
                        int(patch_samples),
                        float(emg_burst_threshold),
                        redistribution_strength=float(emg_burst_redistribution),
                        hard_event_lock=True,
                    )
                burst_lock = (
                    emg_burst_probability >= float(emg_burst_threshold)
                ).to(member.dtype).repeat_interleave(int(patch_samples), dim=-1)
                anchor = float(emg_burst_anchor)
                member[:, emg_index] = (
                    burst_lock
                    * (
                        anchor * base[:, emg_index]
                        + (1.0 - anchor) * member[:, emg_index]
                    )
                    + (1.0 - burst_lock) * member[:, emg_index]
                )
        members.append(member)
    # Keep the deterministic structural forecast as an explicit ensemble anchor.
    # Target-free condition consistency may fall back to it when sampled residuals
    # do not improve the physiology predicted by the belief state.
    member_stack = torch.cat((base.unsqueeze(0), torch.stack(members)), dim=0)
    scores = condition_consistency_score(
        member_stack,
        probabilities,
        structural_condition,
        modalities,
        int(patch_samples),
        emg_burst_probability,
    )
    batch_index = torch.arange(base.shape[0], device=base.device)
    selected = base.clone()
    selected_indices = {}
    for modality, modality_scores in scores.items():
        modality_index = tuple(modalities).index(modality)
        selected_index = modality_scores.argmin(dim=0)
        selected[:, modality_index] = member_stack[
            selected_index, batch_index, modality_index
        ]
        selected_indices[modality] = selected_index
    if preserve_base_emg_bursts:
        window = max(1, round(0.25 * int(sample_rate)))
        base_emg = base[:, emg_index]
        base_envelope = torch.nn.functional.avg_pool1d(
            base_emg.abs().unsqueeze(1), window, stride=window
        ).squeeze(1)
        base_threshold = base_envelope.mean(-1, keepdim=True) + base_envelope.std(
            -1, keepdim=True
        )
        burst_samples = (base_envelope > base_threshold).repeat_interleave(
            window, dim=-1
        )[:, : base_emg.shape[-1]]
        selected[:, emg_index] = torch.where(
            burst_samples, base_emg, selected[:, emg_index]
        )
    return {
        "selected": selected,
        "members": member_stack,
        "lower": torch.quantile(member_stack, 0.10, dim=0),
        "upper": torch.quantile(member_stack, 0.90, dim=0),
        "median": torch.quantile(member_stack, 0.50, dim=0),
        "member_scores": scores,
        "selected_member_index": selected_indices,
    }
