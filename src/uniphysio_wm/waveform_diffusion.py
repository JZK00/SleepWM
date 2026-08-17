from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_embedding(steps: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=steps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = steps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if embedding.shape[-1] < dimension:
        embedding = F.pad(embedding, (0, dimension - embedding.shape[-1]))
    return embedding


def _cosine_alpha_bar(steps: int, offset: float = 0.008) -> torch.Tensor:
    positions = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    values = torch.cos(
        ((positions / steps + offset) / (1.0 + offset)) * math.pi * 0.5
    ).square()
    values = values / values[0]
    betas = 1.0 - values[1:] / values[:-1]
    betas = betas.clamp(1e-5, 0.999)
    return torch.cumprod(1.0 - betas, dim=0).float()


class ConditionedDilatedBlock(nn.Module):
    def __init__(self, channels: int, condition_dim: int, dilation: int) -> None:
        super().__init__()
        self.normalization = nn.GroupNorm(8, channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=int(dilation),
            dilation=int(dilation),
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, 2 * channels, kernel_size=1)
        self.condition = nn.Linear(condition_dim, 2 * channels)
        self.output = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, values: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.depthwise(F.silu(self.normalization(values)))
        hidden = self.pointwise(hidden)
        hidden = hidden + self.condition(condition).unsqueeze(-1)
        gate, candidate = hidden.chunk(2, dim=1)
        hidden = torch.sigmoid(gate) * torch.tanh(candidate)
        return values + self.output(hidden) / math.sqrt(2.0)


class ModalityResidualDenoiser(nn.Module):
    def __init__(
        self,
        input_channels: int,
        channels: int,
        context_dim: int,
        blocks: int,
    ) -> None:
        super().__init__()
        self.step_dim = channels
        condition_dim = 2 * channels
        self.input = nn.Conv1d(input_channels, channels, kernel_size=7, padding=3)
        self.step_projection = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        self.context_projection = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        dilations = [2 ** (index % 6) for index in range(int(blocks))]
        self.blocks = nn.ModuleList(
            ConditionedDilatedBlock(channels, condition_dim, dilation)
            for dilation in dilations
        )
        self.output = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, 1, kernel_size=7, padding=3),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        inputs: torch.Tensor,
        steps: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        step = self.step_projection(_sinusoidal_embedding(steps, self.step_dim))
        global_condition = torch.cat(
            (step, self.context_projection(context)), dim=-1
        )
        hidden = self.input(inputs)
        for block in self.blocks:
            hidden = block(hidden, global_condition)
        return self.output(hidden).squeeze(1)


class StructuralResidualDiffusion(nn.Module):
    """Generate stochastic waveform residuals around a frozen mainline forecast."""

    def __init__(
        self,
        modalities: Sequence[str],
        shared_state_dim: int,
        dynamics_state_dim: int,
        diffusion_steps: int = 50,
        channels: int = 64,
        blocks: int = 8,
        residual_clip: float = 6.0,
        structural_condition_channels: int = 2,
    ) -> None:
        super().__init__()
        self.modalities = tuple(modalities)
        self.diffusion_steps = int(diffusion_steps)
        self.residual_clip = float(residual_clip)
        self.structural_condition_channels = int(structural_condition_channels)
        if self.structural_condition_channels < 2:
            raise ValueError("diffusion requires at least two structural condition channels")
        if self.diffusion_steps < 2:
            raise ValueError("diffusion requires at least two steps")
        self.register_buffer(
            "alpha_bar", _cosine_alpha_bar(self.diffusion_steps), persistent=True
        )
        context_dim = int(shared_state_dim) + int(dynamics_state_dim)
        self.denoisers = nn.ModuleDict(
            {
                modality: ModalityResidualDenoiser(
                    input_channels=3 + self.structural_condition_channels,
                    channels=int(channels),
                    context_dim=context_dim,
                    blocks=int(blocks),
                )
                for modality in self.modalities
            }
        )

    @staticmethod
    def residual_scale(recent: torch.Tensor) -> torch.Tensor:
        return recent.square().mean(dim=-1, keepdim=True).sqrt().clamp(0.05, 4.0)

    def _predict_residual(
        self,
        noisy: torch.Tensor,
        steps: torch.Tensor,
        base: torch.Tensor,
        recent: torch.Tensor,
        structural_condition: torch.Tensor,
        shared_state: torch.Tensor,
        dynamics_state: torch.Tensor,
        active_modalities: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:
        if structural_condition.shape != (
            base.shape[0],
            base.shape[1],
            self.structural_condition_channels,
            base.shape[-1],
        ):
            raise ValueError(
                "structural condition channel count does not match the denoiser"
            )
        scale = self.residual_scale(recent)
        active = set(active_modalities or self.modalities)
        predictions = []
        for index, modality in enumerate(self.modalities):
            if modality not in active:
                predictions.append(torch.zeros_like(noisy[:, index]))
                continue
            inputs = torch.cat(
                (
                    noisy[:, index : index + 1],
                    base[:, index : index + 1] / scale[:, index : index + 1],
                    recent[:, index : index + 1] / scale[:, index : index + 1],
                    structural_condition[:, index],
                ),
                dim=1,
            )
            context = torch.cat(
                (shared_state, dynamics_state[:, index]), dim=-1
            )
            predictions.append(self.denoisers[modality](inputs, steps, context))
        return torch.stack(predictions, dim=1).clamp(
            -self.residual_clip, self.residual_clip
        )

    def training_prediction(
        self,
        base: torch.Tensor,
        recent: torch.Tensor,
        target: torch.Tensor,
        structural_condition: torch.Tensor,
        shared_state: torch.Tensor,
        dynamics_state: torch.Tensor,
        steps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        active_modalities: Optional[Sequence[str]] = None,
        residual_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = base.shape[0]
        if steps is None:
            steps = torch.randint(
                self.diffusion_steps, (batch_size,), device=base.device
            )
        if noise is None:
            noise = torch.randn_like(base)
        if residual_mask is None:
            residual_mask = torch.ones_like(base)
        if residual_mask.shape != base.shape:
            raise ValueError("residual mask must match the waveform shape")
        residual_mask = residual_mask.to(device=base.device, dtype=base.dtype)
        scale = self.residual_scale(recent)
        target_residual = ((target - base) / scale).clamp(
            -self.residual_clip, self.residual_clip
        ) * residual_mask
        noise = noise * residual_mask
        alpha = self.alpha_bar[steps].view(-1, 1, 1).to(base.dtype)
        noisy = alpha.sqrt() * target_residual + (1.0 - alpha).sqrt() * noise
        predicted_residual = self._predict_residual(
            noisy,
            steps,
            base,
            recent,
            structural_condition,
            shared_state,
            dynamics_state,
            active_modalities,
        ) * residual_mask
        return {
            "predicted_residual": predicted_residual,
            "target_residual": target_residual,
            "waveform": base + scale * predicted_residual,
            "steps": steps,
        }

    @torch.no_grad()
    def sample(
        self,
        base: torch.Tensor,
        recent: torch.Tensor,
        structural_condition: torch.Tensor,
        shared_state: torch.Tensor,
        dynamics_state: torch.Tensor,
        sampling_steps: int = 12,
        generator: Optional[torch.Generator] = None,
        active_modalities: Optional[Sequence[str]] = None,
        residual_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if residual_mask is None:
            residual_mask = torch.ones_like(base)
        if residual_mask.shape != base.shape:
            raise ValueError("residual mask must match the waveform shape")
        residual_mask = residual_mask.to(device=base.device, dtype=base.dtype)
        count = max(2, min(int(sampling_steps), self.diffusion_steps))
        schedule = torch.linspace(
            self.diffusion_steps - 1, 0, count, device=base.device
        ).round().long().unique_consecutive()
        current = torch.randn(
            base.shape,
            device=base.device,
            dtype=base.dtype,
            generator=generator,
        ) * residual_mask
        predicted = current
        for schedule_index, step_value in enumerate(schedule):
            steps = torch.full(
                (base.shape[0],), int(step_value), device=base.device, dtype=torch.long
            )
            predicted = self._predict_residual(
                current,
                steps,
                base,
                recent,
                structural_condition,
                shared_state,
                dynamics_state,
                active_modalities,
            ) * residual_mask
            if schedule_index + 1 == len(schedule):
                current = predicted
                continue
            next_step = schedule[schedule_index + 1]
            alpha = self.alpha_bar[step_value].to(base.dtype)
            next_alpha = self.alpha_bar[next_step].to(base.dtype)
            estimated_noise = (
                current - alpha.sqrt() * predicted
            ) / (1.0 - alpha).sqrt().clamp_min(1e-6)
            current = next_alpha.sqrt() * predicted + (
                1.0 - next_alpha
            ).sqrt() * estimated_noise
        return base + self.residual_scale(recent) * current


def _masked_group_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    losses = []
    for modality_index in range(values.shape[1]):
        selected = valid[:, modality_index]
        if selected.any():
            losses.append(values[selected, modality_index].mean())
    if not losses:
        raise ValueError("diffusion loss has no valid modalities")
    return torch.stack(losses).mean()


def _envelope(values: torch.Tensor, kernel: int, derivative: bool) -> torch.Tensor:
    signal = values[..., 1:] - values[..., :-1] if derivative else values
    signal = signal.square() if derivative else signal.abs()
    kernel = max(1, min(int(kernel), signal.shape[-1]))
    return F.avg_pool1d(
        signal.reshape(-1, 1, signal.shape[-1]),
        kernel_size=kernel,
        stride=1,
        padding=kernel // 2,
    )[..., : signal.shape[-1]].reshape_as(signal)


def _spectral_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    fft_sizes: Sequence[int],
) -> torch.Tensor:
    group_losses = []
    for modality_index in range(prediction.shape[1]):
        selected = valid[:, modality_index]
        if not selected.any():
            continue
        modality_losses = []
        predicted = prediction[selected, modality_index]
        truth = target[selected, modality_index]
        for fft_size in fft_sizes:
            size = min(int(fft_size), prediction.shape[-1])
            if size < 4:
                continue
            window = torch.hann_window(
                size, device=prediction.device, dtype=prediction.dtype
            )
            hop = max(1, size // 4)
            predicted_stft = torch.stft(
                predicted,
                n_fft=size,
                hop_length=hop,
                win_length=size,
                window=window,
                return_complex=True,
                normalized=True,
            ).abs()
            target_stft = torch.stft(
                truth,
                n_fft=size,
                hop_length=hop,
                win_length=size,
                window=window,
                return_complex=True,
                normalized=True,
            ).abs()
            modality_losses.append(
                F.l1_loss(torch.log1p(predicted_stft), torch.log1p(target_stft))
            )
        group_losses.append(torch.stack(modality_losses).mean())
    return torch.stack(group_losses).mean()


def _eeg_autocorrelation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int,
) -> torch.Tensor:
    """Match phase-invariant EEG temporal structure over physiological lags."""
    maximum_lag = min(prediction.shape[-1] - 1, int(sample_rate))
    lags = [lag for lag in (1, 2, 4, 8, 16, 32, 64, 128) if lag <= maximum_lag]
    predicted = prediction - prediction.mean(dim=-1, keepdim=True)
    truth = target - target.mean(dim=-1, keepdim=True)
    predicted = predicted / predicted.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    truth = truth / truth.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    predicted_correlations = torch.stack(
        [(predicted[:, :-lag] * predicted[:, lag:]).mean(dim=-1) for lag in lags],
        dim=-1,
    )
    target_correlations = torch.stack(
        [(truth[:, :-lag] * truth[:, lag:]).mean(dim=-1) for lag in lags],
        dim=-1,
    )
    return F.smooth_l1_loss(predicted_correlations, target_correlations)


def _emg_burst_distribution_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Match multiscale EMG energy and peak distributions without fixing burst phase."""
    losses = []
    for kernel in (8, 16, 32, 64):
        if kernel > prediction.shape[-1]:
            continue
        stride = max(1, kernel // 4)
        predicted_energy = F.avg_pool1d(
            prediction.square().unsqueeze(1), kernel, stride=stride
        ).squeeze(1).clamp_min(1e-8).sqrt()
        target_energy = F.avg_pool1d(
            target.square().unsqueeze(1), kernel, stride=stride
        ).squeeze(1).clamp_min(1e-8).sqrt()
        losses.append(
            F.smooth_l1_loss(
                torch.log1p(predicted_energy.sort(dim=-1).values),
                torch.log1p(target_energy.sort(dim=-1).values),
            )
        )
    predicted_derivative = (prediction[:, 1:] - prediction[:, :-1]).abs()
    target_derivative = (target[:, 1:] - target[:, :-1]).abs()
    losses.append(
        F.smooth_l1_loss(
            torch.log1p(predicted_derivative.sort(dim=-1).values),
            torch.log1p(target_derivative.sort(dim=-1).values),
        )
    )
    return torch.stack(losses).mean()


def event_centered_beat_morphology_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    prediction_events: torch.Tensor,
    target_events: torch.Tensor,
    sample_rate: int,
) -> torch.Tensor:
    """Match median ECG beat templates while allowing event times to differ."""
    if prediction.ndim != 2 or target.shape != prediction.shape:
        raise ValueError("ECG prediction and target must be [batch, samples]")
    if prediction_events.shape != prediction.shape or target_events.shape != target.shape:
        raise ValueError("ECG event masks must match their waveform shapes")
    left = max(1, round(0.20 * sample_rate))
    right = max(1, round(0.40 * sample_rate))
    losses = []
    for predicted_row, target_row, predicted_mask, target_mask in zip(
        prediction, target, prediction_events, target_events
    ):
        predicted_beats = []
        for position in predicted_mask.nonzero(as_tuple=False).flatten():
            start = int(position) - left
            stop = int(position) + right
            if start >= 0 and stop <= predicted_row.shape[-1]:
                segment = predicted_row[start:stop]
                predicted_beats.append(segment - 0.5 * (segment[0] + segment[-1]))
        target_beats = []
        for position in target_mask.nonzero(as_tuple=False).flatten():
            start = int(position) - left
            stop = int(position) + right
            if start >= 0 and stop <= target_row.shape[-1]:
                segment = target_row[start:stop]
                target_beats.append(segment - 0.5 * (segment[0] + segment[-1]))
        if not predicted_beats or not target_beats:
            continue
        predicted_template = torch.stack(predicted_beats).median(dim=0).values
        target_template = torch.stack(target_beats).median(dim=0).values
        predicted_template = predicted_template - predicted_template.mean()
        target_template = target_template - target_template.mean()
        correlation = (predicted_template * target_template).sum() / (
            predicted_template.square().sum().sqrt()
            * target_template.square().sum().sqrt()
        ).clamp_min(1e-6)
        losses.append(1.0 - correlation.clamp(-1.0, 1.0))
    if not losses:
        return prediction.sum() * 0.0
    return torch.stack(losses).mean()


@torch.no_grad()
def project_event_centered_beat_morphology(
    candidate: torch.Tensor,
    base: torch.Tensor,
    events: torch.Tensor,
    sample_rate: int,
) -> torch.Tensor:
    """Keep generated beat amplitude/baseline changes without shape distortion."""
    if candidate.ndim != 2 or base.shape != candidate.shape:
        raise ValueError("ECG candidate and base must be [batch, samples]")
    if events.shape != candidate.shape:
        raise ValueError("ECG event mask must match the waveform shape")
    left = max(1, round(0.20 * sample_rate))
    right = max(1, round(0.40 * sample_rate))
    projected = candidate.clone()
    accumulated = torch.zeros_like(candidate)
    counts = torch.zeros_like(candidate)
    for row_index, positions in enumerate(events):
        for position in positions.nonzero(as_tuple=False).flatten():
            start = int(position) - left
            stop = int(position) + right
            if start < 0 or stop > candidate.shape[-1]:
                continue
            base_segment = base[row_index, start:stop]
            candidate_segment = candidate[row_index, start:stop]
            centered_base = base_segment - base_segment.mean()
            centered_candidate = candidate_segment - candidate_segment.mean()
            scale = (
                (centered_base * centered_candidate).sum()
                / centered_base.square().sum().clamp_min(1e-6)
            ).clamp(0.25, 4.0)
            offset = candidate_segment.mean() - scale * base_segment.mean()
            accumulated[row_index, start:stop] += scale * base_segment + offset
            counts[row_index, start:stop] += 1.0
    covered = counts > 0
    projected[covered] = accumulated[covered] / counts[covered]
    return projected


@torch.no_grad()
def causal_ar_phase_anchor(
    candidate: torch.Tensor,
    recent: torch.Tensor,
    sample_rate: int,
    history_seconds: float = 2.0,
    anchor_seconds: float = 0.5,
    fade_seconds: float = 0.5,
    order: int = 24,
) -> torch.Tensor:
    """Anchor only the short-term EEG phase with a causal ridge-AR forecast."""
    if candidate.ndim != 2 or recent.ndim != 2 or len(candidate) != len(recent):
        raise ValueError("EEG candidate and recent signals must be batched waveforms")
    history_samples = min(recent.shape[-1], round(history_seconds * sample_rate))
    forecast_samples = min(
        candidate.shape[-1], round((anchor_seconds + fade_seconds) * sample_rate)
    )
    order = max(4, min(int(order), history_samples // 4))
    output = candidate.clone()
    history = recent[:, -history_samples:].float()
    means = history.mean(dim=-1, keepdim=True)
    centered = history - means
    windows = centered.unfold(1, order + 1, 1)
    predictors = windows[..., :-1].flip(-1)
    targets = windows[..., -1]
    gram = predictors.transpose(1, 2) @ predictors
    ridge = 0.01 * gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-6)
    coefficients = torch.linalg.solve(
        gram + ridge[:, None, None] * torch.eye(order, device=gram.device),
        (predictors.transpose(1, 2) @ targets.unsqueeze(-1)).squeeze(-1),
    )
    state = centered[:, -order:].flip(-1)
    forecast_values = []
    for _ in range(forecast_samples):
        value = (state * coefficients).sum(dim=-1)
        forecast_values.append(value)
        state = torch.cat((value.unsqueeze(-1), state[:, :-1]), dim=-1)
    forecast = torch.stack(forecast_values, dim=-1)
    candidate_rms = candidate[:, :forecast_samples].square().mean(dim=-1, keepdim=True).sqrt()
    forecast_rms = forecast.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    forecast = forecast * candidate_rms / forecast_rms
    forecast = torch.maximum(
        torch.minimum(forecast, 4.0 * candidate_rms), -4.0 * candidate_rms
    ) + means
    anchor_samples = min(forecast_samples, round(anchor_seconds * sample_rate))
    weights = torch.ones(forecast_samples, device=candidate.device)
    if forecast_samples > anchor_samples:
        weights[anchor_samples:] = torch.linspace(
            1.0, 0.0, forecast_samples - anchor_samples, device=candidate.device
        )
    output[:, :forecast_samples] = (
        weights.unsqueeze(0) * forecast
        + (1.0 - weights).unsqueeze(0) * candidate[:, :forecast_samples]
    )
    return output.to(candidate.dtype)


@torch.no_grad()
def align_ecg_to_causal_rr_phase(
    candidate: torch.Tensor,
    predicted_events: torch.Tensor,
    recent_events: torch.Tensor,
    predicted_rr_seconds: torch.Tensor,
    sample_rate: int,
    maximum_shift_seconds: float = 0.25,
) -> torch.Tensor:
    """Align the first future beat to recent QRS phase and the predicted RR."""
    if candidate.shape != predicted_events.shape:
        raise ValueError("predicted ECG events must match the candidate waveform")
    output = candidate.clone()
    maximum_shift = round(maximum_shift_seconds * sample_rate)
    positions = torch.arange(candidate.shape[-1], device=candidate.device).float()
    for row_index in range(len(candidate)):
        recent_positions = recent_events[row_index].nonzero(as_tuple=False).flatten()
        predicted_positions = predicted_events[row_index].nonzero(as_tuple=False).flatten()
        if len(recent_positions) < 3 or not len(predicted_positions):
            continue
        intervals = recent_positions[-6:].diff().float()
        if not len(intervals):
            continue
        recent_rr = intervals.median()
        rr_prediction = predicted_rr_seconds[row_index, 0].float() * sample_rate
        rr_samples = (0.5 * recent_rr + 0.5 * rr_prediction).clamp(
            0.35 * sample_rate, 2.0 * sample_rate
        )
        elapsed = recent_events.shape[-1] - recent_positions[-1].float()
        expected_first = rr_samples - elapsed
        while expected_first < 0:
            expected_first += rr_samples
        variability = intervals.std(unbiased=False) / intervals.mean().clamp_min(1.0)
        reliability = (1.0 - 3.0 * variability).clamp(0.0, 1.0)
        shift = ((predicted_positions[0].float() - expected_first) * reliability).clamp(
            -maximum_shift, maximum_shift
        )
        source = (positions + shift).clamp(0, candidate.shape[-1] - 1)
        lower = source.floor().long()
        upper = (lower + 1).clamp_max(candidate.shape[-1] - 1)
        fraction = source - lower.float()
        output[row_index] = (
            (1.0 - fraction) * candidate[row_index, lower]
            + fraction * candidate[row_index, upper]
        )
    return output


@torch.no_grad()
def align_ecg_events_to_probability_mode(
    candidate: torch.Tensor,
    events: torch.Tensor,
    qrs_logits: torch.Tensor,
    sample_rate: int,
    search_seconds: float = 0.12,
) -> torch.Tensor:
    """Warp predicted ECG beats from soft centers to local QRS probability modes."""
    if candidate.shape != events.shape or qrs_logits.shape != candidate.shape:
        raise ValueError("ECG waveform, events, and QRS logits must have matching shapes")
    output = candidate.clone()
    radius = max(1, round(float(search_seconds) * sample_rate))
    sample_positions = torch.arange(candidate.shape[-1], device=candidate.device).float()
    for row_index in range(len(candidate)):
        source_events = events[row_index].nonzero(as_tuple=False).flatten()
        if not len(source_events):
            continue
        desired_events = []
        for position in source_events:
            start = max(0, int(position) - radius)
            stop = min(candidate.shape[-1], int(position) + radius + 1)
            desired_events.append(
                qrs_logits[row_index, start:stop].argmax() + start
            )
        desired_events = torch.stack(desired_events).float()
        if len(desired_events) > 1 and not torch.all(desired_events[1:] > desired_events[:-1]):
            continue
        desired_anchors = torch.cat(
            (
                desired_events.new_tensor([0.0]),
                desired_events,
                desired_events.new_tensor([candidate.shape[-1] - 1.0]),
            )
        )
        source_anchors = torch.cat(
            (
                desired_events.new_tensor([0.0]),
                source_events.float(),
                desired_events.new_tensor([candidate.shape[-1] - 1.0]),
            )
        )
        if not torch.all(desired_anchors[1:] > desired_anchors[:-1]):
            continue
        interval = torch.searchsorted(
            desired_anchors, sample_positions, right=True
        ).sub(1).clamp(0, len(desired_anchors) - 2)
        desired_start = desired_anchors[interval]
        desired_stop = desired_anchors[interval + 1]
        source_start = source_anchors[interval]
        source_stop = source_anchors[interval + 1]
        fraction = (sample_positions - desired_start) / (
            desired_stop - desired_start
        ).clamp_min(1e-6)
        source = source_start + fraction * (source_stop - source_start)
        lower = source.floor().long().clamp(0, candidate.shape[-1] - 1)
        upper = (lower + 1).clamp_max(candidate.shape[-1] - 1)
        fraction = source - lower.float()
        output[row_index] = (
            (1.0 - fraction) * candidate[row_index, lower]
            + fraction * candidate[row_index, upper]
        )
    return output


@torch.no_grad()
def warp_ecg_to_event_sequence(
    waveform: torch.Tensor,
    source_events: torch.Tensor,
    desired_events: torch.Tensor,
    sample_rate: int = 128,
) -> torch.Tensor:
    """Move ECG events while preserving each local beat morphology."""
    if waveform.shape != source_events.shape or waveform.shape != desired_events.shape:
        raise ValueError("ECG waveform and event sequences must have matching shapes")
    output = waveform.clone()
    samples = waveform.shape[-1]
    positions = torch.arange(samples, device=waveform.device).float()
    for row_index in range(len(waveform)):
        source = source_events[row_index].nonzero(as_tuple=False).flatten().float()
        desired = desired_events[row_index].nonzero(as_tuple=False).flatten().float()
        count = min(len(source), len(desired))
        if count == 0:
            continue
        if len(source) != count:
            source = source[
                torch.linspace(0, len(source) - 1, count, device=waveform.device).round().long()
            ]
        if len(desired) != count:
            desired = desired[
                torch.linspace(0, len(desired) - 1, count, device=waveform.device).round().long()
            ]
        desired_anchor_values = [0.0]
        source_anchor_values = [0.0]
        for event_index, (source_position, desired_position) in enumerate(
            zip(source.tolist(), desired.tolist())
        ):
            source_previous = source_position - (
                source[event_index - 1].item() if event_index else 0.0
            )
            desired_previous = desired_position - (
                desired[event_index - 1].item() if event_index else 0.0
            )
            source_next = (
                source[event_index + 1].item()
                if event_index + 1 < count
                else samples - 1.0
            ) - source_position
            desired_next = (
                desired[event_index + 1].item()
                if event_index + 1 < count
                else samples - 1.0
            ) - desired_position
            left = min(
                round(0.20 * float(sample_rate)),
                int(0.48 * min(source_previous, desired_previous)),
            )
            right = min(
                round(0.40 * float(sample_rate)),
                int(0.48 * min(source_next, desired_next)),
            )
            candidates = (
                (desired_position - left, source_position - left),
                (desired_position, source_position),
                (desired_position + right, source_position + right),
            )
            for desired_anchor, source_anchor in candidates:
                if (
                    desired_anchor > desired_anchor_values[-1]
                    and source_anchor > source_anchor_values[-1]
                    and desired_anchor < samples - 1.0
                    and source_anchor < samples - 1.0
                ):
                    desired_anchor_values.append(desired_anchor)
                    source_anchor_values.append(source_anchor)
        if (
            samples - 1.0 <= desired_anchor_values[-1]
            or samples - 1.0 <= source_anchor_values[-1]
        ):
            continue
        desired_anchor_values.append(samples - 1.0)
        source_anchor_values.append(samples - 1.0)
        desired_anchors = waveform.new_tensor(desired_anchor_values)
        source_anchors = waveform.new_tensor(source_anchor_values)
        interval = torch.searchsorted(desired_anchors, positions, right=True)
        interval = interval.sub(1).clamp(0, len(desired_anchors) - 2)
        desired_start = desired_anchors[interval]
        desired_stop = desired_anchors[interval + 1]
        source_start = source_anchors[interval]
        source_stop = source_anchors[interval + 1]
        fraction = (positions - desired_start) / (
            desired_stop - desired_start
        ).clamp_min(1e-6)
        mapped = source_start + fraction * (source_stop - source_start)
        lower = mapped.floor().long().clamp(0, samples - 1)
        upper = (lower + 1).clamp_max(samples - 1)
        fraction = mapped - lower.float()
        output[row_index] = (
            (1.0 - fraction) * waveform[row_index, lower]
            + fraction * waveform[row_index, upper]
        )
        morphology_left = round(0.20 * float(sample_rate))
        morphology_right = round(0.40 * float(sample_rate))
        for source_position, desired_position in zip(source.long(), desired.long()):
            source_index = int(source_position)
            desired_index = int(desired_position)
            left = min(morphology_left, source_index, desired_index)
            right = min(
                morphology_right,
                samples - source_index,
                samples - desired_index,
            )
            if left + right < 2:
                continue
            output[
                row_index, desired_index - left : desired_index + right
            ] = waveform[
                row_index, source_index - left : source_index + right
            ]
    return output


@torch.no_grad()
def project_emg_rms_and_peaks(
    candidate: torch.Tensor,
    target_rms: torch.Tensor,
    patch_samples: int,
    peak_exponent: float = 1.25,
    maximum_gain: float = 3.0,
) -> torch.Tensor:
    """Restore calibrated EMG patch RMS and burst peak contrast."""
    if candidate.ndim != 2 or target_rms.shape != candidate.shape:
        raise ValueError("EMG waveform and RMS condition must have matching shapes")
    patch_count = candidate.shape[-1] // int(patch_samples)
    usable = patch_count * int(patch_samples)
    patches = candidate[:, :usable].reshape(-1, patch_count, int(patch_samples))
    desired = target_rms[:, :usable].reshape(-1, patch_count, int(patch_samples)).mean(-1)
    means = patches.mean(dim=-1, keepdim=True)
    centered = patches - means
    scale_reference = centered.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    normalized = centered / scale_reference
    shaped = normalized.sign() * normalized.abs().pow(float(peak_exponent))
    shaped_rms = shaped.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    requested_gain = (desired.unsqueeze(-1) / scale_reference).clamp(
        1.0 / float(maximum_gain), float(maximum_gain)
    )
    restored = means + shaped / shaped_rms * scale_reference * requested_gain
    output = candidate.clone()
    output[:, :usable] = restored.reshape(len(candidate), usable)
    return output


@torch.no_grad()
def project_emg_probability_bursts(
    candidate: torch.Tensor,
    target_rms: torch.Tensor,
    burst_probability: torch.Tensor,
    patch_samples: int,
    decision_threshold: float,
    redistribution_strength: float = 0.10,
    burst_dense_exponent: float = 0.75,
    nonburst_sparse_exponent: float = 1.25,
    hard_event_lock: bool = False,
) -> torch.Tensor:
    """Redistribute EMG patch energy toward calibrated burst events."""
    if candidate.ndim != 2 or target_rms.shape != candidate.shape:
        raise ValueError("EMG waveform and RMS condition must have matching shapes")
    patch_count = candidate.shape[-1] // int(patch_samples)
    if burst_probability.shape != (candidate.shape[0], patch_count):
        raise ValueError("burst probability must be [batch, patches]")
    usable = patch_count * int(patch_samples)
    patches = candidate[:, :usable].reshape(-1, patch_count, int(patch_samples))
    desired = target_rms[:, :usable].reshape(
        -1, patch_count, int(patch_samples)
    ).mean(dim=-1)
    centered_probability = burst_probability - burst_probability.mean(dim=-1, keepdim=True)
    if hard_event_lock:
        event = burst_probability >= float(decision_threshold)
        event_fraction = event.float().mean(dim=-1, keepdim=True)
        excessive = event_fraction >= 0.5
        if excessive.any():
            keep = max(1, int(0.3 * patch_count))
            top = burst_probability.topk(keep, dim=-1).indices
            sparse_event = torch.zeros_like(event).scatter(1, top, True)
            event = torch.where(excessive, sparse_event, event)
            event_fraction = event.float().mean(dim=-1, keepdim=True)
        strength = float(redistribution_strength)
        non_event_factor = (
            1.0 - strength * event_fraction / (1.0 - event_fraction).clamp_min(1e-3)
        ).clamp_min(0.35)
        factors = torch.where(
            event,
            torch.full_like(burst_probability, 1.0 + strength),
            non_event_factor.expand_as(burst_probability),
        )
        no_event = event_fraction <= 0.0
        factors = torch.where(no_event, torch.ones_like(factors), factors)
    else:
        factors = (
            1.0 + float(redistribution_strength) * centered_probability
        ).clamp(0.75, 1.25)
    adjusted = desired * factors
    energy_scale = (
        desired.square().mean(dim=-1, keepdim=True)
        / adjusted.square().mean(dim=-1, keepdim=True).clamp_min(1e-8)
    ).sqrt()
    adjusted = adjusted * energy_scale

    means = patches.mean(dim=-1, keepdim=True)
    centered = patches - means
    reference_rms = centered.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    normalized = centered / reference_rms
    activation = torch.sigmoid(
        (burst_probability - float(decision_threshold)) / 0.10
    ).unsqueeze(-1)
    exponent = float(nonburst_sparse_exponent) + activation * (
        float(burst_dense_exponent) - float(nonburst_sparse_exponent)
    )
    shaped = normalized.sign() * normalized.abs().clamp_min(1e-6).pow(exponent)
    shaped_rms = shaped.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    restored = means + shaped / shaped_rms * adjusted.unsqueeze(-1)
    output = candidate.clone()
    output[:, :usable] = restored.reshape(len(candidate), usable)
    return output


def structural_diffusion_loss(
    output: Dict[str, torch.Tensor],
    target: torch.Tensor,
    valid: torch.Tensor,
    sample_rate: int,
    patch_samples: int,
    fft_sizes: Sequence[int],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    weights = dict(weights or {})
    prediction = output["waveform"]
    residual_error = (output["predicted_residual"] - output["target_residual"]).abs()
    diffusion_loss = _masked_group_mean(residual_error.mean(dim=-1), valid)
    point_loss = _masked_group_mean((prediction - target).abs().mean(dim=-1), valid)
    spectral_loss = _spectral_loss(prediction, target, valid, fft_sizes)

    structure_losses = []
    amplitude_losses = []
    invariant_losses = []
    invariant_weight = float(weights.get("invariant", 0.0))
    patch_count = prediction.shape[-1] // int(patch_samples)
    for modality_index in range(prediction.shape[1]):
        selected = valid[:, modality_index]
        if not selected.any():
            continue
        predicted = prediction[selected, modality_index]
        truth = target[selected, modality_index]
        predicted_patches = predicted.reshape(-1, patch_count, patch_samples)
        target_patches = truth.reshape(-1, patch_count, patch_samples)
        predicted_rms = predicted_patches.square().mean(dim=-1).clamp_min(1e-8).sqrt()
        target_rms = target_patches.square().mean(dim=-1).clamp_min(1e-8).sqrt()
        amplitude_losses.append(
            F.l1_loss(torch.log1p(predicted_rms), torch.log1p(target_rms))
        )
        if modality_index == 0:
            predicted_spectrum = torch.fft.rfft(
                predicted_patches - predicted_patches.mean(dim=-1, keepdim=True),
                dim=-1,
                norm="ortho",
            ).abs()
            target_spectrum = torch.fft.rfft(
                target_patches - target_patches.mean(dim=-1, keepdim=True),
                dim=-1,
                norm="ortho",
            ).abs()
            structure_losses.append(
                F.l1_loss(torch.log1p(predicted_spectrum), torch.log1p(target_spectrum))
            )
            if invariant_weight > 0.0:
                invariant_losses.append(
                    _eeg_autocorrelation_loss(predicted, truth, sample_rate)
                )
        elif modality_index == 1:
            structure_losses.append(
                F.smooth_l1_loss(
                    _envelope(predicted, round(0.12 * sample_rate), True),
                    _envelope(truth, round(0.12 * sample_rate), True),
                )
            )
        else:
            structure_losses.append(
                F.smooth_l1_loss(
                    _envelope(predicted, round(0.25 * sample_rate), False),
                    _envelope(truth, round(0.25 * sample_rate), False),
                )
            )
            if invariant_weight > 0.0:
                invariant_losses.append(
                    _emg_burst_distribution_loss(predicted, truth)
                )
    structure_loss = torch.stack(structure_losses).mean()
    amplitude_loss = torch.stack(amplitude_losses).mean()
    invariant_loss = (
        torch.stack(invariant_losses).mean()
        if invariant_losses
        else prediction.new_zeros(())
    )
    total = (
        weights.get("diffusion", 1.0) * diffusion_loss
        + weights.get("point", 0.25) * point_loss
        + weights.get("spectral", 0.5) * spectral_loss
        + weights.get("structure", 0.5) * structure_loss
        + weights.get("amplitude", 0.25) * amplitude_loss
        + invariant_weight * invariant_loss
    )
    return {
        "loss": total,
        "diffusion_loss": diffusion_loss,
        "point_loss": point_loss,
        "spectral_loss": spectral_loss,
        "structure_loss": structure_loss,
        "amplitude_loss": amplitude_loss,
        "invariant_loss": invariant_loss,
    }
