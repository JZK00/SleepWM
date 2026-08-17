from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _zero_linear(layer: nn.Linear) -> nn.Linear:
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class BeliefConditionedWaveformAdapter(nn.Module):
    """Project3-only bounded EEG/EMG correction around a frozen waveform decoder."""

    def __init__(
        self,
        modalities: Sequence[str],
        context_dim: int,
        patch_samples: int,
        num_patches: int,
        hidden_dim: int = 192,
        maximum_recent_blend: float = 0.5,
        maximum_eeg_log_gain: float = 0.75,
        maximum_emg_log_gain: float = 0.8,
        maximum_residual_ratio: float = 0.2,
    ) -> None:
        super().__init__()
        self.modalities = tuple(modalities)
        self.patch_samples = int(patch_samples)
        self.num_patches = int(num_patches)
        self.maximum_recent_blend = float(maximum_recent_blend)
        self.maximum_eeg_log_gain = float(maximum_eeg_log_gain)
        self.maximum_emg_log_gain = float(maximum_emg_log_gain)
        self.maximum_residual_ratio = float(maximum_residual_ratio)
        if "EEG" not in self.modalities or "EMG" not in self.modalities:
            raise ValueError("the waveform adapter requires EEG and EMG")
        if min(self.patch_samples, self.num_patches, hidden_dim) < 1:
            raise ValueError("waveform adapter dimensions must be positive")

        self.context_encoder = nn.Sequential(
            nn.LayerNorm(int(context_dim)),
            nn.Linear(int(context_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
        )
        spectrum_bins = self.patch_samples // 2 + 1
        self.eeg_recent_blend = _zero_linear(
            nn.Linear(hidden_dim, self.num_patches)
        )
        self.eeg_log_spectral_gain = _zero_linear(
            nn.Linear(hidden_dim, self.num_patches * spectrum_bins)
        )
        self.eeg_mean_shift = _zero_linear(
            nn.Linear(hidden_dim, self.num_patches)
        )
        self.eeg_residual = _zero_linear(
            nn.Linear(hidden_dim, self.num_patches * self.patch_samples)
        )
        self.emg_recent_blend = _zero_linear(
            nn.Linear(hidden_dim, self.num_patches)
        )
        self.emg_log_scale = _zero_linear(
            nn.Linear(hidden_dim, self.num_patches)
        )
        self.emg_burst_log_scale = _zero_linear(
            nn.Linear(hidden_dim, self.num_patches)
        )
        self.emg_mean_shift = _zero_linear(
            nn.Linear(hidden_dim, self.num_patches)
        )
        self.emg_residual = _zero_linear(
            nn.Linear(hidden_dim, self.num_patches * self.patch_samples)
        )

    @property
    def maximum_samples(self) -> int:
        return self.patch_samples * self.num_patches

    def _patches(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform[..., : self.maximum_samples].reshape(
            waveform.shape[0], self.num_patches, self.patch_samples
        )

    def forward(
        self,
        frozen_waveforms: torch.Tensor,
        last_observed_waveforms: torch.Tensor,
        context: torch.Tensor,
        eeg_correction_scale: float = 1.0,
        emg_correction_scale: float = 1.0,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if frozen_waveforms.shape != last_observed_waveforms.shape:
            raise ValueError("frozen and last-observed waveforms must have equal shapes")
        if frozen_waveforms.ndim != 3:
            raise ValueError("waveforms must be [batch, modalities, samples]")
        if frozen_waveforms.shape[1] != len(self.modalities):
            raise ValueError("waveform modality count is invalid")
        if frozen_waveforms.shape[-1] < self.maximum_samples:
            raise ValueError("waveforms are shorter than the configured adapter horizon")

        encoded = self.context_encoder(context)
        output = frozen_waveforms[..., : self.maximum_samples].clone()
        auxiliary: Dict[str, torch.Tensor] = {}

        eeg_index = self.modalities.index("EEG")
        eeg_base = self._patches(frozen_waveforms[:, eeg_index])
        eeg_recent = self._patches(last_observed_waveforms[:, eeg_index])
        eeg_blend = self.maximum_recent_blend * torch.tanh(
            self.eeg_recent_blend(encoded)
        ).unsqueeze(-1)
        eeg_source = eeg_base + eeg_blend * (eeg_recent - eeg_base)
        eeg_mean = eeg_source.mean(dim=-1, keepdim=True)
        eeg_centered = eeg_source - eeg_mean
        eeg_spectrum = torch.fft.rfft(eeg_centered, dim=-1, norm="ortho")
        spectrum_bins = eeg_spectrum.shape[-1]
        eeg_log_gain = self.maximum_eeg_log_gain * torch.tanh(
            self.eeg_log_spectral_gain(encoded).reshape(
                -1, self.num_patches, spectrum_bins
            )
        )
        eeg_filtered = torch.fft.irfft(
            eeg_spectrum * eeg_log_gain.exp(),
            n=self.patch_samples,
            dim=-1,
            norm="ortho",
        )
        eeg_scale = eeg_source.std(dim=-1, keepdim=True).clamp_min(1e-3)
        eeg_shift = 0.1 * eeg_scale * torch.tanh(
            self.eeg_mean_shift(encoded)
        ).unsqueeze(-1)
        eeg_residual = self.maximum_residual_ratio * eeg_scale * torch.tanh(
            self.eeg_residual(encoded).reshape(
                -1, self.num_patches, self.patch_samples
            )
        )
        eeg_output = eeg_mean + eeg_shift + eeg_filtered + eeg_residual
        eeg_output = eeg_output.flatten(1)
        output[:, eeg_index] = frozen_waveforms[:, eeg_index, : self.maximum_samples] + float(
            eeg_correction_scale
        ) * (eeg_output - frozen_waveforms[:, eeg_index, : self.maximum_samples])
        auxiliary.update(
            {
                "eeg_recent_blend": eeg_blend.squeeze(-1),
                "eeg_log_spectral_gain": eeg_log_gain,
                "eeg_residual": eeg_residual,
            }
        )

        emg_index = self.modalities.index("EMG")
        emg_base = self._patches(frozen_waveforms[:, emg_index])
        emg_recent = self._patches(last_observed_waveforms[:, emg_index])
        emg_blend = self.maximum_recent_blend * torch.tanh(
            self.emg_recent_blend(encoded)
        ).unsqueeze(-1)
        emg_source = emg_base + emg_blend * (emg_recent - emg_base)
        emg_mean = emg_source.mean(dim=-1, keepdim=True)
        emg_centered = emg_source - emg_mean
        emg_scale = emg_source.std(dim=-1, keepdim=True).clamp_min(1e-3)
        emg_log_gain = (
            self.maximum_emg_log_gain
            * torch.tanh(self.emg_log_scale(encoded))
            + 0.5 * torch.tanh(self.emg_burst_log_scale(encoded))
        ).unsqueeze(-1)
        emg_shift = 0.1 * emg_scale * torch.tanh(
            self.emg_mean_shift(encoded)
        ).unsqueeze(-1)
        emg_residual = self.maximum_residual_ratio * emg_scale * torch.tanh(
            self.emg_residual(encoded).reshape(
                -1, self.num_patches, self.patch_samples
            )
        )
        emg_output = (
            emg_mean
            + emg_shift
            + emg_centered * emg_log_gain.exp()
            + emg_residual
        )
        emg_output = emg_output.flatten(1)
        output[:, emg_index] = frozen_waveforms[:, emg_index, : self.maximum_samples] + float(
            emg_correction_scale
        ) * (emg_output - frozen_waveforms[:, emg_index, : self.maximum_samples])
        auxiliary.update(
            {
                "emg_recent_blend": emg_blend.squeeze(-1),
                "emg_log_gain": emg_log_gain.squeeze(-1),
                "emg_residual": emg_residual,
            }
        )
        return output, auxiliary


def last_observed_waveforms(
    history_signals: torch.Tensor,
    history_present: torch.Tensor,
    maximum_samples: int,
) -> torch.Tensor:
    """Return each modality's latest causally available waveform window."""

    if history_signals.ndim != 4 or history_present.shape != history_signals.shape[:3]:
        raise ValueError("history tensors have incompatible shapes")
    batch, epochs, modalities, samples = history_signals.shape
    if samples < int(maximum_samples):
        raise ValueError("history epochs do not cover the waveform horizon")
    indices = torch.arange(epochs, device=history_signals.device).reshape(1, -1, 1)
    last = torch.where(
        history_present.bool(), indices, -torch.ones_like(indices)
    ).amax(dim=1)
    safe = last.clamp_min(0).reshape(batch, 1, modalities, 1).expand(
        -1, 1, -1, samples
    )
    gathered = history_signals.gather(dim=1, index=safe).squeeze(1)
    has_observation = (last >= 0).unsqueeze(-1)
    return gathered[..., -int(maximum_samples) :].masked_fill(~has_observation, 0.0)


def belief_waveform_context(
    output: Dict[str, torch.Tensor],
    recursive_route: Dict[str, torch.Tensor],
    persistence_route: Dict[str, torch.Tensor],
    history_present: torch.Tensor,
) -> torch.Tensor:
    belief = recursive_route["states"][:, 0]
    persistence = persistence_route["states"][:, 0]
    physiology = recursive_route["future_physiology"][:, 0]
    reliability = output["observation_reliability"]
    freshness = output["observation_freshness"]
    age = torch.log1p(output["observation_age_epochs"])
    availability = history_present.to(belief.dtype).mean(dim=1)
    return torch.cat(
        (
            belief,
            belief - persistence,
            physiology,
            reliability,
            freshness,
            age,
            availability,
        ),
        dim=-1,
    )


def eeg_structure_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int,
    patch_samples: int,
) -> Dict[str, torch.Tensor]:
    patches = prediction.reshape(prediction.shape[0], -1, patch_samples)
    target_patches = target.reshape(target.shape[0], -1, patch_samples)
    predicted_spectrum = torch.log1p(
        torch.fft.rfft(patches - patches.mean(dim=-1, keepdim=True), dim=-1).abs()
    )
    target_spectrum = torch.log1p(
        torch.fft.rfft(
            target_patches - target_patches.mean(dim=-1, keepdim=True), dim=-1
        ).abs()
    )
    frequency = torch.fft.rfftfreq(
        prediction.shape[-1], d=1.0 / float(sample_rate)
    ).to(prediction.device)
    predicted_power = torch.fft.rfft(prediction, dim=-1).abs().square()
    target_power = torch.fft.rfft(target, dim=-1).abs().square()
    band_losses = []
    for lower, upper in ((0.5, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 30.0)):
        selected = (frequency >= lower) & (frequency < upper)
        band_losses.append(
            F.smooth_l1_loss(
                torch.log1p(predicted_power[:, selected].mean(dim=-1)),
                torch.log1p(target_power[:, selected].mean(dim=-1)),
            )
        )
    return {
        "time": F.smooth_l1_loss(prediction, target),
        "spectrum": F.smooth_l1_loss(predicted_spectrum, target_spectrum),
        "band": torch.stack(band_losses).mean(),
    }


def emg_structure_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int,
    patch_samples: int,
) -> Dict[str, torch.Tensor]:
    window = max(1, round(0.25 * int(sample_rate)))
    predicted_envelope = F.avg_pool1d(
        prediction.abs().unsqueeze(1), window, stride=window
    ).squeeze(1)
    target_envelope = F.avg_pool1d(
        target.abs().unsqueeze(1), window, stride=window
    ).squeeze(1)
    predicted_rms = prediction.reshape(
        prediction.shape[0], -1, patch_samples
    ).square().mean(dim=-1).clamp_min(1e-8).sqrt()
    target_rms = target.reshape(
        target.shape[0], -1, patch_samples
    ).square().mean(dim=-1).clamp_min(1e-8).sqrt()
    threshold = target_envelope.mean(dim=-1, keepdim=True) + target_envelope.std(
        dim=-1, keepdim=True
    )
    target_burst = (target_envelope > threshold).to(target.dtype)
    scale = target_envelope.std(dim=-1, keepdim=True).clamp_min(1e-3)
    burst_logits = (predicted_envelope - threshold) / scale
    return {
        "time": F.smooth_l1_loss(prediction, target),
        "envelope": F.smooth_l1_loss(predicted_envelope, target_envelope),
        "rms": F.smooth_l1_loss(predicted_rms, target_rms),
        "burst": F.binary_cross_entropy_with_logits(burst_logits, target_burst),
    }
