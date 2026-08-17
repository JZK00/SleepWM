from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ObservationConfig:
    modalities: Tuple[str, ...] = ("EEG", "ECG", "EMG")
    sample_rate: int = 128
    epoch_seconds: int = 30
    patch_samples: int = 64
    d_model: int = 128
    layers: int = 4
    heads: int = 4
    dropout: float = 0.1
    pooling: str = "mean"

    def __post_init__(self) -> None:
        if self.pooling not in {"mean", "state"}:
            raise ValueError("pooling must be 'mean' or 'state'")

    @property
    def samples_per_epoch(self) -> int:
        return self.sample_rate * self.epoch_seconds

    @property
    def num_patches(self) -> int:
        if self.samples_per_epoch % self.patch_samples:
            raise ValueError("samples_per_epoch must be divisible by patch_samples")
        return self.samples_per_epoch // self.patch_samples

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["modalities"] = list(self.modalities)
        return payload


def sinusoidal_positions(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    even = torch.arange(0, dim, 2, device=device, dtype=dtype)
    frequency = torch.exp(-math.log(10000.0) * even / max(dim, 1))
    angle = position * frequency.unsqueeze(0)
    encoding = torch.zeros(length, dim, device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(angle)
    if dim > 1:
        encoding[:, 1::2] = torch.cos(angle[:, : encoding[:, 1::2].shape[1]])
    return encoding


class ResidualTCNBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("TCN kernel_size must be odd")
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(1, output_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(output_channels, output_channels, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(1, output_channels),
        )
        self.skip = nn.Identity() if input_channels == output_channels else nn.Conv1d(input_channels, output_channels, 1)
        self.activation = nn.GELU()

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(signals) + self.skip(signals))


class TCNSleepClassifier(nn.Module):
    def __init__(
        self,
        modalities: int,
        channels: int = 64,
        levels: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.1,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        blocks = []
        input_channels = modalities
        for level in range(levels):
            blocks.append(ResidualTCNBlock(input_channels, channels, kernel_size, 2**level, dropout))
            input_channels = channels
        self.encoder = nn.Sequential(*blocks, nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, num_classes))

    def forward(self, signals: torch.Tensor, modality_present: Optional[torch.Tensor] = None) -> torch.Tensor:
        if modality_present is not None:
            signals = signals * modality_present.to(signals.dtype).unsqueeze(-1)
        return self.head(self.encoder(signals))


class MultiScaleEpochTokenizer(nn.Module):
    """Compact dual-scale epoch encoder used by conventional sleep baselines."""

    def __init__(self, modalities: int, channels: int, tokens: int, dropout: float) -> None:
        super().__init__()
        if channels % 2:
            raise ValueError("channels must be even")
        branch_channels = channels // 2

        def branch(kernel_size: int, stride: int) -> nn.Sequential:
            padding = kernel_size // 2
            return nn.Sequential(
                nn.Conv1d(modalities, branch_channels, kernel_size, stride=stride, padding=padding),
                nn.BatchNorm1d(branch_channels),
                nn.GELU(),
                nn.MaxPool1d(4, 4),
                nn.Conv1d(branch_channels, branch_channels, 7, padding=3),
                nn.BatchNorm1d(branch_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.AdaptiveAvgPool1d(tokens),
            )

        self.short_scale = branch(kernel_size=25, stride=8)
        self.long_scale = branch(kernel_size=101, stride=16)
        self.norm = nn.LayerNorm(channels)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        tokens = torch.cat((self.short_scale(signals), self.long_scale(signals)), dim=1)
        return self.norm(tokens.transpose(1, 2))


class CNNBiLSTMSleepClassifier(nn.Module):
    """DeepSleepNet-inspired dual-scale CNN and bidirectional LSTM reimplementation."""

    def __init__(
        self,
        modalities: int,
        channels: int = 64,
        tokens: int = 16,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        self.tokenizer = MultiScaleEpochTokenizer(modalities, channels, tokens, dropout)
        self.temporal = nn.LSTM(
            channels,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, signals: torch.Tensor, modality_present: Optional[torch.Tensor] = None) -> torch.Tensor:
        if modality_present is not None:
            signals = signals * modality_present.to(signals.dtype).unsqueeze(-1)
        encoded, _ = self.temporal(self.tokenizer(signals))
        weights = self.attention(encoded).softmax(dim=1)
        return self.head((encoded * weights).sum(dim=1))


class AttentiveCNNSleepClassifier(nn.Module):
    """AttnSleep-inspired multi-scale CNN and self-attention reimplementation."""

    def __init__(
        self,
        modalities: int,
        channels: int = 64,
        tokens: int = 16,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        self.tokenizer = MultiScaleEpochTokenizer(modalities, channels, tokens, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=heads,
            dim_feedforward=channels * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=layers)
        self.pool_query = nn.Parameter(torch.empty(channels))
        self.head = nn.Sequential(nn.LayerNorm(channels), nn.Dropout(dropout), nn.Linear(channels, num_classes))
        nn.init.normal_(self.pool_query, std=0.02)

    def forward(self, signals: torch.Tensor, modality_present: Optional[torch.Tensor] = None) -> torch.Tensor:
        if modality_present is not None:
            signals = signals * modality_present.to(signals.dtype).unsqueeze(-1)
        encoded = self.temporal(self.tokenizer(signals))
        weights = torch.einsum("btc,c->bt", encoded, self.pool_query).softmax(dim=1)
        return self.head((encoded * weights.unsqueeze(-1)).sum(dim=1))


class PatchTokenizer(nn.Module):
    def __init__(self, patch_samples: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(patch_samples),
            nn.Linear(patch_samples, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.net(patches)


class MultiModalEncoder(nn.Module):
    def __init__(self, config: ObservationConfig) -> None:
        super().__init__()
        self.config = config
        modality_count = len(config.modalities)
        self.tokenizers = nn.ModuleList(
            [PatchTokenizer(config.patch_samples, config.d_model, config.dropout) for _ in config.modalities]
        )
        self.modality_embedding = nn.Parameter(torch.empty(modality_count, config.d_model))
        self.mask_tokens = nn.Parameter(torch.empty(modality_count, config.d_model))
        self.missing_tokens = nn.Parameter(torch.empty(modality_count, config.d_model))
        self.state_token = nn.Parameter(torch.empty(1, config.d_model)) if config.pooling == "state" else None
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.norm = nn.LayerNorm(config.d_model)
        nn.init.normal_(self.modality_embedding, std=0.02)
        nn.init.normal_(self.mask_tokens, std=0.02)
        nn.init.normal_(self.missing_tokens, std=0.02)
        if self.state_token is not None:
            nn.init.normal_(self.state_token, std=0.02)

    def patchify(self, signals: torch.Tensor) -> torch.Tensor:
        if signals.ndim != 3:
            raise ValueError("signals must be [batch, modalities, samples]")
        batch, modalities, samples = signals.shape
        if modalities != len(self.config.modalities):
            raise ValueError("signal modality count does not match the model")
        if samples != self.config.samples_per_epoch:
            raise ValueError(f"expected {self.config.samples_per_epoch} samples, got {samples}")
        return signals.reshape(batch, modalities, self.config.num_patches, self.config.patch_samples)

    def modality_token_residuals(self, signals: torch.Tensor) -> Optional[torch.Tensor]:
        return None

    def encode_with_representation(
        self,
        signals: torch.Tensor,
        modality_present: Optional[torch.Tensor] = None,
        target_modality: Optional[int] = None,
        patch_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        patches = self.patchify(signals)
        batch, modality_count, num_patches, patch_samples = patches.shape
        frontend_residuals = self.modality_token_residuals(signals)
        if frontend_residuals is not None and frontend_residuals.shape != (
            batch,
            modality_count,
            num_patches,
            self.config.d_model,
        ):
            raise ValueError("modality frontend residual shape does not match patch tokens")
        if modality_present is None:
            modality_present = torch.ones(batch, modality_count, dtype=torch.bool, device=signals.device)
        else:
            modality_present = modality_present.to(device=signals.device, dtype=torch.bool)
        if modality_present.shape != (batch, modality_count):
            raise ValueError("modality_present must be [batch, modalities]")
        if target_modality is not None:
            if not 0 <= target_modality < modality_count:
                raise IndexError("target_modality is out of range")
            if patch_mask is None or patch_mask.shape != (batch, num_patches):
                raise ValueError("patch_mask must be [batch, patches]")
            patch_mask = patch_mask.to(device=signals.device, dtype=torch.bool)

        position = sinusoidal_positions(num_patches, self.config.d_model, signals.device, signals.dtype)
        sequences = []
        for modality_index in range(modality_count):
            flat = patches[:, modality_index].reshape(batch * num_patches, patch_samples)
            tokens = self.tokenizers[modality_index](flat).reshape(batch, num_patches, self.config.d_model)
            if frontend_residuals is not None:
                residual = frontend_residuals[:, modality_index]
                if target_modality == modality_index and patch_mask is not None:
                    residual = residual.masked_fill(patch_mask.unsqueeze(-1), 0.0)
                tokens = tokens + residual
            missing = self.missing_tokens[modality_index].reshape(1, 1, -1).expand_as(tokens)
            visible = modality_present[:, modality_index].reshape(batch, 1, 1)
            tokens = torch.where(visible, tokens, missing)
            if target_modality == modality_index and patch_mask is not None:
                mask_token = self.mask_tokens[modality_index].reshape(1, 1, -1).expand_as(tokens)
                tokens = torch.where(patch_mask.unsqueeze(-1), mask_token, tokens)
            tokens = tokens + self.modality_embedding[modality_index].reshape(1, 1, -1)
            tokens = tokens + position.reshape(1, num_patches, -1)
            sequences.append(tokens)
        sequence = torch.cat(sequences, dim=1)
        if self.state_token is not None:
            state = self.state_token.reshape(1, 1, -1).expand(batch, 1, -1)
            encoded = self.norm(self.transformer(torch.cat((state, sequence), dim=1)))
            representation = encoded[:, 0]
            encoded = encoded[:, 1:]
        else:
            encoded = self.norm(self.transformer(sequence))
            representation = encoded.mean(dim=1)
        tokens = encoded.reshape(batch, modality_count, num_patches, self.config.d_model)
        return representation, tokens

    def encode_tokens(
        self,
        signals: torch.Tensor,
        modality_present: Optional[torch.Tensor] = None,
        target_modality: Optional[int] = None,
        patch_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.encode_with_representation(
            signals,
            modality_present=modality_present,
            target_modality=target_modality,
            patch_mask=patch_mask,
        )[1]

    def forward(self, signals: torch.Tensor, modality_present: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encode_with_representation(signals, modality_present=modality_present)[0]


class ZeroInitializedFeatureProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, output_dim)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(self.norm(features))


class PhysiologyFrontendEncoder(MultiModalEncoder):
    """Observation encoder with zero-init physiology-informed token residuals."""

    def __init__(self, config: ObservationConfig, ecg_rr_lags: int = 24) -> None:
        super().__init__(config)
        if config.patch_samples % 8:
            raise ValueError("physiology frontend requires patch_samples divisible by eight")
        self.frontend_enabled = True
        self.physiology_frontends = nn.ModuleDict()
        if "EEG" in config.modalities:
            self.physiology_frontends["EEG_global"] = ZeroInitializedFeatureProjection(
                5, config.d_model
            )
        if "ECG" in config.modalities:
            self.physiology_frontends["ECG_global"] = ZeroInitializedFeatureProjection(
                ecg_rr_lags + 3, config.d_model
            )
            self.physiology_frontends["ECG_local"] = ZeroInitializedFeatureProjection(
                10, config.d_model
            )
        if "EMG" in config.modalities:
            self.physiology_frontends["EMG_global"] = ZeroInitializedFeatureProjection(
                3, config.d_model
            )
        minimum_lag = max(1, int(round(0.3 * config.sample_rate)))
        maximum_lag = min(config.samples_per_epoch - 1, int(round(2.0 * config.sample_rate)))
        lags = torch.linspace(minimum_lag, maximum_lag, ecg_rr_lags).round().long()
        if len(torch.unique(lags)) != ecg_rr_lags:
            raise ValueError("ECG RR lag bank contains duplicate lags")
        self.register_buffer("ecg_rr_lags", lags, persistent=False)

    def _spectrum(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        centered = signal - signal.mean(dim=-1, keepdim=True)
        spectrum = torch.fft.rfft(centered, dim=-1)
        power = spectrum.real.square() + spectrum.imag.square()
        power = power / float(signal.shape[-1])
        frequencies = torch.fft.rfftfreq(
            signal.shape[-1],
            d=1.0 / float(self.config.sample_rate),
            device=signal.device,
        ).to(dtype=signal.dtype)
        return power, frequencies

    @staticmethod
    def _band_power(
        power: torch.Tensor,
        frequencies: torch.Tensor,
        low: float,
        high: float,
    ) -> torch.Tensor:
        selected = (frequencies >= low) & (frequencies < high)
        if not bool(selected.any()):
            return power.new_full((power.shape[0],), 1e-8)
        return power[:, selected].mean(dim=-1).clamp_min(1e-8)

    def _eeg_features(self, signal: torch.Tensor) -> torch.Tensor:
        power, frequencies = self._spectrum(signal)
        bands = [
            self._band_power(power, frequencies, low, high).log()
            for low, high in ((0.5, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 30.0))
        ]
        selected = (frequencies >= 0.5) & (frequencies <= 30.0)
        selected_power = power[:, selected]
        centroid = (
            selected_power * frequencies[selected].unsqueeze(0)
        ).sum(dim=-1) / selected_power.sum(dim=-1).clamp_min(1e-8)
        return torch.stack([*bands, centroid / 30.0], dim=-1)

    def _ecg_features(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        centered = signal - signal.mean(dim=-1, keepdim=True)
        derivative = F.pad(centered[:, 1:] - centered[:, :-1], (1, 0))
        window = max(3, int(round(0.15 * self.config.sample_rate)))
        if window % 2 == 0:
            window += 1
        qrs_energy = F.avg_pool1d(
            derivative.square().unsqueeze(1),
            kernel_size=window,
            stride=1,
            padding=window // 2,
        ).squeeze(1)
        qrs_log = torch.log1p(qrs_energy)
        qrs_normalized = (qrs_log - qrs_log.mean(dim=-1, keepdim=True)) / qrs_log.std(
            dim=-1, keepdim=True
        ).clamp_min(1e-5)
        subpatch = qrs_normalized.reshape(
            signal.shape[0],
            self.config.num_patches,
            8,
            self.config.patch_samples // 8,
        )
        subpatch_max = subpatch.amax(dim=-1)
        patch_mean = subpatch.mean(dim=(-1, -2)).unsqueeze(-1)
        patch_max = subpatch.amax(dim=(-1, -2)).unsqueeze(-1)
        local_features = torch.cat((subpatch_max, patch_mean, patch_max), dim=-1)
        correlations = []
        for lag in self.ecg_rr_lags.tolist():
            correlations.append(
                (qrs_normalized[:, :-lag] * qrs_normalized[:, lag:]).mean(dim=-1)
            )
        global_features = torch.stack(correlations, dim=-1)
        morphology = torch.stack(
            (
                centered.square().mean(dim=-1).clamp_min(1e-8).log(),
                derivative.square().mean(dim=-1).clamp_min(1e-8).log(),
                qrs_log.amax(dim=-1),
            ),
            dim=-1,
        )
        return torch.cat((global_features, morphology), dim=-1), local_features

    def _emg_features(self, signal: torch.Tensor) -> torch.Tensor:
        centered = signal - signal.mean(dim=-1, keepdim=True)
        rms = centered.square().mean(dim=-1).clamp_min(1e-8).sqrt().log()
        rectified = centered.abs().mean(dim=-1).clamp_min(1e-8).log()
        power, frequencies = self._spectrum(centered)
        high = power[:, (frequencies >= 20.0) & (frequencies <= 45.0)].sum(dim=-1)
        total = power[:, (frequencies >= 0.5) & (frequencies <= 45.0)].sum(dim=-1)
        ratio = high / total.clamp_min(1e-8)
        return torch.stack((rms, rectified, ratio), dim=-1)

    def modality_token_residuals(self, signals: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.frontend_enabled:
            return None
        residuals = []
        for modality_index, modality in enumerate(self.config.modalities):
            signal = signals[:, modality_index]
            if modality == "EEG":
                global_residual = self.physiology_frontends["EEG_global"](
                    self._eeg_features(signal)
                )
                residual = global_residual.unsqueeze(1).expand(
                    -1, self.config.num_patches, -1
                )
            elif modality == "ECG":
                global_features, local_features = self._ecg_features(signal)
                global_residual = self.physiology_frontends["ECG_global"](global_features)
                local_residual = self.physiology_frontends["ECG_local"](local_features)
                residual = global_residual.unsqueeze(1) + local_residual
            elif modality == "EMG":
                global_residual = self.physiology_frontends["EMG_global"](
                    self._emg_features(signal)
                )
                residual = global_residual.unsqueeze(1).expand(
                    -1, self.config.num_patches, -1
                )
            else:
                residual = signals.new_zeros(
                    signals.shape[0], self.config.num_patches, self.config.d_model
                )
            residuals.append(residual)
        return torch.stack(residuals, dim=1)


def masked_patch_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    error = (prediction - target).pow(2).mean(dim=-1)
    weight = mask.to(dtype=error.dtype)
    return (error * weight).sum() / weight.sum().clamp_min(1.0)


def masked_patch_latent_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    prediction = F.layer_norm(prediction, (prediction.shape[-1],))
    target = F.layer_norm(target.detach(), (target.shape[-1],))
    error = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
    weight = mask.to(dtype=error.dtype)
    return (error * weight).sum() / weight.sum().clamp_min(1.0)


class MaskedMultiModalModel(nn.Module):
    def __init__(
        self,
        encoder: MultiModalEncoder,
        latent_prediction: bool = False,
        teacher_target: str = "tokenizer",
    ) -> None:
        super().__init__()
        if teacher_target not in {"tokenizer", "contextual"}:
            raise ValueError("teacher_target must be 'tokenizer' or 'contextual'")
        self.encoder = encoder
        self.teacher_target = teacher_target
        self.teacher_encoder = copy.deepcopy(encoder) if latent_prediction else None
        if self.teacher_encoder is not None:
            self.teacher_encoder.requires_grad_(False)
            self.teacher_encoder.eval()
        self.decoders = nn.ModuleList(
            [nn.Linear(encoder.config.d_model, encoder.config.patch_samples) for _ in encoder.config.modalities]
        )
        self.latent_predictors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(encoder.config.d_model),
                    nn.Linear(encoder.config.d_model, encoder.config.d_model * 2),
                    nn.GELU(),
                    nn.Linear(encoder.config.d_model * 2, encoder.config.d_model),
                )
                for _ in encoder.config.modalities
            ]
        )

    @property
    def config(self) -> ObservationConfig:
        return self.encoder.config

    def teacher_tokens(self, signals: torch.Tensor, modality_present: torch.Tensor) -> torch.Tensor:
        if self.teacher_encoder is None:
            raise RuntimeError("latent teacher is not enabled")
        self.teacher_encoder.eval()
        with torch.no_grad():
            if self.teacher_target == "contextual":
                _, targets = self.teacher_encoder.encode_with_representation(
                    signals,
                    modality_present=modality_present,
                )
                return targets
            patches = self.teacher_encoder.patchify(signals)
            targets = []
            for modality_index, tokenizer in enumerate(self.teacher_encoder.tokenizers):
                flat = patches[:, modality_index].reshape(-1, self.config.patch_samples)
                targets.append(tokenizer(flat).reshape(signals.shape[0], self.config.num_patches, -1))
            return torch.stack(targets, dim=1)

    def teacher_representation(self, signals: torch.Tensor, modality_present: torch.Tensor) -> torch.Tensor:
        if self.teacher_encoder is None:
            raise RuntimeError("latent teacher is not enabled")
        self.teacher_encoder.eval()
        with torch.no_grad():
            return self.teacher_encoder(signals, modality_present=modality_present)

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        if self.teacher_encoder is None:
            return
        if not 0.0 <= momentum < 1.0:
            raise ValueError("teacher momentum must be in [0, 1)")
        for teacher, student in zip(self.teacher_encoder.parameters(), self.encoder.parameters()):
            teacher.mul_(momentum).add_(student, alpha=1.0 - momentum)
        for teacher, student in zip(self.teacher_encoder.buffers(), self.encoder.buffers()):
            teacher.copy_(student)

    def pretrained_encoder_state(self):
        source = self.teacher_encoder if self.teacher_encoder is not None else self.encoder
        return source.state_dict()

    def forward(
        self,
        signals: torch.Tensor,
        target_modality: int,
        patch_mask: torch.Tensor,
        modality_present: Optional[torch.Tensor] = None,
        target_latent: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        patches = self.encoder.patchify(signals)
        representation, encoded = self.encoder.encode_with_representation(
            signals,
            modality_present=modality_present,
            target_modality=target_modality,
            patch_mask=patch_mask,
        )
        prediction = self.decoders[target_modality](encoded[:, target_modality])
        latent_prediction = self.latent_predictors[target_modality](encoded[:, target_modality])
        target = patches[:, target_modality]
        latent_loss = (
            masked_patch_latent_loss(latent_prediction, target_latent, patch_mask)
            if target_latent is not None
            else prediction.new_zeros(())
        )
        return {
            "reconstruction_loss": masked_patch_mse(prediction, target, patch_mask),
            "latent_reconstruction_loss": latent_loss,
            "prediction": prediction,
            "latent_prediction": latent_prediction,
            "target": target,
            "representation": representation,
        }


class SleepStageClassifier(nn.Module):
    def __init__(
        self,
        encoder: MultiModalEncoder,
        num_classes: int = 5,
        hidden_dim: int = 0,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for parameter in encoder.parameters():
                parameter.requires_grad = False
        if hidden_dim > 0:
            self.head = nn.Sequential(
                nn.LayerNorm(encoder.config.d_model),
                nn.Linear(encoder.config.d_model, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.head = nn.Sequential(nn.LayerNorm(encoder.config.d_model), nn.Linear(encoder.config.d_model, num_classes))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, signals: torch.Tensor, modality_present: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                representation = self.encoder(signals, modality_present)
        else:
            representation = self.encoder(signals, modality_present)
        return self.head(representation)


class CausalPhysioWorldModel(nn.Module):
    def __init__(
        self,
        encoder: MultiModalEncoder,
        horizons: Sequence[int],
        num_classes: int = 5,
        transition_layers: int = 2,
        transition_heads: int = 4,
        dropout: float = 0.1,
        freeze_observation_encoder: bool = False,
        stage_residual_from_current: bool = False,
        use_frozen_target_encoder: bool = False,
        factorized_transition_head: bool = False,
        change_prior_probabilities: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.target_encoder = copy.deepcopy(encoder) if use_frozen_target_encoder else None
        self.horizons = tuple(int(value) for value in horizons)
        self.freeze_observation_encoder = freeze_observation_encoder
        self.stage_residual_from_current = stage_residual_from_current
        self.factorized_transition_head = factorized_transition_head
        if factorized_transition_head and not stage_residual_from_current:
            raise ValueError("factorized_transition_head requires stage_residual_from_current")
        if self.target_encoder is not None:
            self.target_encoder.requires_grad_(False)
            self.target_encoder.eval()
            if hasattr(self.target_encoder, "frontend_enabled"):
                self.target_encoder.frontend_enabled = False
        if freeze_observation_encoder:
            for parameter in encoder.parameters():
                parameter.requires_grad = False
        layer = nn.TransformerEncoderLayer(
            d_model=encoder.config.d_model,
            nhead=transition_heads,
            dim_feedforward=encoder.config.d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transition = nn.TransformerEncoder(layer, num_layers=transition_layers)
        self.horizon_embedding = nn.Parameter(torch.empty(len(self.horizons), encoder.config.d_model))
        self.state_predictor = nn.Sequential(
            nn.LayerNorm(encoder.config.d_model),
            nn.Linear(encoder.config.d_model, encoder.config.d_model * 2),
            nn.GELU(),
            nn.Linear(encoder.config.d_model * 2, encoder.config.d_model),
        )
        self.stage_head = (
            None
            if factorized_transition_head
            else nn.Sequential(nn.LayerNorm(encoder.config.d_model), nn.Linear(encoder.config.d_model, num_classes))
        )
        self.change_head = (
            nn.Sequential(
                nn.LayerNorm(encoder.config.d_model),
                nn.Linear(encoder.config.d_model, 1, bias=False),
            )
            if factorized_transition_head
            else None
        )
        self.destination_head = (
            nn.Sequential(
                nn.LayerNorm(encoder.config.d_model),
                nn.Linear(encoder.config.d_model, num_classes),
            )
            if factorized_transition_head
            else None
        )
        self.change_horizon_bias = None
        if factorized_transition_head:
            priors = (
                tuple(float(value) for value in change_prior_probabilities)
                if change_prior_probabilities is not None
                else tuple(0.2 for _ in self.horizons)
            )
            if len(priors) != len(self.horizons) or any(not 0.0 < value < 1.0 for value in priors):
                raise ValueError("change priors must match horizons and lie in (0, 1)")
            prior_tensor = torch.tensor(priors, dtype=torch.float32)
            self.change_horizon_bias = nn.Parameter(torch.logit(prior_tensor))
        self.current_stage_head = (
            nn.Sequential(nn.LayerNorm(encoder.config.d_model), nn.Linear(encoder.config.d_model, num_classes))
            if stage_residual_from_current
            else None
        )
        nn.init.normal_(self.horizon_embedding, std=0.02)
        if stage_residual_from_current and self.stage_head is not None:
            nn.init.zeros_(self.stage_head[-1].weight)
            nn.init.zeros_(self.stage_head[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_observation_encoder:
            self.encoder.eval()
        if self.target_encoder is not None:
            self.target_encoder.eval()
        return self

    def _encode_epochs(
        self,
        signals: torch.Tensor,
        modality_present: Optional[torch.Tensor],
        require_gradient: bool,
        encoder: Optional[MultiModalEncoder] = None,
    ) -> torch.Tensor:
        if signals.ndim != 4:
            raise ValueError("sequence signals must be [batch, time, modalities, samples]")
        batch, time, modalities, samples = signals.shape
        flat_signals = signals.reshape(batch * time, modalities, samples)
        flat_present = None
        if modality_present is not None:
            flat_present = modality_present.reshape(batch * time, modalities)
        active_encoder = self.encoder if encoder is None else encoder
        if require_gradient:
            encoded = active_encoder(flat_signals, flat_present)
        else:
            with torch.no_grad():
                encoded = active_encoder(flat_signals, flat_present)
        return encoded.reshape(batch, time, -1)

    def rollout(
        self,
        history_signals: torch.Tensor,
        history_present: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        states = self._encode_epochs(
            history_signals,
            history_present,
            require_gradient=not self.freeze_observation_encoder,
        )
        return self._rollout_from_states(states)

    def _rollout_from_states(self, states: torch.Tensor) -> Dict[str, torch.Tensor]:
        if states.ndim != 3:
            raise ValueError("history states must be [batch, time, dimensions]")
        positions = sinusoidal_positions(states.shape[1], states.shape[2], states.device, states.dtype)
        causal_mask = torch.triu(
            torch.ones(states.shape[1], states.shape[1], dtype=torch.bool, device=states.device), diagonal=1
        )
        transitioned = self.transition(states + positions.unsqueeze(0), mask=causal_mask)
        context = transitioned[:, -1].unsqueeze(1) + self.horizon_embedding.unsqueeze(0)
        predicted_states = self.state_predictor(context)
        current_stage_logits = None
        if self.current_stage_head is not None:
            current_stage_logits = self.current_stage_head(states[:, -1])
        extra = {}
        if self.factorized_transition_head:
            if self.change_head is None or self.destination_head is None or self.change_horizon_bias is None:
                raise RuntimeError("factorized transition modules are not initialized")
            change_logits = self.change_head(predicted_states).squeeze(-1) + self.change_horizon_bias.unsqueeze(0)
            destination_logits = self.destination_head(predicted_states)
            change_probabilities = change_logits.sigmoid().unsqueeze(-1)
            current_probabilities = current_stage_logits.softmax(dim=-1).unsqueeze(1)
            destination_probabilities = destination_logits.softmax(dim=-1)
            future_probabilities = (
                (1.0 - change_probabilities) * current_probabilities
                + change_probabilities * destination_probabilities
            )
            stage_logits = future_probabilities.clamp_min(1e-8).log()
            extra = {
                "change_logits": change_logits,
                "change_probabilities": change_probabilities.squeeze(-1),
                "destination_logits": destination_logits,
            }
        else:
            if self.stage_head is None:
                raise RuntimeError("stage head is not initialized")
            stage_logits = self.stage_head(predicted_states)
            if current_stage_logits is not None:
                stage_logits = stage_logits + current_stage_logits.unsqueeze(1)
        return {
            "predicted_states": predicted_states,
            "stage_logits": stage_logits,
            "history_state": states[:, -1],
            **extra,
            **({"current_stage_logits": current_stage_logits} if current_stage_logits is not None else {}),
        }

    def forward(
        self,
        history_signals: torch.Tensor,
        history_present: Optional[torch.Tensor] = None,
        future_signals: Optional[torch.Tensor] = None,
        future_present: Optional[torch.Tensor] = None,
        future_labels: Optional[torch.Tensor] = None,
        history_labels: Optional[torch.Tensor] = None,
        latent_weight: float = 1.0,
        stage_weight: float = 1.0,
        current_stage_weight: float = 0.0,
        transition_stage_weight: float = 1.0,
        change_loss_weight: float = 0.0,
        destination_loss_weight: float = 0.0,
        stage_class_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        output = self.rollout(history_signals, history_present)
        total_loss = output["predicted_states"].sum() * 0.0
        if future_signals is not None:
            target_states = self._encode_epochs(
                future_signals,
                future_present,
                require_gradient=False,
                encoder=self.target_encoder,
            )
            latent_loss = F.smooth_l1_loss(output["predicted_states"], target_states)
            output["target_states"] = target_states
            output["latent_loss"] = latent_loss
            total_loss = total_loss + float(latent_weight) * latent_loss
        if future_labels is not None:
            if transition_stage_weight <= 0.0:
                raise ValueError("transition_stage_weight must be positive")
            flat_labels = future_labels.reshape(-1)
            stage_losses = F.cross_entropy(
                output["stage_logits"].reshape(-1, output["stage_logits"].shape[-1]),
                flat_labels,
                weight=stage_class_weights,
                reduction="none",
            )
            transition_weights = torch.ones_like(future_labels, dtype=stage_losses.dtype)
            if history_labels is not None and transition_stage_weight != 1.0:
                transition_mask = future_labels != history_labels[:, -1].unsqueeze(1)
                transition_weights = torch.where(
                    transition_mask,
                    transition_weights.new_full((), float(transition_stage_weight)),
                    transition_weights,
                )
            flat_transition_weights = transition_weights.reshape(-1)
            denominator_weights = flat_transition_weights
            if stage_class_weights is not None:
                denominator_weights = denominator_weights * stage_class_weights[flat_labels]
            stage_loss = (stage_losses * flat_transition_weights).sum() / denominator_weights.sum().clamp_min(1.0)
            output["stage_loss"] = stage_loss
            total_loss = total_loss + float(stage_weight) * stage_loss
            if self.factorized_transition_head:
                if history_labels is None:
                    raise ValueError("factorized transition losses require history_labels")
                transition_mask = future_labels != history_labels[:, -1].unsqueeze(1)
                change_loss = F.binary_cross_entropy_with_logits(
                    output["change_logits"],
                    transition_mask.to(dtype=output["change_logits"].dtype),
                )
                if transition_mask.any():
                    destination_loss = F.cross_entropy(
                        output["destination_logits"][transition_mask],
                        future_labels[transition_mask],
                        weight=stage_class_weights,
                    )
                else:
                    destination_loss = output["destination_logits"].sum() * 0.0
                output["change_loss"] = change_loss
                output["destination_loss"] = destination_loss
                total_loss = total_loss + float(change_loss_weight) * change_loss
                total_loss = total_loss + float(destination_loss_weight) * destination_loss
        if history_labels is not None and self.current_stage_head is not None:
            current_stage_loss = F.cross_entropy(
                output["current_stage_logits"],
                history_labels[:, -1],
                weight=stage_class_weights,
            )
            output["current_stage_loss"] = current_stage_loss
            total_loss = total_loss + float(current_stage_weight) * current_stage_loss
        output["loss"] = total_loss
        return output


class PhysiologyAwareWorldModel(CausalPhysioWorldModel):
    """Causal world model with modality-specific current and residual future heads."""

    def __init__(
        self,
        *args,
        physiology_group_sizes: Dict[str, int],
        physiology_hidden_dim: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not physiology_group_sizes or any(int(size) < 1 for size in physiology_group_sizes.values()):
            raise ValueError("physiology groups must have positive sizes")
        self.physiology_group_sizes = {
            str(group): int(size) for group, size in physiology_group_sizes.items()
        }
        state_dim = self.encoder.config.d_model

        def make_head(output_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(state_dim),
                nn.Linear(state_dim, physiology_hidden_dim),
                nn.GELU(),
                nn.Dropout(float(self.encoder.config.dropout)),
                nn.Linear(physiology_hidden_dim, output_dim),
            )

        self.current_physiology_heads = nn.ModuleDict(
            {group: make_head(size) for group, size in self.physiology_group_sizes.items()}
        )
        self.future_physiology_delta_heads = nn.ModuleDict(
            {group: make_head(size) for group, size in self.physiology_group_sizes.items()}
        )
        for head in self.future_physiology_delta_heads.values():
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def rollout(
        self,
        history_signals: torch.Tensor,
        history_present: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        output = super().rollout(history_signals, history_present)
        current_by_group = {
            group: head(output["history_state"])
            for group, head in self.current_physiology_heads.items()
        }
        future_by_group = {
            group: current_by_group[group].unsqueeze(1)
            + self.future_physiology_delta_heads[group](output["predicted_states"])
            for group in self.physiology_group_sizes
        }
        output["current_physiology"] = torch.cat(list(current_by_group.values()), dim=-1)
        output["future_physiology"] = torch.cat(list(future_by_group.values()), dim=-1)
        return output

    def _group_masked_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        group_losses = {}
        offset = 0
        for group, size in self.physiology_group_sizes.items():
            selected = slice(offset, offset + size)
            element_loss = F.smooth_l1_loss(
                prediction[..., selected], target[..., selected], reduction="none"
            )
            weight = valid[..., selected].to(dtype=element_loss.dtype)
            group_losses[group] = (element_loss * weight).sum() / weight.sum().clamp_min(1.0)
            offset += size
        return torch.stack(list(group_losses.values())).mean(), group_losses

    def forward(
        self,
        *args,
        current_physiology_targets: Optional[torch.Tensor] = None,
        current_physiology_valid: Optional[torch.Tensor] = None,
        future_physiology_targets: Optional[torch.Tensor] = None,
        future_physiology_valid: Optional[torch.Tensor] = None,
        current_physiology_weight: float = 1.0,
        future_physiology_weight: float = 1.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        output = super().forward(*args, **kwargs)
        if current_physiology_targets is not None:
            if current_physiology_valid is None:
                raise ValueError("current physiology targets require a validity mask")
            loss, by_group = self._group_masked_loss(
                output["current_physiology"],
                current_physiology_targets,
                current_physiology_valid,
            )
            output["current_physiology_loss"] = loss
            output["current_physiology_loss_by_group"] = by_group
            output["loss"] = output["loss"] + float(current_physiology_weight) * loss
        if future_physiology_targets is not None:
            if future_physiology_valid is None:
                raise ValueError("future physiology targets require a validity mask")
            loss, by_group = self._group_masked_loss(
                output["future_physiology"],
                future_physiology_targets,
                future_physiology_valid,
            )
            output["future_physiology_loss"] = loss
            output["future_physiology_loss_by_group"] = by_group
            output["loss"] = output["loss"] + float(future_physiology_weight) * loss
        return output


class TrajectoryPhysiologyWorldModel(PhysiologyAwareWorldModel):
    """World model with modality-token supervision across the full observed history."""

    uses_history_trajectory = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        unknown_groups = set(self.physiology_group_sizes) - set(self.encoder.config.modalities)
        if unknown_groups:
            raise ValueError(f"physiology groups are not encoder modalities: {sorted(unknown_groups)}")
        state_dim = self.encoder.config.d_model
        self.physiology_state_adapters = nn.ModuleDict(
            {
                group: nn.Linear(size, state_dim, bias=False)
                for group, size in self.physiology_group_sizes.items()
            }
        )
        for adapter in self.physiology_state_adapters.values():
            nn.init.zeros_(adapter.weight)

    def _encode_history_with_modality_states(
        self,
        history_signals: torch.Tensor,
        history_present: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if history_signals.ndim != 4:
            raise ValueError("sequence signals must be [batch, time, modalities, samples]")
        batch, time, modalities, samples = history_signals.shape
        flat_signals = history_signals.reshape(batch * time, modalities, samples)
        flat_present = None
        if history_present is not None:
            flat_present = history_present.reshape(batch * time, modalities)

        def encode() -> tuple[torch.Tensor, torch.Tensor]:
            representation, tokens = self.encoder.encode_with_representation(
                flat_signals,
                modality_present=flat_present,
            )
            return representation, tokens.mean(dim=2)

        if self.freeze_observation_encoder:
            with torch.no_grad():
                states, modality_states = encode()
        else:
            states, modality_states = encode()
        return (
            states.reshape(batch, time, -1),
            modality_states.reshape(batch, time, modalities, -1),
        )

    def rollout(
        self,
        history_signals: torch.Tensor,
        history_present: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        states, modality_states = self._encode_history_with_modality_states(
            history_signals,
            history_present,
        )
        history_by_group = {}
        adapter_delta = torch.zeros_like(states)
        for group, head in self.current_physiology_heads.items():
            modality_index = self.encoder.config.modalities.index(group)
            prediction = head(modality_states[:, :, modality_index])
            history_by_group[group] = prediction
            adapter_delta = adapter_delta + self.physiology_state_adapters[group](prediction)
        adapted_states = states + adapter_delta
        output = self._rollout_from_states(adapted_states)
        current_by_group = {
            group: prediction[:, -1] for group, prediction in history_by_group.items()
        }
        future_by_group = {
            group: current_by_group[group].unsqueeze(1)
            + self.future_physiology_delta_heads[group](output["predicted_states"])
            for group in self.physiology_group_sizes
        }
        output["history_physiology"] = torch.cat(list(history_by_group.values()), dim=-1)
        output["current_physiology"] = torch.cat(list(current_by_group.values()), dim=-1)
        output["future_physiology"] = torch.cat(list(future_by_group.values()), dim=-1)
        output["physiology_adapter_delta"] = adapter_delta
        return output

    def forward(
        self,
        *args,
        history_physiology_targets: Optional[torch.Tensor] = None,
        history_physiology_valid: Optional[torch.Tensor] = None,
        future_physiology_targets: Optional[torch.Tensor] = None,
        future_physiology_valid: Optional[torch.Tensor] = None,
        history_physiology_weight: float = 1.0,
        future_physiology_weight: float = 1.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        output = CausalPhysioWorldModel.forward(self, *args, **kwargs)
        if history_physiology_targets is not None:
            if history_physiology_valid is None:
                raise ValueError("history physiology targets require a validity mask")
            loss, by_group = self._group_masked_loss(
                output["history_physiology"],
                history_physiology_targets,
                history_physiology_valid,
            )
            output["history_physiology_loss"] = loss
            output["history_physiology_loss_by_group"] = by_group
            output["loss"] = output["loss"] + float(history_physiology_weight) * loss
        if future_physiology_targets is not None:
            if future_physiology_valid is None:
                raise ValueError("future physiology targets require a validity mask")
            loss, by_group = self._group_masked_loss(
                output["future_physiology"],
                future_physiology_targets,
                future_physiology_valid,
            )
            output["future_physiology_loss"] = loss
            output["future_physiology_loss_by_group"] = by_group
            output["loss"] = output["loss"] + float(future_physiology_weight) * loss
        return output


class PhysiologyStateSpaceWorldModel(TrajectoryPhysiologyWorldModel):
    """F4 model augmented with causal dynamics over decoded physiology trajectories."""

    uses_physiology_state_space_dynamics = True

    def __init__(
        self,
        *args,
        physiology_dynamics_dim: int = 64,
        physiology_dynamics_layers: int = 1,
        physiology_dynamics_heads: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if physiology_dynamics_dim % physiology_dynamics_heads:
            raise ValueError("physiology dynamics dimensions must divide attention heads")
        self.physiology_dynamics_embeddings = nn.ModuleDict()
        self.physiology_dynamics_transitions = nn.ModuleDict()
        self.physiology_dynamics_horizon_embeddings = nn.ParameterDict()
        self.physiology_dynamics_predictors = nn.ModuleDict()
        self.physiology_dynamics_delta_heads = nn.ModuleDict()
        self.physiology_dynamics_state_adapters = nn.ModuleDict()
        state_dim = self.encoder.config.d_model
        for group, feature_count in self.physiology_group_sizes.items():
            self.physiology_dynamics_embeddings[group] = nn.Sequential(
                nn.LayerNorm(feature_count),
                nn.Linear(feature_count, physiology_dynamics_dim),
            )
            layer = nn.TransformerEncoderLayer(
                d_model=physiology_dynamics_dim,
                nhead=physiology_dynamics_heads,
                dim_feedforward=physiology_dynamics_dim * 4,
                dropout=float(self.encoder.config.dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.physiology_dynamics_transitions[group] = nn.TransformerEncoder(
                layer,
                num_layers=physiology_dynamics_layers,
            )
            horizon_embedding = nn.Parameter(
                torch.empty(len(self.horizons), physiology_dynamics_dim)
            )
            nn.init.normal_(horizon_embedding, std=0.02)
            self.physiology_dynamics_horizon_embeddings[group] = horizon_embedding
            self.physiology_dynamics_predictors[group] = nn.Sequential(
                nn.LayerNorm(physiology_dynamics_dim),
                nn.Linear(physiology_dynamics_dim, physiology_dynamics_dim * 2),
                nn.GELU(),
                nn.Linear(physiology_dynamics_dim * 2, physiology_dynamics_dim),
            )
            delta_head = nn.Sequential(
                nn.LayerNorm(physiology_dynamics_dim),
                nn.Linear(physiology_dynamics_dim, physiology_dynamics_dim),
                nn.GELU(),
                nn.Linear(physiology_dynamics_dim, feature_count),
            )
            nn.init.zeros_(delta_head[-1].weight)
            nn.init.zeros_(delta_head[-1].bias)
            self.physiology_dynamics_delta_heads[group] = delta_head
            state_adapter = nn.Linear(physiology_dynamics_dim, state_dim, bias=False)
            nn.init.zeros_(state_adapter.weight)
            self.physiology_dynamics_state_adapters[group] = state_adapter

    def rollout(
        self,
        history_signals: torch.Tensor,
        history_present: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        states, modality_states = self._encode_history_with_modality_states(
            history_signals,
            history_present,
        )
        history_by_group = {}
        adapter_delta = torch.zeros_like(states)
        for group, head in self.current_physiology_heads.items():
            modality_index = self.encoder.config.modalities.index(group)
            prediction = head(modality_states[:, :, modality_index])
            history_by_group[group] = prediction
            adapter_delta = adapter_delta + self.physiology_state_adapters[group](prediction)
        output = self._rollout_from_states(states + adapter_delta)

        positions = sinusoidal_positions(
            states.shape[1],
            next(iter(self.physiology_dynamics_embeddings.values()))[-1].out_features,
            states.device,
            states.dtype,
        )
        causal_mask = torch.triu(
            torch.ones(
                states.shape[1],
                states.shape[1],
                dtype=torch.bool,
                device=states.device,
            ),
            diagonal=1,
        )
        dynamics_by_group = {}
        trajectory_delta_by_group = {}
        future_state_delta = torch.zeros_like(output["predicted_states"])
        for group, history_prediction in history_by_group.items():
            embedded = self.physiology_dynamics_embeddings[group](history_prediction)
            transitioned = self.physiology_dynamics_transitions[group](
                embedded + positions.unsqueeze(0),
                mask=causal_mask,
            )
            context = transitioned[:, -1].unsqueeze(1)
            context = context + self.physiology_dynamics_horizon_embeddings[group].unsqueeze(0)
            dynamics_state = self.physiology_dynamics_predictors[group](context)
            dynamics_by_group[group] = dynamics_state
            trajectory_delta_by_group[group] = self.physiology_dynamics_delta_heads[group](
                dynamics_state
            )
            future_state_delta = future_state_delta + self.physiology_dynamics_state_adapters[
                group
            ](dynamics_state)

        predicted_states = output["predicted_states"] + future_state_delta
        if self.factorized_transition_head or self.stage_head is None:
            raise ValueError("physiology state-space dynamics require the standard stage head")
        stage_logits = self.stage_head(predicted_states)
        if self.current_stage_head is not None:
            stage_logits = stage_logits + output["current_stage_logits"].unsqueeze(1)
        current_by_group = {
            group: prediction[:, -1] for group, prediction in history_by_group.items()
        }
        future_by_group = {
            group: current_by_group[group].unsqueeze(1)
            + self.future_physiology_delta_heads[group](predicted_states)
            + trajectory_delta_by_group[group]
            for group in self.physiology_group_sizes
        }
        output["predicted_states"] = predicted_states
        output["stage_logits"] = stage_logits
        output["history_physiology"] = torch.cat(list(history_by_group.values()), dim=-1)
        output["current_physiology"] = torch.cat(list(current_by_group.values()), dim=-1)
        output["future_physiology"] = torch.cat(list(future_by_group.values()), dim=-1)
        output["physiology_dynamics_states"] = torch.stack(
            [dynamics_by_group[group] for group in self.encoder.config.modalities],
            dim=2,
        )
        output["physiology_trajectory_delta"] = torch.cat(
            [trajectory_delta_by_group[group] for group in self.physiology_group_sizes],
            dim=-1,
        )
        output["physiology_dynamics_state_delta"] = future_state_delta
        output["history_modality_states"] = modality_states
        output["history_epoch_states"] = states + adapter_delta
        return output


class ShortHorizonWaveformDecoder(nn.Module):
    """Decode future waveforms from physiological state and recent signal context."""

    def __init__(
        self,
        modalities: Sequence[str],
        shared_state_dim: int,
        dynamics_state_dim: int,
        sample_rate: int,
        waveform_seconds: int = 10,
        patch_samples: int = 64,
        decoder_dim: int = 64,
        decoder_layers: int = 1,
        decoder_heads: int = 4,
        dropout: float = 0.1,
        structured_event_heads: bool = False,
        probabilistic_event_heads: bool = False,
        ecg_refractory_event_head: bool = False,
        ecg_rr_bins: int = 48,
        output_baseline: str = "repeat",
        physiological_event_renderer: bool = False,
        ecg_time_aligned_renderer: bool = False,
        ecg_recursive_event_renderer: bool = False,
        ecg_recent_rr_residual: bool = False,
        ecg_recent_amplitude_calibration: bool = False,
        safe_modality_residual_refinement: bool = False,
        missing_modality_conditioning: bool = False,
        missing_physiology_calibration: bool = False,
        missing_emg_teacher_calibration: bool = False,
        ecg_event_sigma_seconds: float = 0.02,
    ) -> None:
        super().__init__()
        self.modalities = tuple(modalities)
        self.sample_rate = int(sample_rate)
        self.waveform_seconds = int(waveform_seconds)
        self.patch_samples = int(patch_samples)
        self.structured_event_heads = bool(structured_event_heads)
        self.probabilistic_event_heads = bool(probabilistic_event_heads)
        self.ecg_refractory_event_enabled = bool(ecg_refractory_event_head)
        self.output_baseline = str(output_baseline)
        self.physiological_event_renderer = bool(physiological_event_renderer)
        self.ecg_time_aligned_renderer = bool(ecg_time_aligned_renderer)
        self.ecg_recursive_event_renderer = bool(ecg_recursive_event_renderer)
        self.ecg_recent_rr_residual = bool(ecg_recent_rr_residual)
        self.ecg_recent_amplitude_calibration = bool(
            ecg_recent_amplitude_calibration
        )
        self.safe_modality_residual_refinement = bool(
            safe_modality_residual_refinement
        )
        self.missing_modality_conditioning = bool(missing_modality_conditioning)
        self.missing_physiology_calibration = bool(
            missing_physiology_calibration
        )
        self.missing_emg_teacher_calibration = bool(
            missing_emg_teacher_calibration
        )
        self.missing_emg_log_rms_bias = (
            nn.Parameter(torch.zeros(3))
            if self.missing_emg_teacher_calibration
            else None
        )
        self.ecg_event_sigma_seconds = float(ecg_event_sigma_seconds)
        if self.output_baseline not in ("repeat", "none"):
            raise ValueError("waveform output baseline must be 'repeat' or 'none'")
        if self.probabilistic_event_heads and not self.structured_event_heads:
            raise ValueError("probabilistic event heads require structured event heads")
        if self.physiological_event_renderer and not self.probabilistic_event_heads:
            raise ValueError("physiological event rendering requires probability heads")
        if self.ecg_refractory_event_enabled and not self.probabilistic_event_heads:
            raise ValueError("ECG refractory events require probabilistic event heads")
        if self.ecg_time_aligned_renderer and not (
            self.physiological_event_renderer and self.ecg_refractory_event_enabled
        ):
            raise ValueError(
                "time-aligned ECG rendering requires physiological and refractory heads"
            )
        if self.ecg_recursive_event_renderer and not self.ecg_time_aligned_renderer:
            raise ValueError(
                "recursive ECG events require the time-aligned renderer"
            )
        if self.ecg_recent_rr_residual and not self.ecg_recursive_event_renderer:
            raise ValueError("recent-RR residuals require recursive ECG events")
        if self.ecg_recent_amplitude_calibration and not self.ecg_recent_rr_residual:
            raise ValueError("ECG amplitude calibration requires recent-RR residuals")
        if self.safe_modality_residual_refinement and not (
            self.physiological_event_renderer and self.probabilistic_event_heads
        ):
            raise ValueError(
                "safe modality refinement requires physiological probability heads"
            )
        if self.ecg_event_sigma_seconds <= 0.0:
            raise ValueError("ECG event rasterization sigma must be positive")
        self.ecg_rr_bins = int(ecg_rr_bins)
        if self.ecg_rr_bins < 2:
            raise ValueError("ECG RR distribution requires at least two bins")
        self.register_buffer(
            "ecg_rr_bin_centers_seconds",
            torch.linspace(0.25, 2.0, self.ecg_rr_bins),
            persistent=False,
        )
        self.max_samples = self.sample_rate * self.waveform_seconds
        if self.max_samples < 1 or self.max_samples % self.patch_samples:
            raise ValueError("waveform samples must be positive and divisible by patch size")
        if decoder_dim % decoder_heads:
            raise ValueError("waveform decoder dimensions must divide attention heads")
        self.num_patches = self.max_samples // self.patch_samples
        self.patch_encoders = nn.ModuleDict()
        self.memory_encoders = nn.ModuleDict()
        self.context_projections = nn.ModuleDict()
        self.future_queries = nn.ParameterDict()
        self.decoders = nn.ModuleDict()
        self.output_heads = nn.ModuleDict()
        self.structure_heads = nn.ModuleDict()
        self.structure_adapters = nn.ModuleDict()
        self.probability_heads = nn.ModuleDict()
        self.probability_adapters = nn.ModuleDict()
        self.event_output_heads = nn.ModuleDict()
        self.ecg_point_process_head: Optional[nn.Module] = None
        self.ecg_point_process_adapter: Optional[nn.Module] = None
        self.ecg_point_process_output_head: Optional[nn.Module] = None
        self.ecg_event_timing_head: Optional[nn.Module] = None
        self.ecg_rr_residual_gate_head: Optional[nn.Module] = None
        self.safe_refinement_heads = nn.ModuleDict()
        self.missing_memory_tokens = nn.ParameterDict()
        self.missing_context_adapters = nn.ModuleDict()
        self.missing_physiology_calibration_heads = nn.ModuleDict()
        for modality in self.modalities:
            self.patch_encoders[modality] = nn.Linear(self.patch_samples, decoder_dim)
            memory_layer = nn.TransformerEncoderLayer(
                d_model=decoder_dim,
                nhead=decoder_heads,
                dim_feedforward=decoder_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.memory_encoders[modality] = nn.TransformerEncoder(
                memory_layer, num_layers=decoder_layers
            )
            self.context_projections[modality] = nn.Sequential(
                nn.LayerNorm(shared_state_dim + dynamics_state_dim),
                nn.Linear(shared_state_dim + dynamics_state_dim, decoder_dim),
            )
            if self.missing_modality_conditioning:
                self.missing_memory_tokens[modality] = nn.Parameter(
                    torch.zeros(decoder_dim)
                )
                missing_adapter = nn.Sequential(
                    nn.LayerNorm(shared_state_dim + dynamics_state_dim),
                    nn.Linear(shared_state_dim + dynamics_state_dim, decoder_dim),
                )
                nn.init.zeros_(missing_adapter[-1].weight)
                nn.init.zeros_(missing_adapter[-1].bias)
                self.missing_context_adapters[modality] = missing_adapter
            if self.missing_physiology_calibration and modality in ("EEG", "EMG"):
                calibration_dim = (
                    self.patch_samples // 2 + 1 + 4 if modality == "EEG" else 2
                )
                calibration_head = nn.Sequential(
                    nn.LayerNorm(decoder_dim),
                    nn.Linear(decoder_dim, calibration_dim),
                )
                nn.init.zeros_(calibration_head[-1].weight)
                nn.init.zeros_(calibration_head[-1].bias)
                self.missing_physiology_calibration_heads[
                    modality
                ] = calibration_head
            query = nn.Parameter(torch.empty(self.num_patches, decoder_dim))
            nn.init.normal_(query, std=0.02)
            self.future_queries[modality] = query
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=decoder_dim,
                nhead=decoder_heads,
                dim_feedforward=decoder_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.decoders[modality] = nn.TransformerDecoder(
                decoder_layer, num_layers=decoder_layers
            )
            output_head = nn.Sequential(
                nn.LayerNorm(decoder_dim), nn.Linear(decoder_dim, self.patch_samples)
            )
            nn.init.zeros_(output_head[-1].weight)
            nn.init.zeros_(output_head[-1].bias)
            self.output_heads[modality] = output_head
            if self.structured_event_heads:
                if modality == "EEG":
                    structure_dim = self.patch_samples // 2 + 1 + 4
                elif modality == "ECG":
                    structure_dim = self.patch_samples + 1
                elif modality == "EMG":
                    structure_dim = 3
                else:
                    raise ValueError(f"unsupported structured waveform modality: {modality}")
                self.structure_heads[modality] = nn.Sequential(
                    nn.LayerNorm(decoder_dim), nn.Linear(decoder_dim, structure_dim)
                )
                self.structure_adapters[modality] = nn.Linear(
                    structure_dim, decoder_dim, bias=False
                )
                if self.probabilistic_event_heads:
                    if modality == "EEG":
                        probability_dim = 2 * structure_dim
                    elif modality == "ECG":
                        probability_dim = self.patch_samples + 2
                    else:
                        probability_dim = 5
                    self.probability_heads[modality] = nn.Sequential(
                        nn.LayerNorm(decoder_dim),
                        nn.Linear(decoder_dim, probability_dim),
                    )
                    self.probability_adapters[modality] = nn.Linear(
                        probability_dim, decoder_dim, bias=False
                    )
                    event_output = nn.Sequential(
                        nn.LayerNorm(decoder_dim),
                        nn.Linear(decoder_dim, self.patch_samples),
                    )
                    nn.init.zeros_(event_output[-1].weight)
                    nn.init.zeros_(event_output[-1].bias)
                    self.event_output_heads[modality] = event_output
                    if modality == "ECG" and self.ecg_refractory_event_enabled:
                        point_process_dim = self.patch_samples + self.ecg_rr_bins
                        self.ecg_point_process_head = nn.Sequential(
                            nn.LayerNorm(decoder_dim),
                            nn.Linear(decoder_dim, point_process_dim),
                        )
                        self.ecg_point_process_adapter = nn.Linear(
                            point_process_dim, decoder_dim, bias=False
                        )
                        self.ecg_point_process_output_head = nn.Sequential(
                            nn.LayerNorm(decoder_dim),
                            nn.Linear(decoder_dim, self.patch_samples),
                        )
                        nn.init.zeros_(self.ecg_point_process_output_head[-1].weight)
                        nn.init.zeros_(self.ecg_point_process_output_head[-1].bias)
                        if self.ecg_time_aligned_renderer:
                            self.ecg_event_timing_head = nn.Sequential(
                                nn.LayerNorm(decoder_dim),
                                nn.Linear(decoder_dim, 3),
                            )
                            nn.init.zeros_(self.ecg_event_timing_head[-1].weight)
                            nn.init.zeros_(self.ecg_event_timing_head[-1].bias)
                            if self.ecg_recent_rr_residual:
                                self.ecg_rr_residual_gate_head = nn.Sequential(
                                    nn.LayerNorm(decoder_dim),
                                    nn.Linear(decoder_dim, 1),
                                )
                                nn.init.zeros_(
                                    self.ecg_rr_residual_gate_head[-1].weight
                                )
                                nn.init.zeros_(
                                    self.ecg_rr_residual_gate_head[-1].bias
                                )
            if self.safe_modality_residual_refinement:
                if modality == "EEG":
                    refinement_dim = self.patch_samples // 2 + 1 + 4
                elif modality == "ECG":
                    refinement_dim = 1
                elif modality == "EMG":
                    refinement_dim = 3
                else:
                    raise ValueError(
                        f"unsupported safe waveform modality: {modality}"
                    )
                refinement_head = nn.Sequential(
                    nn.LayerNorm(decoder_dim),
                    nn.Linear(decoder_dim, refinement_dim),
                )
                nn.init.zeros_(refinement_head[-1].weight)
                nn.init.zeros_(refinement_head[-1].bias)
                self.safe_refinement_heads[modality] = refinement_head

    def missing_condition_parameters(self):
        yield from self.missing_memory_tokens.parameters()
        yield from self.missing_context_adapters.parameters()

    def missing_calibration_parameters(self):
        yield from self.missing_physiology_calibration_heads.parameters()

    def missing_teacher_calibration_parameters(self):
        if self.missing_emg_log_rms_bias is not None:
            yield self.missing_emg_log_rms_bias

    def forward(
        self,
        recent_waveforms: torch.Tensor,
        shared_future_state: torch.Tensor,
        modality_dynamics_states: torch.Tensor,
        return_structure: bool = False,
        return_probabilities: bool = False,
        stochastic_rendering: bool = False,
        modality_availability: Optional[torch.Tensor] = None,
    ) -> object:
        if recent_waveforms.ndim != 3:
            raise ValueError("recent waveforms must be [batch, modalities, samples]")
        if recent_waveforms.shape[1] != len(self.modalities):
            raise ValueError("recent waveform modality count is invalid")
        if recent_waveforms.shape[-1] < self.max_samples:
            raise ValueError("recent waveform does not cover the requested output duration")
        if shared_future_state.ndim != 2 or modality_dynamics_states.ndim != 3:
            raise ValueError("waveform state inputs have invalid dimensions")
        if modality_availability is None:
            modality_availability = recent_waveforms.new_ones(
                recent_waveforms.shape[0], len(self.modalities)
            )
        if modality_availability.shape != (
            recent_waveforms.shape[0],
            len(self.modalities),
        ):
            raise ValueError("waveform modality availability has invalid dimensions")
        recent = recent_waveforms[..., -self.max_samples :]
        positions = sinusoidal_positions(
            self.num_patches,
            next(iter(self.future_queries.values())).shape[-1],
            recent.device,
            recent.dtype,
        )
        causal_mask = torch.triu(
            torch.ones(
                self.num_patches,
                self.num_patches,
                dtype=torch.bool,
                device=recent.device,
            ),
            diagonal=1,
        )
        predictions = []
        structure_predictions: Dict[str, Dict[str, torch.Tensor]] = {}
        probability_predictions: Dict[str, Dict[str, torch.Tensor]] = {}
        for modality_index, modality in enumerate(self.modalities):
            patches = recent[:, modality_index].reshape(
                recent.shape[0], self.num_patches, self.patch_samples
            )
            context_input = torch.cat(
                (
                    shared_future_state,
                    modality_dynamics_states[:, modality_index],
                ),
                dim=-1,
            )
            missing_weight = (
                1.0 - modality_availability[:, modality_index].to(recent.dtype)
            ).unsqueeze(-1)
            memory = self.patch_encoders[modality](patches)
            if self.missing_modality_conditioning:
                memory = memory + missing_weight.unsqueeze(1) * self.missing_memory_tokens[
                    modality
                ].reshape(1, 1, -1)
            memory = self.memory_encoders[modality](memory + positions.unsqueeze(0))
            context = self.context_projections[modality](context_input)
            if self.missing_modality_conditioning:
                context = context + missing_weight * self.missing_context_adapters[
                    modality
                ](context_input)
            queries = self.future_queries[modality].unsqueeze(0) + context.unsqueeze(1)
            decoded = self.decoders[modality](queries, memory, tgt_mask=causal_mask)
            waveform_tokens = decoded
            if self.structured_event_heads:
                structure = self.structure_heads[modality](decoded)
                waveform_tokens = waveform_tokens + self.structure_adapters[modality](
                    structure
                )
                if modality == "EEG":
                    spectrum_bins = self.patch_samples // 2 + 1
                    structure_predictions[modality] = {
                        "log_spectrum": structure[..., :spectrum_bins],
                        "band_log_power": structure[..., spectrum_bins:],
                    }
                elif modality == "ECG":
                    structure_predictions[modality] = {
                        "qrs_logits": structure[..., : self.patch_samples].reshape(
                            recent.shape[0], self.max_samples
                        ),
                        "rr_seconds": 0.25 + F.softplus(structure[..., -1]),
                    }
                else:
                    structure_predictions[modality] = {
                        "envelope": F.softplus(structure[..., 0]),
                        "rms": F.softplus(structure[..., 1]),
                        "burst_logits": structure[..., 2],
                    }
            residual = self.output_heads[modality](waveform_tokens).reshape(
                recent.shape[0], self.max_samples
            )
            if self.probabilistic_event_heads:
                probability = self.probability_heads[modality](decoded)
                probability_tokens = decoded + self.probability_adapters[modality](
                    probability
                )
                residual = residual + self.event_output_heads[modality](
                    probability_tokens
                ).reshape(recent.shape[0], self.max_samples)
                if modality == "EEG":
                    descriptor_dim = self.patch_samples // 2 + 1 + 4
                    probability_predictions[modality] = {
                        "spectral_mean": probability[..., :descriptor_dim],
                        "spectral_scale": 1e-3
                        + F.softplus(probability[..., descriptor_dim:]),
                    }
                elif modality == "ECG":
                    probability_predictions[modality] = {
                        "qrs_logits": probability[..., : self.patch_samples].reshape(
                            recent.shape[0], self.max_samples
                        ),
                        "rr_mean_seconds": 0.25
                        + F.softplus(probability[..., self.patch_samples]),
                        "rr_scale_seconds": 1e-2
                        + F.softplus(probability[..., self.patch_samples + 1]),
                    }
                else:
                    probability_predictions[modality] = {
                        "envelope_mean": F.softplus(probability[..., 0]),
                        "envelope_scale": 1e-3 + F.softplus(probability[..., 1]),
                        "rms_mean": F.softplus(probability[..., 2]),
                        "rms_scale": 1e-3 + F.softplus(probability[..., 3]),
                        "burst_logits": probability[..., 4],
                    }
                if modality == "ECG" and self.ecg_refractory_event_enabled:
                    if (
                        self.ecg_point_process_head is None
                        or self.ecg_point_process_adapter is None
                        or self.ecg_point_process_output_head is None
                    ):
                        raise RuntimeError("ECG point-process modules were not initialized")
                    point_process = self.ecg_point_process_head(decoded)
                    qrs_patch_logits = point_process[..., : self.patch_samples]
                    rr_logits = point_process[..., self.patch_samples :]
                    rr_probability = rr_logits.softmax(dim=-1)
                    rr_centers = self.ecg_rr_bin_centers_seconds.to(
                        device=rr_logits.device, dtype=rr_logits.dtype
                    )
                    rr_mean = (rr_probability * rr_centers).sum(dim=-1)
                    rr_scale = (
                        rr_probability
                        * (rr_centers - rr_mean.unsqueeze(-1)).square()
                    ).sum(dim=-1).clamp_min(1e-6).sqrt()
                    point_process_descriptor = torch.cat(
                        (qrs_patch_logits.sigmoid(), rr_probability), dim=-1
                    )
                    point_process_tokens = decoded + self.ecg_point_process_adapter(
                        point_process_descriptor
                    )
                    residual = residual + self.ecg_point_process_output_head(
                        point_process_tokens
                    ).reshape(recent.shape[0], self.max_samples)
                    probability_predictions[modality] = {
                        "qrs_logits": qrs_patch_logits.reshape(
                            recent.shape[0], self.max_samples
                        ),
                        "rr_logits": rr_logits,
                        "rr_mean_seconds": rr_mean,
                        "rr_scale_seconds": rr_scale,
                    }
                    if self.ecg_time_aligned_renderer:
                        if self.ecg_event_timing_head is None:
                            raise RuntimeError("ECG timing head was not initialized")
                        event_timing = self.ecg_event_timing_head(decoded)
                        probability_predictions[modality].update(
                            {
                                "event_hazard_logits": event_timing[..., 0],
                                "event_offset_samples": 0.5
                                * self.patch_samples
                                * event_timing[..., 1].tanh(),
                                "event_amplitude": event_timing[..., 2]
                                .clamp(-1.0, 1.0)
                                .exp(),
                            }
                        )
                        if self.ecg_recent_rr_residual:
                            if self.ecg_rr_residual_gate_head is None:
                                raise RuntimeError(
                                    "ECG RR residual gate was not initialized"
                                )
                            probability_predictions[modality][
                                "rr_residual_gate_logits"
                            ] = self.ecg_rr_residual_gate_head(decoded).squeeze(-1)
                if self.safe_modality_residual_refinement:
                    refinement = self.safe_refinement_heads[modality](decoded)
                    if modality == "EEG":
                        bounded = 0.25 * refinement.tanh()
                        probability_predictions[modality][
                            "spectral_mean"
                        ] = probability_predictions[modality]["spectral_mean"] + bounded
                        probability_predictions[modality][
                            "safe_spectral_residual"
                        ] = bounded
                    elif modality == "ECG":
                        timing = (
                            0.12
                            * self.sample_rate
                            * refinement[..., 0].tanh()
                        )
                        probability_predictions[modality][
                            "local_timing_residual_samples"
                        ] = timing
                    else:
                        envelope_scale = (0.5 * refinement[..., 0].tanh()).exp()
                        rms_scale = (0.5 * refinement[..., 1].tanh()).exp()
                        burst_residual = refinement[..., 2].tanh()
                        probability_predictions[modality][
                            "envelope_mean"
                        ] = probability_predictions[modality][
                            "envelope_mean"
                        ] * envelope_scale
                        probability_predictions[modality][
                            "rms_mean"
                        ] = probability_predictions[modality]["rms_mean"] * rms_scale
                        probability_predictions[modality][
                            "burst_logits"
                        ] = probability_predictions[modality][
                            "burst_logits"
                        ] + burst_residual
                        probability_predictions[modality][
                            "safe_envelope_log_scale"
                        ] = envelope_scale.log()
                        probability_predictions[modality][
                            "safe_rms_log_scale"
                        ] = rms_scale.log()
                        probability_predictions[modality][
                            "safe_burst_residual"
                        ] = burst_residual
                if (
                    self.missing_physiology_calibration
                    and modality in self.missing_physiology_calibration_heads
                ):
                    calibration = self.missing_physiology_calibration_heads[
                        modality
                    ](decoded)
                    calibration = 1.5 * calibration.tanh() * missing_weight.unsqueeze(1)
                    if modality == "EEG":
                        probability_predictions[modality][
                            "spectral_mean"
                        ] = probability_predictions[modality][
                            "spectral_mean"
                        ] + calibration
                        probability_predictions[modality][
                            "missing_log_spectral_energy_shift"
                        ] = calibration
                    else:
                        envelope_scale = calibration[..., 0].exp()
                        rms_scale = calibration[..., 1].exp()
                        probability_predictions[modality][
                            "envelope_mean"
                        ] = probability_predictions[modality][
                            "envelope_mean"
                        ] * envelope_scale
                        probability_predictions[modality][
                            "rms_mean"
                        ] = probability_predictions[modality][
                            "rms_mean"
                        ] * rms_scale
                        probability_predictions[modality][
                            "missing_envelope_log_scale"
                        ] = calibration[..., 0]
                        probability_predictions[modality][
                            "missing_rms_log_scale"
                        ] = calibration[..., 1]
                if modality == "EMG" and self.missing_emg_log_rms_bias is not None:
                    eeg_available = modality_availability[
                        :, self.modalities.index("EEG")
                    ] >= 0.5
                    ecg_available = modality_availability[
                        :, self.modalities.index("ECG")
                    ] >= 0.5
                    condition_index = (
                        eeg_available.to(torch.long)
                        + 2 * ecg_available.to(torch.long)
                        - 1
                    ).clamp(0, 2)
                    teacher_log_scale = self.missing_emg_log_rms_bias[
                        condition_index
                    ].clamp(
                        -1.38629436112, 1.38629436112
                    ).unsqueeze(-1) * missing_weight
                    probability_predictions[modality][
                        "rms_mean"
                    ] = probability_predictions[modality][
                        "rms_mean"
                    ] * teacher_log_scale.exp()
                    probability_predictions[modality][
                        "missing_teacher_rms_log_scale"
                    ] = teacher_log_scale.expand(-1, self.num_patches)
            if self.physiological_event_renderer:
                if modality == "EEG":
                    rendered = self._render_eeg_spectrum(
                        probability_predictions[modality],
                        patches,
                        stochastic=stochastic_rendering,
                    )
                elif modality == "ECG":
                    if self.ecg_time_aligned_renderer:
                        if self.ecg_recursive_event_renderer:
                            rendered = self._render_recursive_ecg_events(
                                probability_predictions[modality],
                                recent[:, modality_index],
                            )
                        else:
                            rendered = self._render_time_aligned_ecg_events(
                                probability_predictions[modality],
                                recent[:, modality_index],
                            )
                    else:
                        rendered = self._render_ecg_events(
                            probability_predictions[modality],
                            recent[:, modality_index],
                        )
                else:
                    rendered = self._render_emg_envelope(
                        probability_predictions[modality],
                        residual,
                        stochastic=stochastic_rendering,
                    )
                residual_scale = (
                    0.0
                    if modality == "EEG"
                    or (modality == "ECG" and self.ecg_time_aligned_renderer)
                    else 0.1
                )
                predictions.append(rendered + residual_scale * residual)
            else:
                baseline = (
                    recent[:, modality_index]
                    if self.output_baseline == "repeat"
                    else torch.zeros_like(residual)
                )
                predictions.append(baseline + residual)
        waveforms = torch.stack(predictions, dim=1)
        if return_probabilities:
            if not self.probabilistic_event_heads:
                raise ValueError("probabilistic waveform outputs are not enabled")
            if return_structure:
                return waveforms, structure_predictions, probability_predictions
            return waveforms, probability_predictions
        if return_structure:
            if not self.structured_event_heads:
                raise ValueError("structured waveform outputs are not enabled")
            return waveforms, structure_predictions
        return waveforms

    def _render_eeg_spectrum(
        self,
        probability: Dict[str, torch.Tensor],
        recent_patches: torch.Tensor,
        stochastic: bool = False,
    ) -> torch.Tensor:
        frequency_bins = self.patch_samples // 2 + 1
        hop = self.patch_samples // 2
        window_count = (self.max_samples - self.patch_samples) // hop + 1
        log_amplitude = probability["spectral_mean"][..., :frequency_bins]
        log_amplitude = F.interpolate(
            log_amplitude.transpose(1, 2),
            size=window_count,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
        amplitude = torch.expm1(log_amplitude.clamp(0.0, 4.0))
        amplitude = amplitude.clone()
        amplitude[..., 0] = 0.0
        recent = recent_patches.reshape(recent_patches.shape[0], self.max_samples)
        recent_windows = recent.unfold(-1, self.patch_samples, hop)
        if stochastic:
            phase = 2.0 * torch.pi * torch.rand_like(amplitude) - torch.pi
            phase[..., 0] = 0.0
            if self.patch_samples % 2 == 0:
                phase[..., -1] = 0.0
        else:
            recent_spectrum = torch.fft.rfft(recent_windows, dim=-1, norm="ortho")
            phase = torch.angle(recent_spectrum)
        synthesized_windows = torch.fft.irfft(
            torch.polar(amplitude, phase),
            n=self.patch_samples,
            dim=-1,
            norm="ortho",
        )
        window = torch.hann_window(
            self.patch_samples,
            periodic=False,
            device=recent.device,
            dtype=recent.dtype,
        )
        output = torch.zeros_like(recent)
        denominator = torch.zeros_like(recent)
        for index in range(window_count):
            start = index * hop
            stop = start + self.patch_samples
            output[:, start:stop] += synthesized_windows[:, index] * window
            denominator[:, start:stop] += window
        return output / denominator.clamp_min(1e-3)

    def _render_ecg_events(
        self,
        probability: Dict[str, torch.Tensor],
        recent: torch.Tensor,
    ) -> torch.Tensor:
        events = self._qrs_events(recent, self.sample_rate)
        logits = probability["qrs_logits"]
        rr_seconds = probability["rr_mean_seconds"]
        generated = torch.zeros_like(recent)
        left = max(1, round(0.20 * self.sample_rate))
        right = max(1, round(0.60 * self.sample_rate))
        tolerance = max(1, round(0.12 * self.sample_rate))
        for batch_index in range(recent.shape[0]):
            positions = events[batch_index].nonzero(as_tuple=False).flatten()
            beat_segments = []
            for position in positions[-6:]:
                start = int(position) - left
                stop = int(position) + right
                if start >= 0 and stop <= recent.shape[-1]:
                    segment = recent[batch_index, start:stop]
                    edge = 0.5 * (segment[0] + segment[-1])
                    beat_segments.append(segment - edge)
            if beat_segments:
                template = torch.stack(beat_segments).median(dim=0).values
            else:
                sample_positions = torch.arange(
                    -left,
                    right,
                    device=recent.device,
                    dtype=recent.dtype,
                )
                template = 4.0 * torch.exp(
                    -0.5 * (sample_positions / max(1.0, 0.025 * self.sample_rate)).square()
                )
                template = template - 1.2 * torch.exp(
                    -0.5
                    * (
                        (sample_positions - 0.04 * self.sample_rate)
                        / max(1.0, 0.025 * self.sample_rate)
                    ).square()
                )
            if len(positions) >= 2:
                recent_rr = float(
                    (positions[1:] - positions[:-1]).float().median().detach()
                )
            else:
                recent_rr = float(self.sample_rate)
            predicted_rr = float(
                rr_seconds[batch_index].median().detach() * self.sample_rate
            )
            rr_samples = max(
                round(0.25 * self.sample_rate),
                int(round(0.5 * (recent_rr + predicted_rr))),
            )
            last_event = int(positions[-1]) if len(positions) else recent.shape[-1] - rr_samples
            next_event = last_event + rr_samples - recent.shape[-1]
            while next_event < 0:
                next_event += rr_samples
            while next_event < self.max_samples:
                start_search = max(0, next_event - tolerance)
                stop_search = min(self.max_samples, next_event + tolerance + 1)
                if stop_search > start_search:
                    offset = int(
                        logits[batch_index, start_search:stop_search].argmax().detach()
                    )
                    event_position = start_search + offset
                else:
                    event_position = next_event
                timing_residual = probability.get("local_timing_residual_samples")
                if timing_residual is None:
                    start = max(0, event_position - left)
                    stop = min(self.max_samples, event_position + right)
                    template_start = start - (event_position - left)
                    template_stop = template_start + (stop - start)
                    generated[batch_index, start:stop] += template[
                        template_start:template_stop
                    ]
                else:
                    patch_index = min(
                        event_position // self.patch_samples,
                        self.num_patches - 1,
                    )
                    shift = timing_residual[batch_index, patch_index]
                    margin = tolerance
                    start = max(0, event_position - left - margin)
                    stop = min(self.max_samples, event_position + right + margin)
                    output_positions = torch.arange(
                        start,
                        stop,
                        device=recent.device,
                        dtype=recent.dtype,
                    )
                    coordinates = (
                        output_positions - float(event_position) - shift + left
                    )
                    valid_coordinates = (coordinates >= 0.0) & (
                        coordinates <= template.shape[0] - 1
                    )
                    lower = coordinates.floor().long().clamp(
                        0, template.shape[0] - 1
                    )
                    upper = (lower + 1).clamp(0, template.shape[0] - 1)
                    fraction = coordinates - lower.to(coordinates.dtype)
                    shifted_template = (
                        (1.0 - fraction) * template[lower]
                        + fraction * template[upper]
                    ) * valid_coordinates.to(template.dtype)
                    generated[batch_index, start:stop] += shifted_template
                next_event += rr_samples
        return generated

    def _recent_ecg_templates(self, recent: torch.Tensor) -> torch.Tensor:
        events = self._qrs_events(recent, self.sample_rate)
        left = max(1, round(0.20 * self.sample_rate))
        right = max(1, round(0.60 * self.sample_rate))
        templates = []
        for batch_index in range(recent.shape[0]):
            positions = events[batch_index].nonzero(as_tuple=False).flatten()
            beat_segments = []
            for position in positions[-6:]:
                start = int(position) - left
                stop = int(position) + right
                if start >= 0 and stop <= recent.shape[-1]:
                    segment = recent[batch_index, start:stop]
                    edge = 0.5 * (segment[0] + segment[-1])
                    beat_segments.append(segment - edge)
            if beat_segments:
                template = torch.stack(beat_segments).median(dim=0).values
            else:
                positions_from_qrs = torch.arange(
                    -left,
                    right,
                    device=recent.device,
                    dtype=recent.dtype,
                )
                template = 4.0 * torch.exp(
                    -0.5
                    * (
                        positions_from_qrs / max(1.0, 0.025 * self.sample_rate)
                    ).square()
                )
                template = template - 1.2 * torch.exp(
                    -0.5
                    * (
                        (positions_from_qrs - 0.04 * self.sample_rate)
                        / max(1.0, 0.025 * self.sample_rate)
                    ).square()
                )
            templates.append(template)
        return torch.stack(templates)

    def _render_time_aligned_ecg_events(
        self,
        probability: Dict[str, torch.Tensor],
        recent: torch.Tensor,
    ) -> torch.Tensor:
        qrs_logits = probability["qrs_logits"].reshape(
            recent.shape[0], self.num_patches, self.patch_samples
        )
        local_samples = torch.arange(
            self.patch_samples, device=recent.device, dtype=recent.dtype
        )
        local_centers = (
            (qrs_logits / 0.25).softmax(dim=-1) * local_samples
        ).sum(dim=-1)
        patch_starts = self.patch_samples * torch.arange(
            self.num_patches, device=recent.device, dtype=recent.dtype
        )
        centers = (
            patch_starts.unsqueeze(0)
            + local_centers
            + probability["event_offset_samples"]
        )
        sample_positions = torch.arange(
            self.max_samples, device=recent.device, dtype=recent.dtype
        )
        sigma = max(1.0, self.ecg_event_sigma_seconds * self.sample_rate)
        pulses = torch.exp(
            -0.5
            * (
                (sample_positions.view(1, 1, -1) - centers.unsqueeze(-1)) / sigma
            ).square()
        )
        pulses = pulses / pulses.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        event_weight = (
            probability["event_hazard_logits"].sigmoid()
            * probability["event_amplitude"]
        )
        event_map = (event_weight.unsqueeze(-1) * pulses).sum(dim=1)

        templates = self._recent_ecg_templates(recent)
        template_samples = templates.shape[-1]
        left = max(1, round(0.20 * self.sample_rate))
        full = F.conv1d(
            F.pad(event_map.unsqueeze(0), (template_samples - 1,) * 2),
            templates.flip(-1).unsqueeze(1),
            groups=recent.shape[0],
        )
        return full[0, :, left : left + self.max_samples]

    def _recursive_ecg_event_geometry(
        self,
        probability: Dict[str, torch.Tensor],
        recent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rr_probability = probability["rr_logits"].softmax(dim=-1)
        rr_centers = self.ecg_rr_bin_centers_seconds.to(
            device=recent.device, dtype=recent.dtype
        )
        rr_samples = (rr_probability * rr_centers).sum(dim=-1) * self.sample_rate
        recent_events = self._qrs_events(recent, self.sample_rate)
        elapsed = []
        recent_rr = []
        for row in recent_events:
            positions = row.nonzero(as_tuple=False).flatten()
            if len(positions):
                elapsed.append(
                    recent.new_tensor(recent.shape[-1] - 1 - int(positions[-1]))
                )
                if len(positions) >= 2:
                    recent_rr.append(
                        (positions[1:] - positions[:-1]).float().median()
                    )
                else:
                    recent_rr.append(rr_samples[len(recent_rr), 0].detach())
            else:
                elapsed.append(rr_samples[len(elapsed), 0].detach())
                recent_rr.append(rr_samples[len(recent_rr), 0].detach())
        elapsed_samples = torch.stack(elapsed)
        if self.ecg_recent_rr_residual:
            gate = probability["rr_residual_gate_logits"].sigmoid()
            recent_rr_samples = torch.stack(recent_rr).unsqueeze(-1)
            rr_samples = recent_rr_samples + gate * (
                rr_samples - recent_rr_samples
            )
        base_centers = rr_samples.cumsum(dim=-1) - elapsed_samples.unsqueeze(-1)
        search_centers = base_centers + probability["event_offset_samples"]

        qrs_logits = probability["qrs_logits"]
        sample_positions = torch.arange(
            self.max_samples, device=recent.device, dtype=recent.dtype
        )
        tolerance = max(1.0, 0.12 * self.sample_rate)
        local_scores = qrs_logits.unsqueeze(1) / 0.5 - 0.5 * (
            (
                sample_positions.view(1, 1, -1)
                - search_centers.unsqueeze(-1)
            )
            / tolerance
        ).square()
        local_attention = local_scores.softmax(dim=-1)
        aligned_centers = (local_attention * sample_positions).sum(dim=-1)
        return aligned_centers, base_centers

    def _render_recursive_ecg_events(
        self,
        probability: Dict[str, torch.Tensor],
        recent: torch.Tensor,
    ) -> torch.Tensor:
        centers, base_centers = self._recursive_ecg_event_geometry(
            probability, recent
        )
        sample_positions = torch.arange(
            self.max_samples, device=recent.device, dtype=recent.dtype
        )
        sigma = max(1.0, self.ecg_event_sigma_seconds * self.sample_rate)
        pulses = torch.exp(
            -0.5
            * (
                (sample_positions.view(1, 1, -1) - centers.unsqueeze(-1)) / sigma
            ).square()
        )
        if self.ecg_recent_rr_residual:
            pulses = pulses / pulses.square().sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-6).sqrt()
        else:
            pulses = pulses / pulses.sum(dim=-1, keepdim=True).clamp_min(1.0)
        hazard_probability = probability["event_hazard_logits"].sigmoid()
        hard_presence = (hazard_probability >= 0.5).to(hazard_probability.dtype)
        event_presence = (
            hard_presence.detach()
            - hazard_probability.detach()
            + hazard_probability
        )
        in_range = (
            (base_centers >= -3.0 * sigma)
            & (base_centers < self.max_samples + 3.0 * sigma)
        ).to(event_presence.dtype)
        if self.ecg_recent_rr_residual:
            event_weight = in_range
        else:
            event_weight = event_presence * in_range * probability["event_amplitude"]
        event_map = (event_weight.unsqueeze(-1) * pulses).sum(dim=1)

        templates = self._recent_ecg_templates(recent)
        template_samples = templates.shape[-1]
        left = max(1, round(0.20 * self.sample_rate))
        full = F.conv1d(
            F.pad(event_map.unsqueeze(0), (template_samples - 1,) * 2),
            templates.flip(-1).unsqueeze(1),
            groups=recent.shape[0],
        )
        generated = full[0, :, left : left + self.max_samples]
        if self.ecg_recent_amplitude_calibration:
            recent_rms = recent.square().mean(dim=-1).clamp_min(1e-8).sqrt()
            generated_rms = generated.square().mean(dim=-1).clamp_min(1e-8).sqrt()
            scale = (recent_rms / generated_rms).clamp(0.25, 4.0)
            generated = generated * scale.unsqueeze(-1)
        return generated

    def _render_emg_envelope(
        self,
        probability: Dict[str, torch.Tensor],
        residual: torch.Tensor,
        stochastic: bool = False,
    ) -> torch.Tensor:
        if stochastic:
            carrier = torch.randn_like(residual)
            smooth = F.avg_pool1d(
                carrier.unsqueeze(1), kernel_size=9, stride=1, padding=4
            ).squeeze(1)
            carrier = carrier - smooth
        else:
            carrier = residual
        carrier = carrier.reshape(-1, self.num_patches, self.patch_samples)
        carrier = carrier - carrier.mean(dim=-1, keepdim=True)
        carrier_rms = carrier.square().mean(dim=-1, keepdim=True).clamp_min(1e-6).sqrt()
        carrier = carrier / carrier_rms
        scale = probability["rms_mean"].unsqueeze(-1)
        burst = 0.75 + 0.5 * probability["burst_logits"].sigmoid().unsqueeze(-1)
        return (carrier * scale * burst).reshape(-1, self.max_samples)

    def sample_waveform_distribution(
        self,
        recent_waveforms: torch.Tensor,
        shared_future_state: torch.Tensor,
        modality_dynamics_states: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        if int(num_samples) < 1:
            raise ValueError("waveform sample count must be positive")
        draws = []
        for _ in range(int(num_samples)):
            output = self(
                recent_waveforms,
                shared_future_state,
                modality_dynamics_states,
                stochastic_rendering=True,
            )
            if isinstance(output, tuple):
                output = output[0]
            draws.append(output)
        return torch.stack(draws, dim=1)

    @staticmethod
    def _envelope(signals: torch.Tensor, kernel: int, derivative: bool) -> torch.Tensor:
        values = signals[..., 1:] - signals[..., :-1] if derivative else signals
        values = values.square() if derivative else values.abs()
        kernel = max(1, min(int(kernel), values.shape[-1]))
        return F.avg_pool1d(
            values.reshape(-1, 1, values.shape[-1]),
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        )[..., : values.shape[-1]].reshape(*values.shape)

    @classmethod
    def _qrs_events(cls, signals: torch.Tensor, sample_rate: int) -> torch.Tensor:
        envelope = cls._envelope(signals, round(0.12 * sample_rate), derivative=True)
        threshold = envelope.mean(dim=-1, keepdim=True) + envelope.std(
            dim=-1, keepdim=True
        )
        refractory = max(3, round(0.25 * sample_rate))
        if refractory % 2 == 0:
            refractory += 1
        local_max = F.max_pool1d(
            envelope.reshape(-1, 1, envelope.shape[-1]),
            kernel_size=refractory,
            stride=1,
            padding=refractory // 2,
        ).reshape_as(envelope)
        events = (envelope >= local_max) & (envelope > threshold)
        return F.pad(events, (1, 0), value=False)

    def _band_log_power(self, patches: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(patches, dim=-1, norm="ortho")
        power = spectrum.real.square() + spectrum.imag.square()
        frequencies = torch.fft.rfftfreq(
            self.patch_samples,
            d=1.0 / self.sample_rate,
            device=patches.device,
        )
        bands = ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0))
        values = []
        for low, high in bands:
            selected = (frequencies >= low) & (frequencies < high)
            if selected.any():
                values.append(torch.log1p(power[..., selected].mean(dim=-1)))
            else:
                values.append(power.sum(dim=-1) * 0.0)
        return torch.stack(values, dim=-1)

    @staticmethod
    def _rr_targets(
        events: torch.Tensor, sample_rate: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        targets = []
        valid = []
        for row in events:
            positions = row.nonzero(as_tuple=False).flatten()
            if len(positions) >= 2:
                targets.append(
                    (positions[1:] - positions[:-1]).float().median() / sample_rate
                )
                valid.append(True)
            else:
                targets.append(row.float().sum() * 0.0)
                valid.append(False)
        return torch.stack(targets), torch.tensor(valid, device=events.device)

    @staticmethod
    def _multi_resolution_spectral_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        fft_sizes: Sequence[int],
    ) -> torch.Tensor:
        losses = []
        sizes = sorted(
            {
                max(4, min(int(size), prediction.shape[-1]))
                for size in fft_sizes
                if int(size) >= 4
            }
        )
        for size in sizes:
            window = torch.hann_window(
                size, device=prediction.device, dtype=prediction.dtype
            )
            hop = max(1, size // 4)
            predicted_spectrum = torch.log1p(
                torch.stft(
                    prediction,
                    n_fft=size,
                    hop_length=hop,
                    win_length=size,
                    window=window,
                    center=True,
                    normalized=True,
                    return_complex=True,
                ).abs()
            )
            target_spectrum = torch.log1p(
                torch.stft(
                    target,
                    n_fft=size,
                    hop_length=hop,
                    win_length=size,
                    window=window,
                    center=True,
                    normalized=True,
                    return_complex=True,
                ).abs()
            )
            losses.append(F.smooth_l1_loss(predicted_spectrum, target_spectrum))
        if not losses:
            raise ValueError("multi-resolution spectral loss has no valid FFT size")
        return torch.stack(losses).mean()

    def _structured_auxiliary_loss(
        self,
        modality: str,
        predictions: Dict[str, torch.Tensor],
        target: torch.Tensor,
        samples: int,
    ) -> torch.Tensor:
        if samples % self.patch_samples:
            raise ValueError("structured waveform horizon must align to patches")
        patch_count = samples // self.patch_samples
        target_patches = target.reshape(
            target.shape[0], patch_count, self.patch_samples
        )
        if modality == "EEG":
            centered = target_patches - target_patches.mean(dim=-1, keepdim=True)
            target_spectrum = torch.log1p(
                torch.fft.rfft(centered, dim=-1, norm="ortho").abs()
            )
            spectrum_loss = F.smooth_l1_loss(
                predictions["log_spectrum"][:, :patch_count], target_spectrum
            )
            band_loss = F.smooth_l1_loss(
                predictions["band_log_power"][:, :patch_count],
                self._band_log_power(centered),
            )
            return 0.5 * (spectrum_loss + band_loss)
        if modality == "ECG":
            target_events = self._qrs_events(target, self.sample_rate)
            logits = predictions["qrs_logits"][:, :samples]
            positives = target_events.sum().to(logits.dtype)
            negatives = target_events.numel() - positives
            positive_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 20.0)
            event_loss = F.binary_cross_entropy_with_logits(
                logits, target_events.to(logits.dtype), pos_weight=positive_weight
            )
            rr_target, rr_valid = self._rr_targets(target_events, self.sample_rate)
            if rr_valid.any():
                rr_prediction = predictions["rr_seconds"][:, :patch_count].mean(dim=1)
                rr_loss = F.smooth_l1_loss(
                    rr_prediction[rr_valid], rr_target[rr_valid]
                )
                return 0.5 * (event_loss + rr_loss)
            return event_loss
        envelope = self._envelope(
            target, round(0.25 * self.sample_rate), derivative=False
        ).reshape(target.shape[0], patch_count, self.patch_samples).mean(dim=-1)
        rms = target_patches.square().mean(dim=-1).clamp_min(1e-8).sqrt()
        burst_threshold = envelope.mean(dim=-1, keepdim=True) + envelope.std(
            dim=-1, keepdim=True
        )
        burst = (envelope > burst_threshold).to(target.dtype)
        envelope_loss = F.smooth_l1_loss(
            predictions["envelope"][:, :patch_count], envelope
        )
        rms_loss = F.smooth_l1_loss(predictions["rms"][:, :patch_count], rms)
        burst_loss = F.binary_cross_entropy_with_logits(
            predictions["burst_logits"][:, :patch_count], burst
        )
        return (envelope_loss + rms_loss + burst_loss) / 3.0

    def waveform_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        horizons_seconds: Sequence[int],
        time_weight: float = 1.0,
        spectral_weight: float = 0.25,
        structure_weight: float = 0.25,
        structure_predictions: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        multi_resolution_fft_sizes: Sequence[int] = (),
        auxiliary_structure_weight: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        if prediction.shape != target.shape or prediction.ndim != 3:
            raise ValueError("waveform prediction and target shapes must match")
        if valid.shape != prediction.shape[:2]:
            raise ValueError("waveform validity must be [batch, modalities]")
        time_losses = []
        spectral_losses = []
        structure_losses = []
        auxiliary_losses = []
        for seconds in horizons_seconds:
            samples = int(seconds) * self.sample_rate
            if samples < 1 or samples > self.max_samples:
                raise ValueError("waveform horizon exceeds decoder output")
            for modality_index, modality in enumerate(self.modalities):
                selected = valid[:, modality_index]
                if not selected.any():
                    continue
                predicted_segment = prediction[selected, modality_index, :samples]
                target_segment = target[selected, modality_index, :samples]
                time_losses.append(F.smooth_l1_loss(predicted_segment, target_segment))
                if multi_resolution_fft_sizes:
                    spectral_losses.append(
                        self._multi_resolution_spectral_loss(
                            predicted_segment,
                            target_segment,
                            multi_resolution_fft_sizes,
                        )
                    )
                else:
                    predicted_spectrum = torch.log1p(
                        torch.fft.rfft(predicted_segment, dim=-1, norm="ortho").abs()
                    )
                    target_spectrum = torch.log1p(
                        torch.fft.rfft(target_segment, dim=-1, norm="ortho").abs()
                    )
                    spectral_losses.append(
                        F.smooth_l1_loss(predicted_spectrum, target_spectrum)
                    )
                if modality == "ECG":
                    predicted_structure = self._envelope(
                        predicted_segment, round(0.12 * self.sample_rate), True
                    )
                    target_structure = self._envelope(
                        target_segment, round(0.12 * self.sample_rate), True
                    )
                    structure_losses.append(
                        F.smooth_l1_loss(predicted_structure, target_structure)
                    )
                elif modality == "EMG":
                    predicted_structure = self._envelope(
                        predicted_segment, round(0.25 * self.sample_rate), False
                    )
                    target_structure = self._envelope(
                        target_segment, round(0.25 * self.sample_rate), False
                    )
                    structure_losses.append(
                        F.smooth_l1_loss(predicted_structure, target_structure)
                    )
                if structure_predictions is not None:
                    auxiliary_losses.append(
                        self._structured_auxiliary_loss(
                            modality,
                            {
                                key: value[selected]
                                for key, value in structure_predictions[modality].items()
                            },
                            target_segment,
                            samples,
                        )
                    )
        if not time_losses or not spectral_losses:
            raise ValueError("waveform loss has no valid targets")
        time_loss = torch.stack(time_losses).mean()
        spectral_loss = torch.stack(spectral_losses).mean()
        structure_loss = (
            torch.stack(structure_losses).mean()
            if structure_losses
            else prediction.sum() * 0.0
        )
        auxiliary_loss = (
            torch.stack(auxiliary_losses).mean()
            if auxiliary_losses
            else prediction.sum() * 0.0
        )
        total = (
            float(time_weight) * time_loss
            + float(spectral_weight) * spectral_loss
            + float(structure_weight) * structure_loss
            + float(auxiliary_structure_weight) * auxiliary_loss
        )
        return {
            "loss": total,
            "time_loss": time_loss,
            "spectral_loss": spectral_loss,
            "structure_loss": structure_loss,
            "auxiliary_loss": auxiliary_loss,
        }


class ShortHorizonWaveformWorldModel(PhysiologyStateSpaceWorldModel):
    """Frozen-F5-compatible model with a modality-specific short waveform decoder."""

    uses_short_waveform_decoder = True

    def __init__(
        self,
        *args,
        waveform_seconds: int = 10,
        waveform_patch_samples: int = 64,
        waveform_decoder_dim: int = 64,
        waveform_decoder_layers: int = 1,
        waveform_decoder_heads: int = 4,
        waveform_structured_event_heads: bool = False,
        waveform_probabilistic_event_heads: bool = False,
        waveform_ecg_refractory_event_head: bool = False,
        waveform_ecg_rr_bins: int = 48,
        waveform_output_baseline: str = "repeat",
        waveform_physiological_event_renderer: bool = False,
        waveform_ecg_time_aligned_renderer: bool = False,
        waveform_ecg_recursive_event_renderer: bool = False,
        waveform_ecg_recent_rr_residual: bool = False,
        waveform_ecg_recent_amplitude_calibration: bool = False,
        waveform_safe_modality_residual_refinement: bool = False,
        waveform_missing_modality_conditioning: bool = False,
        waveform_missing_physiology_calibration: bool = False,
        waveform_missing_emg_teacher_calibration: bool = False,
        waveform_ecg_event_sigma_seconds: float = 0.02,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        dynamics_dim = next(
            iter(self.physiology_dynamics_horizon_embeddings.values())
        ).shape[-1]
        self.waveform_decoder = ShortHorizonWaveformDecoder(
            self.encoder.config.modalities,
            self.encoder.config.d_model,
            dynamics_dim,
            self.encoder.config.sample_rate,
            waveform_seconds=waveform_seconds,
            patch_samples=waveform_patch_samples,
            decoder_dim=waveform_decoder_dim,
            decoder_layers=waveform_decoder_layers,
            decoder_heads=waveform_decoder_heads,
            dropout=float(self.encoder.config.dropout),
            structured_event_heads=waveform_structured_event_heads,
            probabilistic_event_heads=waveform_probabilistic_event_heads,
            ecg_refractory_event_head=waveform_ecg_refractory_event_head,
            ecg_rr_bins=waveform_ecg_rr_bins,
            output_baseline=waveform_output_baseline,
            physiological_event_renderer=waveform_physiological_event_renderer,
            ecg_time_aligned_renderer=waveform_ecg_time_aligned_renderer,
            ecg_recursive_event_renderer=waveform_ecg_recursive_event_renderer,
            ecg_recent_rr_residual=waveform_ecg_recent_rr_residual,
            ecg_recent_amplitude_calibration=waveform_ecg_recent_amplitude_calibration,
            safe_modality_residual_refinement=waveform_safe_modality_residual_refinement,
            missing_modality_conditioning=waveform_missing_modality_conditioning,
            missing_physiology_calibration=waveform_missing_physiology_calibration,
            missing_emg_teacher_calibration=waveform_missing_emg_teacher_calibration,
            ecg_event_sigma_seconds=waveform_ecg_event_sigma_seconds,
        )

    def rollout_context(
        self,
        history_signals: torch.Tensor,
        history_present: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return super().rollout(history_signals, history_present)

    def rollout(
        self,
        history_signals: torch.Tensor,
        history_present: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        output = self.rollout_context(history_signals, history_present)
        modality_availability = (
            history_signals.new_ones(
                history_signals.shape[0], history_signals.shape[2]
            )
            if history_present is None
            else history_present.to(history_signals.dtype).mean(dim=1)
        )
        decoded = self.waveform_decoder(
            history_signals[:, -1],
            output["predicted_states"][:, 0],
            output["physiology_dynamics_states"][:, 0],
            return_structure=self.waveform_decoder.structured_event_heads,
            return_probabilities=self.waveform_decoder.probabilistic_event_heads,
            modality_availability=modality_availability,
        )
        if self.waveform_decoder.probabilistic_event_heads:
            (
                output["future_waveforms"],
                output["future_waveform_structure"],
                output["future_waveform_probabilities"],
            ) = decoded
        elif self.waveform_decoder.structured_event_heads:
            output["future_waveforms"], output["future_waveform_structure"] = decoded
        else:
            output["future_waveforms"] = decoded
        return output

    def forward(
        self,
        *args,
        future_waveform_targets: Optional[torch.Tensor] = None,
        future_waveform_valid: Optional[torch.Tensor] = None,
        waveform_horizons_seconds: Sequence[int] = (1, 5, 10),
        waveform_weight: float = 1.0,
        waveform_time_weight: float = 1.0,
        waveform_spectral_weight: float = 0.25,
        waveform_structure_weight: float = 0.25,
        waveform_auxiliary_structure_weight: float = 0.0,
        waveform_multi_resolution_fft_sizes: Sequence[int] = (),
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        output = super().forward(*args, **kwargs)
        if future_waveform_targets is not None:
            if future_waveform_valid is None:
                raise ValueError("future waveform targets require a validity mask")
            waveform_losses = self.waveform_decoder.waveform_loss(
                output["future_waveforms"],
                future_waveform_targets,
                future_waveform_valid,
                waveform_horizons_seconds,
                waveform_time_weight,
                waveform_spectral_weight,
                waveform_structure_weight,
                output.get("future_waveform_structure"),
                waveform_multi_resolution_fft_sizes,
                waveform_auxiliary_structure_weight,
            )
            output["future_waveform_loss"] = waveform_losses["loss"]
            output["future_waveform_time_loss"] = waveform_losses["time_loss"]
            output["future_waveform_spectral_loss"] = waveform_losses["spectral_loss"]
            output["future_waveform_structure_loss"] = waveform_losses[
                "structure_loss"
            ]
            output["future_waveform_auxiliary_loss"] = waveform_losses[
                "auxiliary_loss"
            ]
            output["loss"] = output["loss"] + float(waveform_weight) * waveform_losses[
                "loss"
            ]
        return output


class ReliabilityGatedObservationWaveformWorldModel(ShortHorizonWaveformWorldModel):
    """Frozen M1 rollout with modality-specific reliability-gated state repair."""

    uses_reliability_gated_observation = True

    def __init__(
        self,
        *args,
        observation_adapter_hidden_dim: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        state_dim = self.encoder.config.d_model
        hidden_dim = int(observation_adapter_hidden_dim)
        if hidden_dim < 1:
            raise ValueError("observation adapter hidden dimensions must be positive")
        self.observation_state_adapters = nn.ModuleDict()
        self.observation_reliability_heads = nn.ModuleDict()
        for modality in self.encoder.config.modalities:
            adapter = nn.Sequential(
                nn.LayerNorm(state_dim),
                nn.Linear(state_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, state_dim),
            )
            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)
            self.observation_state_adapters[modality] = adapter
            self.observation_reliability_heads[modality] = nn.Sequential(
                nn.LayerNorm(state_dim + 2),
                nn.Linear(state_dim + 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

    def observation_parameters(self):
        yield from self.observation_state_adapters.parameters()
        yield from self.observation_reliability_heads.parameters()

    def direct_rollout_context(
        self,
        history_signals: torch.Tensor,
        history_present: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return ShortHorizonWaveformWorldModel.rollout_context(
            self, history_signals, history_present
        )

    def _observation_adjustment(
        self,
        output: Dict[str, torch.Tensor],
        history_signals: torch.Tensor,
        history_present: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        modality_states = output["history_modality_states"][:, -1]
        if history_present is None:
            availability = torch.ones(
                history_signals.shape[0],
                history_signals.shape[2],
                device=history_signals.device,
                dtype=history_signals.dtype,
            )
        else:
            availability = history_present.to(history_signals.dtype).mean(dim=1)
        signal_scale = history_signals.float().std(dim=-1).mean(dim=1)
        quality = signal_scale / (1.0 + signal_scale)
        weighted_adjustment = torch.zeros_like(output["history_state"])
        reliability_values = []
        denominator = availability.new_zeros(availability.shape[0], 1)
        for modality_index, modality in enumerate(self.encoder.config.modalities):
            state = modality_states[:, modality_index]
            metadata = torch.stack(
                (availability[:, modality_index], quality[:, modality_index]), dim=-1
            ).to(dtype=state.dtype)
            reliability = self.observation_reliability_heads[modality](
                torch.cat((state, metadata), dim=-1)
            ).sigmoid()
            reliability = reliability * availability[:, modality_index].unsqueeze(-1)
            weighted_adjustment = weighted_adjustment + reliability * self.observation_state_adapters[
                modality
            ](state)
            denominator = denominator + reliability
            reliability_values.append(reliability.squeeze(-1))
        adjustment = weighted_adjustment / denominator.clamp_min(1e-3)
        return adjustment, torch.stack(reliability_values, dim=-1), quality

    def rollout_context(
        self,
        history_signals: torch.Tensor,
        history_present: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        output = self.direct_rollout_context(history_signals, history_present)
        adjustment, reliability, quality = self._observation_adjustment(
            output, history_signals, history_present
        )
        direct_states = output["predicted_states"]
        predicted_states = direct_states + adjustment.unsqueeze(1)
        if self.factorized_transition_head or self.stage_head is None:
            raise ValueError("observation repair requires the standard stage head")
        stage_logits = self.stage_head(predicted_states)
        if self.current_stage_head is not None:
            stage_logits = stage_logits + output["current_stage_logits"].unsqueeze(1)

        direct_future_physiology = output["future_physiology"]
        future_by_group = {}
        offset = 0
        for group, size in self.physiology_group_sizes.items():
            current = output["current_physiology"][:, offset : offset + size]
            trajectory_delta = output["physiology_trajectory_delta"][
                :, :, offset : offset + size
            ]
            future_by_group[group] = (
                current.unsqueeze(1)
                + self.future_physiology_delta_heads[group](predicted_states)
                + trajectory_delta
            )
            offset += size

        output["direct_predicted_states"] = direct_states
        output["direct_stage_logits"] = output["stage_logits"]
        output["direct_future_physiology"] = direct_future_physiology
        output["direct_history_state"] = output["history_state"]
        output["observation_state_correction"] = adjustment
        output["corrected_history_state"] = output["history_state"] + adjustment
        output["observation_reliability"] = reliability
        output["observation_quality"] = quality
        output["predicted_states"] = predicted_states
        output["stage_logits"] = stage_logits
        output["future_physiology"] = torch.cat(
            [future_by_group[group] for group in self.physiology_group_sizes], dim=-1
        )
        return output


class TaskAwareReliabilityObservationWaveformWorldModel(
    ReliabilityGatedObservationWaveformWorldModel
):
    """Observation repair with modality-specific task-space residual heads."""

    uses_task_aware_observation_repair = True

    def __init__(self, *args, task_adapter_hidden_dim: int = 128, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        hidden_dim = int(task_adapter_hidden_dim)
        state_dim = self.encoder.config.d_model
        class_count = self.stage_head[-1].out_features
        feature_count = sum(self.physiology_group_sizes.values())
        self.task_stage_residual_heads = nn.ModuleDict()
        self.task_physiology_residual_heads = nn.ModuleDict()
        for modality in self.encoder.config.modalities:
            stage_head = nn.Sequential(
                nn.LayerNorm(state_dim),
                nn.Linear(state_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, class_count),
            )
            physiology_head = nn.Sequential(
                nn.LayerNorm(state_dim),
                nn.Linear(state_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, feature_count),
            )
            nn.init.zeros_(stage_head[-1].weight)
            nn.init.zeros_(stage_head[-1].bias)
            nn.init.zeros_(physiology_head[-1].weight)
            nn.init.zeros_(physiology_head[-1].bias)
            self.task_stage_residual_heads[modality] = stage_head
            self.task_physiology_residual_heads[modality] = physiology_head

    def task_parameters(self):
        yield from self.task_stage_residual_heads.parameters()
        yield from self.task_physiology_residual_heads.parameters()

    def _task_residuals(
        self,
        predicted_states: torch.Tensor,
        reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        denominator = reliability.sum(dim=-1, keepdim=True).clamp_min(1e-3)
        stage_delta = predicted_states.new_zeros(
            (*predicted_states.shape[:-1], self.stage_head[-1].out_features)
        )
        physiology_delta = predicted_states.new_zeros(
            (*predicted_states.shape[:-1], sum(self.physiology_group_sizes.values()))
        )
        for modality_index, modality in enumerate(self.encoder.config.modalities):
            weight = reliability[:, modality_index].reshape(-1, 1, 1)
            stage_delta = stage_delta + weight * self.task_stage_residual_heads[
                modality
            ](predicted_states)
            physiology_delta = physiology_delta + weight * self.task_physiology_residual_heads[
                modality
            ](predicted_states)
        return (
            stage_delta / denominator.unsqueeze(-1),
            physiology_delta / denominator.unsqueeze(-1),
        )

    def rollout_context(
        self,
        history_signals: torch.Tensor,
        history_present: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        output = super().rollout_context(history_signals, history_present)
        reliability = output["observation_reliability"]
        stage_delta, physiology_delta = self._task_residuals(
            output["predicted_states"], reliability
        )
        output["observation_base_stage_logits"] = output["stage_logits"]
        output["observation_base_future_physiology"] = output["future_physiology"]
        output["task_stage_residual"] = stage_delta
        output["task_physiology_residual"] = physiology_delta
        output["stage_logits"] = output["stage_logits"] + stage_delta
        output["future_physiology"] = output["future_physiology"] + physiology_delta
        return output


class GatedRecursiveTaskAwareWaveformWorldModel(
    TaskAwareReliabilityObservationWaveformWorldModel
):
    """Quality- and uncertainty-gated pure recursive latent rollout."""

    uses_gated_recursive_rollout = True

    def __init__(
        self,
        *args,
        recursive_hidden_dim: int = 128,
        rollout_horizons: Optional[Sequence[int]] = None,
        recursive_anchor_direct: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        state_dim = self.encoder.config.d_model
        hidden_dim = int(recursive_hidden_dim)
        modality_count = len(self.encoder.config.modalities)
        self.rollout_horizons = tuple(
            sorted(
                {
                    int(value)
                    for value in (
                        self.horizons if rollout_horizons is None else rollout_horizons
                    )
                }
            )
        )
        self.recursive_anchor_direct = bool(recursive_anchor_direct)
        if hidden_dim < 1 or not self.rollout_horizons or self.rollout_horizons[0] < 1:
            raise ValueError("recursive dimensions and horizons must be positive")
        self.recursive_state_norm = nn.LayerNorm(state_dim)
        self.recursive_hidden_initializer = nn.Sequential(
            nn.LayerNorm(state_dim), nn.Linear(state_dim, hidden_dim), nn.Tanh()
        )
        self.recursive_state_cell = nn.GRUCell(state_dim, hidden_dim)
        self.recursive_uncertainty_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1)
        )
        nn.init.zeros_(self.recursive_uncertainty_head[-1].weight)
        nn.init.constant_(self.recursive_uncertainty_head[-1].bias, -2.0)
        gate_input_dim = state_dim + 3 * modality_count + 1
        self.recursive_update_gate = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Sigmoid(),
        )
        self.recursive_delta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, state_dim),
        )
        nn.init.zeros_(self.recursive_delta_head[-1].weight)
        nn.init.zeros_(self.recursive_delta_head[-1].bias)

    def recursive_parameters(self):
        modules = (
            self.recursive_state_norm,
            self.recursive_hidden_initializer,
            self.recursive_state_cell,
            self.recursive_uncertainty_head,
            self.recursive_update_gate,
            self.recursive_delta_head,
        )
        for module in modules:
            yield from module.parameters()

    def _recursive_states(
        self,
        initial_state: torch.Tensor,
        availability: torch.Tensor,
        quality: torch.Tensor,
        reliability: torch.Tensor,
        horizons: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        selected_horizons = tuple(int(value) for value in horizons)
        if not selected_horizons or min(selected_horizons) < 1:
            raise ValueError("recursive rollout horizons must be positive")
        hidden = self.recursive_hidden_initializer(initial_state)
        state = initial_state
        metadata = torch.cat((availability, quality, 1.0 - reliability), dim=-1)
        states = []
        gates = []
        log_variances = []
        raw_deltas = []
        for _ in range(max(selected_horizons)):
            normalized_state = self.recursive_state_norm(state)
            hidden = self.recursive_state_cell(normalized_state, hidden)
            log_variance = self.recursive_uncertainty_head(hidden).clamp(-6.0, 3.0)
            gate = self.recursive_update_gate(
                torch.cat((normalized_state, metadata, log_variance.sigmoid()), dim=-1)
            )
            raw_delta = self.recursive_delta_head(hidden)
            state = state + gate * raw_delta
            states.append(state)
            gates.append(gate)
            log_variances.append(log_variance.squeeze(-1))
            raw_deltas.append(raw_delta)
        indices = [horizon - 1 for horizon in selected_horizons]
        return (
            torch.stack([states[index] for index in indices], dim=1),
            torch.stack([gates[index] for index in indices], dim=1),
            torch.stack([log_variances[index] for index in indices], dim=1),
            torch.stack([raw_deltas[index] for index in indices], dim=1),
        )

    def rollout_context_horizons(
        self,
        history_signals: torch.Tensor,
        history_present: Optional[torch.Tensor] = None,
        horizons: Optional[Sequence[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        output = super().rollout_context(history_signals, history_present)
        selected_horizons = self.rollout_horizons if horizons is None else tuple(horizons)
        if history_present is None:
            availability = torch.ones_like(output["observation_reliability"])
        else:
            availability = history_present.to(history_signals.dtype).mean(dim=1)
        recursive_baseline_states = output["corrected_history_state"].unsqueeze(1)
        recursive_baseline_states = recursive_baseline_states.expand(
            -1, len(selected_horizons), -1
        )
        if self.recursive_anchor_direct:
            anchor_horizon = max(self.horizons)
            direct_indices = {horizon: index for index, horizon in enumerate(self.horizons)}
            unsupported = [
                horizon
                for horizon in selected_horizons
                if horizon <= anchor_horizon and horizon not in direct_indices
            ]
            if unsupported:
                raise ValueError(
                    f"anchored recursive rollout lacks direct states for {unsupported}"
                )
            extended = [horizon for horizon in selected_horizons if horizon > anchor_horizon]
            long_state_by_horizon = {}
            long_gate_by_horizon = {}
            long_variance_by_horizon = {}
            long_delta_by_horizon = {}
            anchor_state = output["predicted_states"][:, direct_indices[anchor_horizon]]
            if extended:
                long_states, long_gates, long_variances, long_deltas = self._recursive_states(
                    anchor_state,
                    availability,
                    output["observation_quality"],
                    output["observation_reliability"],
                    [horizon - anchor_horizon for horizon in extended],
                )
                for index, horizon in enumerate(extended):
                    long_state_by_horizon[horizon] = long_states[:, index]
                    long_gate_by_horizon[horizon] = long_gates[:, index]
                    long_variance_by_horizon[horizon] = long_variances[:, index]
                    long_delta_by_horizon[horizon] = long_deltas[:, index]
            zero_state = torch.zeros_like(anchor_state)
            zero_variance = anchor_state.new_full((anchor_state.shape[0],), -2.0)
            predicted_states = torch.stack(
                [
                    output["predicted_states"][:, direct_indices[horizon]]
                    if horizon in direct_indices
                    else long_state_by_horizon[horizon]
                    for horizon in selected_horizons
                ],
                dim=1,
            )
            gates = torch.stack(
                [
                    zero_state
                    if horizon in direct_indices
                    else long_gate_by_horizon[horizon]
                    for horizon in selected_horizons
                ],
                dim=1,
            )
            log_variance = torch.stack(
                [
                    zero_variance
                    if horizon in direct_indices
                    else long_variance_by_horizon[horizon]
                    for horizon in selected_horizons
                ],
                dim=1,
            )
            raw_delta = torch.stack(
                [
                    zero_state
                    if horizon in direct_indices
                    else long_delta_by_horizon[horizon]
                    for horizon in selected_horizons
                ],
                dim=1,
            )
            recursive_baseline_states = torch.stack(
                [
                    output["corrected_history_state"]
                    if horizon in direct_indices
                    else anchor_state
                    for horizon in selected_horizons
                ],
                dim=1,
            )
            recursive_correction = torch.stack(
                [
                    zero_state
                    if horizon in direct_indices
                    else long_state_by_horizon[horizon] - anchor_state
                    for horizon in selected_horizons
                ],
                dim=1,
            )
        else:
            predicted_states, gates, log_variance, raw_delta = self._recursive_states(
                output["corrected_history_state"],
                availability,
                output["observation_quality"],
                output["observation_reliability"],
                selected_horizons,
            )
            recursive_correction = (
                predicted_states - output["corrected_history_state"].unsqueeze(1)
            )
        if self.factorized_transition_head or self.stage_head is None:
            raise ValueError("gated recursive rollout requires the standard stage head")
        stage_logits = self.stage_head(predicted_states)
        if self.current_stage_head is not None:
            stage_logits = stage_logits + output["current_stage_logits"].unsqueeze(1)
        task_stage, task_physiology = self._task_residuals(
            predicted_states, output["observation_reliability"]
        )
        stage_logits = stage_logits + task_stage
        future_by_group = {}
        offset = 0
        for group, size in self.physiology_group_sizes.items():
            current = output["current_physiology"][:, offset : offset + size]
            future_by_group[group] = current.unsqueeze(1) + self.future_physiology_delta_heads[
                group
            ](predicted_states)
            offset += size
        future_physiology = torch.cat(
            [future_by_group[group] for group in self.physiology_group_sizes], dim=-1
        ) + task_physiology
        if self.recursive_anchor_direct:
            stage_logits = torch.stack(
                [
                    output["stage_logits"][:, direct_indices[horizon]]
                    if horizon in direct_indices
                    else stage_logits[:, index]
                    for index, horizon in enumerate(selected_horizons)
                ],
                dim=1,
            )
            future_physiology = torch.stack(
                [
                    output["future_physiology"][:, direct_indices[horizon]]
                    if horizon in direct_indices
                    else future_physiology[:, index]
                    for index, horizon in enumerate(selected_horizons)
                ],
                dim=1,
            )
        output["observation_predicted_states"] = output["predicted_states"]
        output["observation_stage_logits"] = output["stage_logits"]
        output["observation_future_physiology"] = output["future_physiology"]
        output["recursive_horizons"] = torch.tensor(
            selected_horizons, device=predicted_states.device, dtype=torch.long
        )
        output["recursive_update_gate"] = gates
        output["recursive_log_variance"] = log_variance
        output["recursive_raw_delta"] = raw_delta
        output["recursive_baseline_states"] = recursive_baseline_states
        output["recursive_state_correction"] = recursive_correction
        output["predicted_states"] = predicted_states
        output["stage_logits"] = stage_logits
        output["future_physiology"] = future_physiology
        return output

    def rollout_context(
        self,
        history_signals: torch.Tensor,
        history_present: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.rollout_context_horizons(
            history_signals, history_present, self.horizons
        )


class HybridRecursiveWaveformWorldModel(ShortHorizonWaveformWorldModel):
    """Frozen direct rollout plus a shared recursive latent residual."""

    uses_hybrid_recursive_rollout = True

    def __init__(
        self,
        *args,
        recursive_hidden_dim: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        state_dim = self.encoder.config.d_model
        hidden_dim = int(recursive_hidden_dim)
        if hidden_dim < 1:
            raise ValueError("recursive hidden dimensions must be positive")
        self.recursive_state_norm = nn.LayerNorm(state_dim)
        self.recursive_hidden_initializer = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
        )
        self.recursive_state_cell = nn.GRUCell(state_dim, hidden_dim)
        self.recursive_delta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, state_dim),
        )
        nn.init.zeros_(self.recursive_delta_head[-1].weight)
        nn.init.zeros_(self.recursive_delta_head[-1].bias)

    def recursive_parameters(self):
        modules = (
            self.recursive_state_norm,
            self.recursive_hidden_initializer,
            self.recursive_state_cell,
            self.recursive_delta_head,
        )
        for module in modules:
            yield from module.parameters()

    def _recursive_corrections(self, initial_state: torch.Tensor) -> torch.Tensor:
        hidden = self.recursive_hidden_initializer(initial_state)
        state = initial_state
        corrections = []
        for _ in range(max(self.horizons)):
            hidden = self.recursive_state_cell(
                self.recursive_state_norm(state), hidden
            )
            state = state + self.recursive_delta_head(hidden)
            corrections.append(state - initial_state)
        return torch.stack(
            [corrections[horizon - 1] for horizon in self.horizons], dim=1
        )

    def rollout_context(
        self,
        history_signals: torch.Tensor,
        history_present: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        output = super().rollout_context(history_signals, history_present)
        direct_states = output["predicted_states"]
        corrections = self._recursive_corrections(output["history_state"])
        predicted_states = direct_states + corrections

        if self.factorized_transition_head or self.stage_head is None:
            raise ValueError("hybrid recursive rollout requires the standard stage head")
        stage_logits = self.stage_head(predicted_states)
        if self.current_stage_head is not None:
            stage_logits = stage_logits + output["current_stage_logits"].unsqueeze(1)

        direct_future_physiology = output["future_physiology"]
        future_by_group = {}
        offset = 0
        for group, size in self.physiology_group_sizes.items():
            current = output["current_physiology"][:, offset : offset + size]
            trajectory_delta = output["physiology_trajectory_delta"][
                :, :, offset : offset + size
            ]
            future_by_group[group] = (
                current.unsqueeze(1)
                + self.future_physiology_delta_heads[group](predicted_states)
                + trajectory_delta
            )
            offset += size

        output["direct_predicted_states"] = direct_states
        output["direct_stage_logits"] = output["stage_logits"]
        output["direct_future_physiology"] = direct_future_physiology
        output["recursive_state_correction"] = corrections
        output["recursive_rollout_states"] = (
            output["history_state"].unsqueeze(1) + corrections
        )
        output["predicted_states"] = predicted_states
        output["stage_logits"] = stage_logits
        output["future_physiology"] = torch.cat(
            [future_by_group[group] for group in self.physiology_group_sizes], dim=-1
        )
        return output


class PrivateTemporalPhysiologyWorldModel(TrajectoryPhysiologyWorldModel):
    """Full-history model with causal modality-private dynamics and late fusion."""

    uses_private_temporal_dynamics = True

    def __init__(
        self,
        *args,
        private_transition_layers: int = 1,
        private_transition_heads: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        del self.physiology_state_adapters
        state_dim = self.encoder.config.d_model
        self.private_transitions = nn.ModuleDict()
        self.private_horizon_embeddings = nn.ParameterDict()
        self.private_state_predictors = nn.ModuleDict()
        self.private_future_state_adapters = nn.ModuleDict()
        for group in self.physiology_group_sizes:
            layer = nn.TransformerEncoderLayer(
                d_model=state_dim,
                nhead=private_transition_heads,
                dim_feedforward=state_dim * 4,
                dropout=float(self.encoder.config.dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.private_transitions[group] = nn.TransformerEncoder(
                layer,
                num_layers=private_transition_layers,
            )
            horizon_embedding = nn.Parameter(torch.empty(len(self.horizons), state_dim))
            nn.init.normal_(horizon_embedding, std=0.02)
            self.private_horizon_embeddings[group] = horizon_embedding
            self.private_state_predictors[group] = nn.Sequential(
                nn.LayerNorm(state_dim),
                nn.Linear(state_dim, state_dim * 2),
                nn.GELU(),
                nn.Linear(state_dim * 2, state_dim),
            )
            adapter = nn.Linear(state_dim, state_dim, bias=False)
            nn.init.zeros_(adapter.weight)
            self.private_future_state_adapters[group] = adapter

    def rollout(
        self,
        history_signals: torch.Tensor,
        history_present: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        states, modality_states = self._encode_history_with_modality_states(
            history_signals,
            history_present,
        )
        output = self._rollout_from_states(states)
        positions = sinusoidal_positions(
            states.shape[1], states.shape[2], states.device, states.dtype
        )
        causal_mask = torch.triu(
            torch.ones(
                states.shape[1],
                states.shape[1],
                dtype=torch.bool,
                device=states.device,
            ),
            diagonal=1,
        )
        history_by_group = {}
        private_future_by_group = {}
        fusion_delta = torch.zeros_like(output["predicted_states"])
        for group, head in self.current_physiology_heads.items():
            modality_index = self.encoder.config.modalities.index(group)
            group_states = modality_states[:, :, modality_index]
            history_by_group[group] = head(group_states)
            transitioned = self.private_transitions[group](
                group_states + positions.unsqueeze(0),
                mask=causal_mask,
            )
            context = transitioned[:, -1].unsqueeze(1)
            context = context + self.private_horizon_embeddings[group].unsqueeze(0)
            private_future = self.private_state_predictors[group](context)
            private_future_by_group[group] = private_future
            fusion_delta = fusion_delta + self.private_future_state_adapters[group](
                private_future
            )

        predicted_states = output["predicted_states"] + fusion_delta
        if self.factorized_transition_head or self.stage_head is None:
            raise ValueError("private temporal dynamics require the standard stage head")
        stage_logits = self.stage_head(predicted_states)
        if self.current_stage_head is not None:
            stage_logits = stage_logits + output["current_stage_logits"].unsqueeze(1)
        current_by_group = {
            group: prediction[:, -1] for group, prediction in history_by_group.items()
        }
        future_by_group = {
            group: current_by_group[group].unsqueeze(1)
            + self.future_physiology_delta_heads[group](private_future_by_group[group])
            for group in self.physiology_group_sizes
        }
        output["predicted_states"] = predicted_states
        output["stage_logits"] = stage_logits
        output["history_physiology"] = torch.cat(list(history_by_group.values()), dim=-1)
        output["current_physiology"] = torch.cat(list(current_by_group.values()), dim=-1)
        output["future_physiology"] = torch.cat(list(future_by_group.values()), dim=-1)
        output["private_predicted_states"] = torch.stack(
            [private_future_by_group[group] for group in self.encoder.config.modalities],
            dim=2,
        )
        output["private_fusion_delta"] = fusion_delta
        return output

    def _target_modality_states(
        self,
        signals: torch.Tensor,
        modality_present: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if signals.ndim != 4:
            raise ValueError("sequence signals must be [batch, time, modalities, samples]")
        batch, time, modalities, samples = signals.shape
        flat_signals = signals.reshape(batch * time, modalities, samples)
        flat_present = None
        if modality_present is not None:
            flat_present = modality_present.reshape(batch * time, modalities)
        target_encoder = self.target_encoder if self.target_encoder is not None else self.encoder
        with torch.no_grad():
            _, tokens = target_encoder.encode_with_representation(
                flat_signals,
                modality_present=flat_present,
            )
            states = tokens.mean(dim=2)
        return states.reshape(batch, time, modalities, -1)

    def forward(
        self,
        history_signals: torch.Tensor,
        history_present: Optional[torch.Tensor] = None,
        future_signals: Optional[torch.Tensor] = None,
        future_present: Optional[torch.Tensor] = None,
        future_labels: Optional[torch.Tensor] = None,
        history_labels: Optional[torch.Tensor] = None,
        private_latent_weight: float = 1.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        output = super().forward(
            history_signals,
            history_present,
            future_signals=future_signals,
            future_present=future_present,
            future_labels=future_labels,
            history_labels=history_labels,
            **kwargs,
        )
        if future_signals is not None:
            targets = self._target_modality_states(future_signals, future_present)
            group_losses = {}
            for group in self.physiology_group_sizes:
                modality_index = self.encoder.config.modalities.index(group)
                group_losses[group] = F.smooth_l1_loss(
                    output["private_predicted_states"][:, :, modality_index],
                    targets[:, :, modality_index],
                )
            private_latent_loss = torch.stack(list(group_losses.values())).mean()
            output["private_latent_loss"] = private_latent_loss
            output["private_latent_loss_by_group"] = group_losses
            output["loss"] = output["loss"] + float(private_latent_weight) * private_latent_loss
        return output


def observation_config(data: Dict[str, object], model: Dict[str, object]) -> ObservationConfig:
    return ObservationConfig(
        modalities=tuple(data["modalities"]),  # type: ignore[arg-type]
        sample_rate=int(data["sample_rate"]),
        epoch_seconds=int(data["epoch_seconds"]),
        patch_samples=int(model.get("patch_samples", 64)),
        d_model=int(model.get("d_model", 128)),
        layers=int(model.get("layers", 4)),
        heads=int(model.get("heads", 4)),
        dropout=float(model.get("dropout", 0.1)),
        pooling=str(model.get("pooling", "mean")),
    )


def build_supervised_model(config: Dict[str, Dict[str, object]]) -> nn.Module:
    data = config["data"]
    model = config["model"]
    if model["name"] == "tcn":
        return TCNSleepClassifier(
            modalities=len(data["modalities"]),  # type: ignore[arg-type]
            channels=int(model.get("channels", 64)),
            levels=int(model.get("levels", 4)),
            kernel_size=int(model.get("kernel_size", 7)),
            dropout=float(model.get("dropout", 0.1)),
            num_classes=int(data.get("num_classes", 5)),
        )
    if model["name"] == "transformer":
        encoder = MultiModalEncoder(observation_config(data, model))
        return SleepStageClassifier(
            encoder,
            num_classes=int(data.get("num_classes", 5)),
            hidden_dim=int(model.get("hidden_dim", 0)),
        )
    if model["name"] == "cnn_bilstm":
        return CNNBiLSTMSleepClassifier(
            modalities=len(data["modalities"]),  # type: ignore[arg-type]
            channels=int(model.get("channels", 64)),
            tokens=int(model.get("tokens", 16)),
            hidden_dim=int(model.get("hidden_dim", 64)),
            dropout=float(model.get("dropout", 0.1)),
            num_classes=int(data.get("num_classes", 5)),
        )
    if model["name"] == "attentive_cnn":
        return AttentiveCNNSleepClassifier(
            modalities=len(data["modalities"]),  # type: ignore[arg-type]
            channels=int(model.get("channels", 64)),
            tokens=int(model.get("tokens", 16)),
            layers=int(model.get("layers", 2)),
            heads=int(model.get("heads", 4)),
            dropout=float(model.get("dropout", 0.1)),
            num_classes=int(data.get("num_classes", 5)),
        )
    raise ValueError(f"unknown model: {model['name']}")
