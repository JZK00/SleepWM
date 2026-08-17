from __future__ import annotations

from typing import Dict, Sequence

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def observation_age(present: torch.Tensor) -> torch.Tensor:
    """Epochs since the latest valid observation for each modality."""
    if present.ndim != 3:
        raise ValueError("present must have shape [batch, history, modalities]")
    ages = torch.zeros_like(present, dtype=torch.float32)
    age = torch.zeros_like(present[:, 0], dtype=torch.float32)
    for index in range(present.shape[1]):
        age = torch.where(present[:, index], torch.zeros_like(age), age + 1.0)
        ages[:, index] = age
    return ages


class ModalityEpochEncoder(nn.Module):
    def __init__(self, modalities: int, feature_dim: int, dropout: float) -> None:
        super().__init__()
        self.modalities = modalities
        self.feature_dim = feature_dim
        self.signal = nn.Sequential(
            nn.Conv1d(1, 24, 25, stride=8, padding=12),
            nn.GroupNorm(4, 24),
            nn.GELU(),
            nn.Conv1d(24, 32, 9, stride=4, padding=4),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv1d(32, feature_dim, 7, stride=4, padding=3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.modality_embedding = nn.Parameter(torch.empty(modalities, feature_dim))
        nn.init.normal_(self.modality_embedding, std=0.02)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, signals: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        if signals.ndim != 4 or present.shape != signals.shape[:3]:
            raise ValueError("signals/present shape mismatch")
        batch, history, modalities, samples = signals.shape
        values = signals.reshape(batch * history * modalities, 1, samples)
        encoded = self.signal(values).reshape(batch, history, modalities, self.feature_dim)
        encoded = encoded + self.modality_embedding.reshape(1, 1, modalities, -1)
        encoded = self.dropout(self.norm(encoded))
        return encoded * present.to(encoded.dtype).unsqueeze(-1)


class DecayGRUPath(nn.Module):
    def __init__(self, modalities: int, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.modalities = modalities
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.feature_mean = nn.Parameter(torch.zeros(modalities, feature_dim))
        self.input_decay_weight = nn.Parameter(torch.full((modalities, feature_dim), 0.10))
        self.input_decay_bias = nn.Parameter(torch.zeros(modalities, feature_dim))
        self.hidden_decay = nn.Linear(1, hidden_dim)
        input_dim = modalities * feature_dim + modalities * 2
        self.cell = nn.GRUCell(input_dim, hidden_dim)

    def forward(
        self,
        encoded: torch.Tensor,
        present: torch.Tensor,
        ages: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, history, modalities, feature_dim = encoded.shape
        last = self.feature_mean.reshape(1, modalities, feature_dim).expand(batch, -1, -1)
        hidden = encoded.new_zeros(batch, self.hidden_dim)
        trajectory = []
        for index in range(history):
            mask = present[:, index]
            age = ages[:, index]
            gamma_x = torch.exp(
                -torch.relu(
                    age.unsqueeze(-1) * self.input_decay_weight.unsqueeze(0)
                    + self.input_decay_bias.unsqueeze(0)
                )
            )
            decayed = gamma_x * last + (1.0 - gamma_x) * self.feature_mean.unsqueeze(0)
            current = torch.where(mask.unsqueeze(-1), encoded[:, index], decayed)
            last = torch.where(mask.unsqueeze(-1), encoded[:, index], last)
            mean_age = age.mean(dim=-1, keepdim=True)
            gamma_h = torch.exp(-torch.relu(self.hidden_decay(mean_age)))
            hidden = hidden * gamma_h
            cell_input = torch.cat(
                [
                    current.flatten(1),
                    mask.to(current.dtype),
                    torch.log1p(age),
                ],
                dim=-1,
            )
            hidden = self.cell(cell_input, hidden)
            trajectory.append(hidden)
        return hidden, torch.stack(trajectory, dim=1)


class MultiTimeAttention(nn.Module):
    """mTAN multi-time attention adapted from the official MIT implementation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embed_time: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if embed_time % heads != 0:
            raise ValueError("embed_time must be divisible by heads")
        self.heads = heads
        self.embed_time = embed_time
        self.key_dim = embed_time // heads
        self.query_projection = nn.Linear(embed_time, embed_time)
        self.key_projection = nn.Linear(embed_time, embed_time)
        self.output_projection = nn.Linear(input_dim * heads, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch = value.shape[0]
        query = self.query_projection(query).view(
            batch, -1, self.heads, self.key_dim
        ).transpose(1, 2)
        key = self.key_projection(key).view(
            batch, -1, self.heads, self.key_dim
        ).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.key_dim
        )
        scores = scores.unsqueeze(-1).expand(-1, -1, -1, -1, value.shape[-1])
        expanded_mask = mask[:, None, None, :, :]
        scores = scores.masked_fill(~expanded_mask, -1e4)
        attention = self.dropout(torch.softmax(scores, dim=-2))
        attended = torch.sum(attention * value[:, None, None, :, :], dim=-2)
        attended = attended.transpose(1, 2).contiguous().flatten(2)
        return self.output_projection(attended)


class MTANPath(nn.Module):
    def __init__(
        self,
        modalities: int,
        feature_dim: int,
        hidden_dim: int,
        heads: int,
        dropout: float,
        embed_time: int = 32,
    ) -> None:
        super().__init__()
        value_dim = modalities * feature_dim
        self.modalities = modalities
        self.feature_dim = feature_dim
        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)
        self.attention = MultiTimeAttention(
            input_dim=2 * value_dim,
            hidden_dim=hidden_dim,
            embed_time=embed_time,
            heads=heads,
            dropout=dropout,
        )
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def time_embedding(self, time: torch.Tensor) -> torch.Tensor:
        time = time.unsqueeze(-1)
        return torch.cat((self.linear(time), torch.sin(self.periodic(time))), dim=-1)

    def forward(
        self,
        encoded: torch.Tensor,
        present: torch.Tensor,
        ages: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del ages
        batch, history, modalities, feature_dim = encoded.shape
        values = encoded.reshape(batch, history, modalities * feature_dim)
        feature_mask = present.unsqueeze(-1).expand(
            -1, -1, -1, feature_dim
        ).reshape(batch, history, modalities * feature_dim)
        attention_values = torch.cat((values, feature_mask.to(values.dtype)), dim=-1)
        attention_mask = torch.cat((feature_mask, feature_mask), dim=-1)
        time = torch.linspace(
            0.0, 1.0, history, device=values.device, dtype=values.dtype
        ).unsqueeze(0).expand(batch, -1)
        embedded_time = self.time_embedding(time)
        attended = self.attention(
            embedded_time,
            embedded_time,
            attention_values,
            attention_mask,
        )
        trajectory, hidden = self.temporal(attended)
        trajectory = self.norm(trajectory)
        return hidden[-1], trajectory


class DenseSensorGraphPath(nn.Module):
    """Dense three-sensor form of Raindrop's learned sensor graph propagation."""

    def __init__(
        self,
        modalities: int,
        feature_dim: int,
        hidden_dim: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.modalities = modalities
        self.feature_dim = feature_dim
        self.query = nn.Linear(feature_dim, feature_dim, bias=False)
        self.key = nn.Linear(feature_dim, feature_dim, bias=False)
        self.value = nn.Linear(feature_dim, feature_dim, bias=False)
        self.sensor_embedding = nn.Parameter(torch.empty(modalities, feature_dim))
        self.edge_logits = nn.Parameter(torch.zeros(modalities, modalities))
        token_input = modalities * feature_dim + 2 * modalities
        self.token_projection = nn.Linear(token_input, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.sensor_embedding, std=0.02)

    def forward(
        self,
        encoded: torch.Tensor,
        present: torch.Tensor,
        ages: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = encoded + self.sensor_embedding.reshape(
            1, 1, self.modalities, self.feature_dim
        )
        query = self.query(values)
        key = self.key(values)
        message = self.value(values)
        scores = torch.einsum("btmf,btnf->btmn", query, key) / math.sqrt(
            self.feature_dim
        )
        scores = scores + self.edge_logits.reshape(
            1, 1, self.modalities, self.modalities
        )
        key_present = present[:, :, None, :]
        any_present = present.any(dim=-1, keepdim=True).unsqueeze(-1)
        safe_mask = key_present | ~any_present
        scores = scores.masked_fill(~safe_mask, -1e4)
        attention = torch.softmax(scores, dim=-1)
        propagated = torch.einsum("btmn,btnf->btmf", attention, message)
        propagated = propagated + values
        token = torch.cat(
            (
                propagated.flatten(2),
                present.to(propagated.dtype),
                torch.log1p(ages),
            ),
            dim=-1,
        )
        token = self.token_projection(token)
        token = token + DynamicMissingOutcomeBaseline._positions(
            token.shape[1], token.shape[2], token.device, token.dtype
        ).unsqueeze(0)
        trajectory = self.norm(self.temporal(token))
        return trajectory[:, -1], trajectory


class SAITSPath(nn.Module):
    """Two-stage diagonally masked attention adapted to causal history encoding."""

    def __init__(
        self,
        modalities: int,
        feature_dim: int,
        hidden_dim: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.modalities = modalities
        self.feature_dim = feature_dim
        token_dim = modalities * feature_dim + 2 * modalities
        self.first_projection = nn.Linear(token_dim, hidden_dim)
        self.reconstruction = nn.Linear(hidden_dim, modalities * feature_dim)
        self.second_projection = nn.Linear(token_dim, hidden_dim)
        self.combination_gate = nn.Linear(2 * hidden_dim, hidden_dim)

        def block() -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=4 * hidden_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=layers)

        self.first_block = block()
        self.second_block = block()
        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _diagonal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.eye(length, device=device, dtype=torch.bool)

    def forward(
        self,
        encoded: torch.Tensor,
        present: torch.Tensor,
        ages: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, history, modalities, feature_dim = encoded.shape
        metadata = torch.cat(
            (present.to(encoded.dtype), torch.log1p(ages)), dim=-1
        )
        positions = DynamicMissingOutcomeBaseline._positions(
            history, self.first_projection.out_features, encoded.device, encoded.dtype
        ).unsqueeze(0)
        first_input = torch.cat((encoded.flatten(2), metadata), dim=-1)
        first = self.first_projection(first_input) + positions
        diagonal = self._diagonal_mask(history, encoded.device)
        first = self.first_block(first, mask=diagonal)
        reconstructed = self.reconstruction(first).reshape(
            batch, history, modalities, feature_dim
        )
        imputed = torch.where(present.unsqueeze(-1), encoded, reconstructed)
        second_input = torch.cat((imputed.flatten(2), metadata), dim=-1)
        second = self.second_projection(second_input) + positions
        gate = torch.sigmoid(self.combination_gate(torch.cat((first, second), dim=-1)))
        mixed = gate * first + (1.0 - gate) * second
        trajectory = self.norm(self.second_block(mixed, mask=diagonal))

        observed = present.to(encoded.dtype).unsqueeze(-1)
        reconstruction_loss = F.smooth_l1_loss(
            reconstructed, encoded.detach(), reduction="none"
        )
        reconstruction_loss = (
            (reconstruction_loss * observed).sum()
            / observed.expand_as(reconstruction_loss).sum().clamp_min(1.0)
        )
        return trajectory[:, -1], trajectory, reconstruction_loss


class PatchTSTPath(nn.Module):
    """Channel-independent temporal patch encoder followed by sensor fusion."""

    def __init__(
        self,
        modalities: int,
        feature_dim: int,
        hidden_dim: int,
        layers: int,
        heads: int,
        dropout: float,
        patch_size: int = 4,
        patch_stride: int = 2,
    ) -> None:
        super().__init__()
        self.modalities = modalities
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        patch_input = (feature_dim + 2) * patch_size
        self.patch_projection = nn.Linear(patch_input, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=layers)
        self.fusion_score = nn.Linear(hidden_dim + 2, 1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        encoded: torch.Tensor,
        present: torch.Tensor,
        ages: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, history, modalities, _ = encoded.shape
        modality_tokens = torch.cat(
            (
                encoded,
                present.to(encoded.dtype).unsqueeze(-1),
                torch.log1p(ages).unsqueeze(-1),
            ),
            dim=-1,
        ).permute(0, 2, 3, 1).reshape(batch * modalities, -1, history)
        if history < self.patch_size:
            modality_tokens = F.pad(
                modality_tokens, (self.patch_size - history, 0)
            )
        patches = modality_tokens.unfold(
            dimension=-1, size=self.patch_size, step=self.patch_stride
        )
        patches = patches.permute(0, 2, 1, 3).flatten(2)
        patches = self.patch_projection(patches)
        patches = patches + DynamicMissingOutcomeBaseline._positions(
            patches.shape[1], patches.shape[2], patches.device, patches.dtype
        ).unsqueeze(0)
        patches = self.norm(self.temporal(patches))
        modality_state = patches[:, -1].reshape(batch, modalities, -1)
        latest_metadata = torch.stack(
            (present[:, -1].to(encoded.dtype), torch.log1p(ages[:, -1])), dim=-1
        )
        weights = torch.softmax(
            self.fusion_score(torch.cat((modality_state, latest_metadata), dim=-1)),
            dim=1,
        )
        state = torch.sum(weights * modality_state, dim=1)
        trajectory = patches.reshape(batch, modalities, patches.shape[1], -1).mean(dim=1)
        return state, trajectory


class RSSMPath(nn.Module):
    """Compact recurrent state-space baseline with learned prior and posterior."""

    def __init__(
        self,
        modalities: int,
        feature_dim: int,
        hidden_dim: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        token_dim = modalities * feature_dim + 2 * modalities
        self.observation_projection = nn.Sequential(
            nn.Linear(token_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.prior = nn.Linear(hidden_dim, 2 * latent_dim)
        self.posterior = nn.Linear(2 * hidden_dim, 2 * latent_dim)
        self.recurrent = nn.GRUCell(hidden_dim + latent_dim, hidden_dim)
        self.state_projection = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.latent_dim = latent_dim

    @staticmethod
    def _distribution(parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_scale = parameters.chunk(2, dim=-1)
        log_std = torch.clamp(raw_scale, min=-5.0, max=2.0)
        return mean, log_std

    @staticmethod
    def _kl(
        posterior_mean: torch.Tensor,
        posterior_log_std: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_log_std: torch.Tensor,
    ) -> torch.Tensor:
        variance_ratio = torch.exp(
            2.0 * (posterior_log_std - prior_log_std)
        )
        mean_term = (posterior_mean - prior_mean).pow(2) * torch.exp(
            -2.0 * prior_log_std
        )
        return 0.5 * (
            variance_ratio + mean_term - 1.0
            + 2.0 * (prior_log_std - posterior_log_std)
        ).sum(dim=-1)

    def forward(
        self,
        encoded: torch.Tensor,
        present: torch.Tensor,
        ages: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, history, _, _ = encoded.shape
        metadata = torch.cat(
            (present.to(encoded.dtype), torch.log1p(ages)), dim=-1
        )
        observation = self.observation_projection(
            torch.cat((encoded.flatten(2), metadata), dim=-1)
        )
        hidden = encoded.new_zeros(batch, self.prior.in_features)
        trajectory = []
        divergences = []
        for index in range(history):
            prior_mean, prior_log_std = self._distribution(self.prior(hidden))
            posterior_mean, posterior_log_std = self._distribution(
                self.posterior(torch.cat((hidden, observation[:, index]), dim=-1))
            )
            has_observation = present[:, index].any(dim=-1, keepdim=True)
            mean = torch.where(has_observation, posterior_mean, prior_mean)
            log_std = torch.where(has_observation, posterior_log_std, prior_log_std)
            latent = mean
            if self.training:
                latent = mean + torch.randn_like(mean) * torch.exp(log_std)
            divergences.append(
                self._kl(
                    posterior_mean,
                    posterior_log_std,
                    prior_mean,
                    prior_log_std,
                ) * has_observation.squeeze(-1).to(encoded.dtype)
            )
            hidden = self.recurrent(
                torch.cat((observation[:, index], latent), dim=-1), hidden
            )
            trajectory.append(
                self.state_projection(torch.cat((hidden, latent), dim=-1))
            )
        trajectory_tensor = torch.stack(trajectory, dim=1)
        kl_loss = torch.stack(divergences, dim=1).mean()
        return trajectory_tensor[:, -1], trajectory_tensor, kl_loss


class DynamicMissingOutcomeBaseline(nn.Module):
    """Matched multi-outcome baseline for dynamically incomplete PSG histories."""

    def __init__(
        self,
        architecture: str,
        modalities: int,
        horizons: Sequence[int],
        physiology_features: int,
        num_classes: int = 5,
        feature_dim: int = 32,
        hidden_dim: int = 96,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        patch_size: int = 4,
        patch_stride: int = 2,
        latent_dim: int = 32,
    ) -> None:
        super().__init__()
        if architecture not in {
            "grud",
            "brits",
            "masked_transformer",
            "mtan",
            "raindrop",
            "saits",
            "patchtst",
            "rssm",
        }:
            raise ValueError(f"unsupported architecture: {architecture}")
        self.architecture = architecture
        self.horizons = tuple(int(value) for value in horizons)
        self.encoder = ModalityEpochEncoder(modalities, feature_dim, dropout)
        self.modalities = modalities
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        if architecture in {"grud", "brits"}:
            self.forward_path = DecayGRUPath(modalities, feature_dim, hidden_dim)
            if architecture == "brits":
                self.backward_path = DecayGRUPath(modalities, feature_dim, hidden_dim)
                self.path_fusion = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                )
        elif architecture == "mtan":
            self.mtan_path = MTANPath(
                modalities,
                feature_dim,
                hidden_dim,
                heads,
                dropout,
            )
        elif architecture == "raindrop":
            self.raindrop_path = DenseSensorGraphPath(
                modalities,
                feature_dim,
                hidden_dim,
                layers,
                heads,
                dropout,
            )
        elif architecture == "saits":
            self.saits_path = SAITSPath(
                modalities,
                feature_dim,
                hidden_dim,
                layers,
                heads,
                dropout,
            )
        elif architecture == "patchtst":
            self.patchtst_path = PatchTSTPath(
                modalities,
                feature_dim,
                hidden_dim,
                layers,
                heads,
                dropout,
                patch_size,
                patch_stride,
            )
        elif architecture == "rssm":
            self.rssm_path = RSSMPath(
                modalities,
                feature_dim,
                hidden_dim,
                latent_dim,
            )
        else:
            fused_dim = modalities * feature_dim
            self.observation_projection = nn.Linear(fused_dim, hidden_dim)
            self.missingness_projection = nn.Linear(modalities * 2, hidden_dim)
            self.all_missing_token = nn.Parameter(torch.zeros(hidden_dim))
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=layers)
            self.temporal_norm = nn.LayerNorm(hidden_dim)

        self.current_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes)
        )
        self.future_stage_heads = nn.ModuleList(
            [
                nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes))
                for _ in self.horizons
            ]
        )
        self.future_physiology_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, physiology_features),
                )
                for _ in self.horizons
            ]
        )

    @staticmethod
    def _positions(length: int, width: int, device, dtype) -> torch.Tensor:
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, width, 2, device=device, dtype=dtype)
            * (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / width)
        )
        result = torch.zeros(length, width, device=device, dtype=dtype)
        result[:, 0::2] = torch.sin(position * scale)
        result[:, 1::2] = torch.cos(position * scale[: result[:, 1::2].shape[1]])
        return result

    def forward(self, signals: torch.Tensor, present: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.encoder(signals, present)
        ages = observation_age(present).to(encoded.device)
        consistency = encoded.new_zeros(())
        if self.architecture == "grud":
            state, trajectory = self.forward_path(encoded, present, ages)
        elif self.architecture == "brits":
            forward_state, forward_trajectory = self.forward_path(encoded, present, ages)
            reverse_present = present.flip(1)
            reverse_ages = observation_age(reverse_present).to(encoded.device)
            backward_state, backward_reverse = self.backward_path(
                encoded.flip(1), reverse_present, reverse_ages
            )
            backward_trajectory = backward_reverse.flip(1)
            state = self.path_fusion(torch.cat([forward_state, backward_state], dim=-1))
            trajectory = 0.5 * (forward_trajectory + backward_trajectory)
            consistency = (forward_trajectory - backward_trajectory).abs().mean()
        elif self.architecture == "mtan":
            state, trajectory = self.mtan_path(encoded, present, ages)
        elif self.architecture == "raindrop":
            state, trajectory = self.raindrop_path(encoded, present, ages)
        elif self.architecture == "saits":
            state, trajectory, consistency = self.saits_path(encoded, present, ages)
        elif self.architecture == "patchtst":
            state, trajectory = self.patchtst_path(encoded, present, ages)
        elif self.architecture == "rssm":
            state, trajectory, consistency = self.rssm_path(encoded, present, ages)
        else:
            batch, history, modalities, feature_dim = encoded.shape
            flattened = encoded.reshape(batch, history, modalities * feature_dim)
            missingness = torch.cat(
                [present.to(encoded.dtype), torch.log1p(ages)], dim=-1
            )
            tokens = self.observation_projection(flattened) + self.missingness_projection(missingness)
            all_missing = ~present.any(dim=-1)
            tokens = tokens + all_missing.to(tokens.dtype).unsqueeze(-1) * self.all_missing_token
            tokens = tokens + self._positions(
                history, self.hidden_dim, tokens.device, tokens.dtype
            ).unsqueeze(0)
            trajectory = self.temporal_norm(self.temporal(tokens))
            state = trajectory[:, -1]

        state = self.dropout(state)
        return {
            "current_logits": self.current_head(state),
            "future_logits": torch.stack(
                [head(state) for head in self.future_stage_heads], dim=1
            ),
            "future_physiology": torch.stack(
                [head(state) for head in self.future_physiology_heads], dim=1
            ),
            "trajectory": trajectory,
            "consistency_loss": consistency,
        }


def build_dynamic_baseline(config: Dict[str, object]) -> DynamicMissingOutcomeBaseline:
    data = config["data"]
    physiology = config["physiology"]
    model = config["model"]
    return DynamicMissingOutcomeBaseline(
        architecture=str(model["architecture"]),
        modalities=len(data["modalities"]),
        horizons=data["future_horizons"],
        physiology_features=len(physiology["feature_names"]),
        num_classes=int(data.get("num_classes", 5)),
        feature_dim=int(model.get("feature_dim", 32)),
        hidden_dim=int(model.get("hidden_dim", 96)),
        layers=int(model.get("layers", 2)),
        heads=int(model.get("heads", 4)),
        dropout=float(model.get("dropout", 0.1)),
        patch_size=int(model.get("patch_size", 4)),
        patch_stride=int(model.get("patch_stride", 2)),
        latent_dim=int(model.get("latent_dim", 32)),
    )
