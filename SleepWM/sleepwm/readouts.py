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


class DirectObservationBranch(nn.Module):
    """GRU-D observation branch required by the SleepWM PO17/PO18 readout."""

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
        if architecture != "grud":
            raise ValueError("Only the SleepWM GRU-D observation branch is included")
        self.architecture = architecture
        self.horizons = tuple(int(value) for value in horizons)
        self.encoder = ModalityEpochEncoder(modalities, feature_dim, dropout)
        self.modalities = modalities
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.forward_path = DecayGRUPath(modalities, feature_dim, hidden_dim)

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

    def forward(self, signals: torch.Tensor, present: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.encoder(signals, present)
        ages = observation_age(present).to(encoded.device)
        consistency = encoded.new_zeros(())
        state, trajectory = self.forward_path(encoded, present, ages)

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


def build_direct_branch(config: Dict[str, object]) -> DirectObservationBranch:
    data = config["data"]
    physiology = config["physiology"]
    model = config["model"]
    return DirectObservationBranch(
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


import torch
import torch.nn as nn


class DynamicEventHazardAdapter(nn.Module):
    """Matched cumulative-hazard readout for frozen dynamic baselines."""

    def __init__(
        self,
        state_dim: int,
        modality_count: int,
        num_classes: int,
        physiology_features: int,
        hidden_dim: int = 96,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = (
            state_dim
            + 2 * modality_count
            + 2 * num_classes
            + physiology_features
            + 1
        )
        self.context = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.hazard = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.hazard[-1].weight)
        nn.init.constant_(self.hazard[-1].bias, -2.2)

    def forward(
        self,
        baseline_output: dict[str, torch.Tensor],
        present: torch.Tensor,
        ages: torch.Tensor,
        horizons: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        future_logits = baseline_output["future_logits"]
        batch, horizon_count, _ = future_logits.shape
        state = baseline_output["trajectory"][:, -1].unsqueeze(1).expand(
            -1, horizon_count, -1
        )
        current_probability = baseline_output["current_logits"].softmax(dim=-1)
        current_probability = current_probability.unsqueeze(1).expand(
            -1, horizon_count, -1
        )
        future_probability = future_logits.softmax(dim=-1)
        metadata = torch.cat(
            (
                present[:, -1].to(dtype=state.dtype),
                torch.log1p(ages[:, -1]).to(dtype=state.dtype),
            ),
            dim=-1,
        ).unsqueeze(1).expand(-1, horizon_count, -1)
        horizon_feature = (
            torch.log1p(horizons.to(dtype=state.dtype))
            / torch.log1p(horizons.max().clamp_min(1).to(dtype=state.dtype))
        ).reshape(1, horizon_count, 1).expand(batch, -1, -1)
        context = torch.cat(
            (
                state,
                metadata,
                current_probability,
                future_probability,
                baseline_output["future_physiology"],
                horizon_feature,
            ),
            dim=-1,
        )
        hidden = self.context(context)
        hidden, _ = self.temporal(hidden)
        interval_hazard = torch.sigmoid(self.hazard(hidden).squeeze(-1))
        risk = 1.0 - torch.cumprod(1.0 - interval_hazard, dim=1)
        return {
            "transition_risk": risk.clamp(1e-5, 1.0 - 1e-5),
            "interval_hazard": interval_hazard,
        }


from typing import Dict

import torch
import torch.nn as nn


class LatentHazardSafeAdapter(nn.Module):
    """Read future outcomes from belief dynamics with a guarded direct physio path."""

    def __init__(
        self,
        state_dim: int,
        modality_count: int,
        num_classes: int,
        physiology_features: int,
        hidden_dim: int = 128,
        maximum_physiology_residual: float = 0.35,
        initial_direct_gate: float = 0.65,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not 0.0 < initial_direct_gate < 1.0:
            raise ValueError("initial_direct_gate must be between zero and one")

        metadata_dim = 3 * modality_count
        latent_input = (
            3 * state_dim
            + num_classes
            + physiology_features
            + metadata_dim
            + 2
        )
        physiology_input = hidden_dim + 3 * physiology_features

        self.latent_projection = nn.Sequential(
            nn.LayerNorm(latent_input),
            nn.Linear(latent_input, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.trajectory_rnn = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.hazard_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.physiology_context = nn.Sequential(
            nn.LayerNorm(physiology_input),
            nn.Linear(physiology_input, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.physiology_gate = nn.Linear(hidden_dim, physiology_features)
        self.physiology_residual = nn.Linear(hidden_dim, physiology_features)

        gate_bias = torch.logit(torch.tensor(float(initial_direct_gate)))
        nn.init.zeros_(self.hazard_head[-1].weight)
        nn.init.constant_(self.hazard_head[-1].bias, -2.2)
        nn.init.zeros_(self.physiology_gate.weight)
        nn.init.constant_(self.physiology_gate.bias, float(gate_bias))
        nn.init.zeros_(self.physiology_residual.weight)
        nn.init.zeros_(self.physiology_residual.bias)

        self.maximum_physiology_residual = float(maximum_physiology_residual)

    @staticmethod
    def _horizon_scalar(
        value: torch.Tensor,
        horizon_count: int,
    ) -> torch.Tensor:
        if value.ndim == 1:
            value = value[:, None, None]
        elif value.ndim == 2:
            value = value.unsqueeze(-1)
        if value.shape[1] == 1:
            value = value.expand(-1, horizon_count, -1)
        elif value.shape[1] != horizon_count:
            value = value.mean(dim=1, keepdim=True).expand(-1, horizon_count, -1)
        return value

    def forward(
        self,
        sleep_output: Dict[str, torch.Tensor],
        sleep_stage_logits: torch.Tensor,
        sleep_physiology: torch.Tensor,
        direct_physiology: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        belief = sleep_output["predicted_states"]
        base = sleep_output["belief_base_predicted_states"]
        current = sleep_output["belief_trajectory"][:, -1]
        reliability = sleep_output["observation_reliability"]
        freshness = sleep_output["observation_freshness"]
        age = sleep_output["observation_age_epochs"]
        log_variance = sleep_output["recursive_log_variance"]
        horizons = sleep_output["recursive_horizons"].to(dtype=belief.dtype)
        batch, horizon_count, _ = belief.shape

        metadata = torch.cat(
            (
                reliability,
                freshness,
                age / float(sleep_output["belief_trajectory"].shape[1]),
            ),
            dim=-1,
        ).unsqueeze(1).expand(-1, horizon_count, -1)
        current = current.unsqueeze(1).expand(-1, horizon_count, -1)
        horizon_feature = (
            torch.log1p(horizons) / torch.log1p(horizons.max().clamp_min(1.0))
        ).reshape(1, horizon_count, 1).expand(batch, -1, -1)
        uncertainty = self._horizon_scalar(log_variance, horizon_count)
        if uncertainty.shape[-1] != 1:
            uncertainty = uncertainty.mean(dim=-1, keepdim=True)

        latent_context = torch.cat(
            (
                belief,
                belief - base,
                current,
                sleep_stage_logits.softmax(dim=-1),
                sleep_physiology,
                metadata,
                uncertainty,
                horizon_feature,
            ),
            dim=-1,
        )
        latent_hidden = self.latent_projection(latent_context)
        latent_hidden, _ = self.trajectory_rnn(latent_hidden)

        interval_hazard = torch.sigmoid(self.hazard_head(latent_hidden).squeeze(-1))
        transition_risk = 1.0 - torch.cumprod(1.0 - interval_hazard, dim=1)

        physiology_context = torch.cat(
            (
                latent_hidden,
                sleep_physiology,
                direct_physiology,
                (direct_physiology - sleep_physiology).abs(),
            ),
            dim=-1,
        )
        physiology_hidden = self.physiology_context(physiology_context)
        physiology_gate = torch.sigmoid(self.physiology_gate(physiology_hidden))
        physiology_residual = self.maximum_physiology_residual * torch.tanh(
            self.physiology_residual(physiology_hidden)
        )
        future_physiology = (
            sleep_physiology
            + physiology_gate * (direct_physiology - sleep_physiology)
            + physiology_residual
        )
        return {
            "stage_logits": sleep_stage_logits,
            "future_physiology": future_physiology,
            "transition_risk": transition_risk.clamp(1e-5, 1.0 - 1e-5),
            "interval_hazard": interval_hazard,
            "physiology_gate": physiology_gate,
            "physiology_residual": physiology_residual,
        }


import torch
import torch.nn as nn


def _safe_logit(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(1e-5, 1.0 - 1e-5)
    return torch.log(value) - torch.log1p(-value)


class ReliabilityGatedEventCorrection(nn.Module):
    """Correct latent hazards with direct evidence when observations are reliable."""

    def __init__(
        self,
        modality_count: int,
        num_classes: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        initial_direct_gate: float = 0.45,
        maximum_logit_residual: float = 1.0,
    ) -> None:
        super().__init__()
        if not 0.0 < initial_direct_gate < 1.0:
            raise ValueError("initial_direct_gate must be between zero and one")

        input_dim = 3 + 3 * modality_count + 2 * num_classes + 3
        self.context = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.gate = nn.Linear(hidden_dim, 1)
        self.residual = nn.Linear(hidden_dim, 1)

        gate_bias = torch.logit(torch.tensor(float(initial_direct_gate)))
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_bias))
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        self.maximum_logit_residual = float(maximum_logit_residual)

    @staticmethod
    def _uncertainty(
        recursive_log_variance: torch.Tensor,
        horizon_count: int,
    ) -> torch.Tensor:
        value = recursive_log_variance
        if value.ndim == 1:
            value = value[:, None, None]
        elif value.ndim == 2:
            value = value.unsqueeze(-1)
        if value.shape[1] == 1:
            value = value.expand(-1, horizon_count, -1)
        elif value.shape[1] != horizon_count:
            value = value.mean(dim=1, keepdim=True).expand(-1, horizon_count, -1)
        if value.shape[-1] != 1:
            value = value.mean(dim=-1, keepdim=True)
        return value

    def forward(
        self,
        sleep_output: dict[str, torch.Tensor],
        future_stage_logits: torch.Tensor,
        latent_interval_hazard: torch.Tensor,
        direct_interval_hazard: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, horizon_count = latent_interval_hazard.shape
        dtype = latent_interval_hazard.dtype
        reliability = sleep_output["observation_reliability"]
        freshness = sleep_output["observation_freshness"]
        age = sleep_output["observation_age_epochs"]
        history_count = float(sleep_output["belief_trajectory"].shape[1])
        metadata = torch.cat(
            (reliability, freshness, age / max(history_count, 1.0)), dim=-1
        ).unsqueeze(1).expand(-1, horizon_count, -1)

        current_logits = (
            sleep_output["belief_current_stage_logits"]
            if "belief_current_stage_logits" in sleep_output
            else sleep_output["current_stage_logits"]
        )
        current_probability = current_logits.softmax(dim=-1)
        current_probability = current_probability.unsqueeze(1).expand(
            -1, horizon_count, -1
        )
        future_probability = future_stage_logits.softmax(dim=-1)
        stage_change = 1.0 - (current_probability * future_probability).sum(
            dim=-1, keepdim=True
        )

        uncertainty = self._uncertainty(
            sleep_output["recursive_log_variance"], horizon_count
        )
        horizons = sleep_output["recursive_horizons"].to(dtype=dtype)
        horizon_feature = (
            torch.log1p(horizons)
            / torch.log1p(horizons.max().clamp_min(1.0))
        ).reshape(1, horizon_count, 1).expand(batch, -1, -1)

        latent_logit = _safe_logit(latent_interval_hazard).unsqueeze(-1)
        direct_logit = _safe_logit(direct_interval_hazard).unsqueeze(-1)
        context = torch.cat(
            (
                latent_logit,
                direct_logit,
                direct_logit - latent_logit,
                metadata,
                current_probability,
                future_probability,
                stage_change,
                uncertainty,
                horizon_feature,
            ),
            dim=-1,
        )
        hidden = self.context(context)
        hidden, _ = self.temporal(hidden)
        direct_gate = torch.sigmoid(self.gate(hidden)).squeeze(-1)
        residual = self.maximum_logit_residual * torch.tanh(
            self.residual(hidden).squeeze(-1)
        )
        corrected_logit = (
            latent_logit.squeeze(-1)
            + direct_gate * (direct_logit.squeeze(-1) - latent_logit.squeeze(-1))
            + residual
        )
        interval_hazard = torch.sigmoid(corrected_logit)
        transition_risk = 1.0 - torch.cumprod(1.0 - interval_hazard, dim=1)
        return {
            "transition_risk": transition_risk.clamp(1e-5, 1.0 - 1e-5),
            "interval_hazard": interval_hazard,
            "direct_gate": direct_gate,
            "logit_residual": residual,
        }


from typing import Dict

import torch
import torch.nn as nn


class TrajectoryOutcomeAdapter(nn.Module):
    """Read downstream outcomes from a frozen recursive belief trajectory."""

    def __init__(
        self,
        state_dim: int,
        modality_count: int,
        num_classes: int,
        physiology_features: int,
        hidden_dim: int = 128,
        maximum_stage_delta: float = 2.0,
        maximum_physiology_delta: float = 1.0,
    ) -> None:
        super().__init__()
        if min(
            state_dim,
            modality_count,
            num_classes,
            physiology_features,
            hidden_dim,
        ) < 1:
            raise ValueError("outcome adapter dimensions must be positive")
        if min(maximum_stage_delta, maximum_physiology_delta) <= 0.0:
            raise ValueError("outcome adapter delta limits must be positive")
        input_dim = 2 * state_dim + 3 * modality_count + 2
        self.maximum_stage_delta = float(maximum_stage_delta)
        self.maximum_physiology_delta = float(maximum_physiology_delta)
        self.shared = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.stage_head = nn.Linear(hidden_dim, num_classes)
        self.physiology_head = nn.Linear(hidden_dim, physiology_features)
        nn.init.zeros_(self.stage_head.weight)
        nn.init.zeros_(self.stage_head.bias)
        nn.init.zeros_(self.physiology_head.weight)
        nn.init.zeros_(self.physiology_head.bias)

    def forward(self, output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        belief = output["predicted_states"]
        base = output["belief_base_predicted_states"]
        reliability = output["observation_reliability"]
        freshness = output["observation_freshness"]
        age = output["observation_age_epochs"]
        log_variance = output["recursive_log_variance"]
        horizons = output["recursive_horizons"].to(dtype=belief.dtype)
        batch, horizon_count, _ = belief.shape
        if base.shape != belief.shape or log_variance.shape != belief.shape[:2]:
            raise ValueError("belief outcome tensors have incompatible shapes")
        metadata = torch.cat(
            (
                reliability,
                freshness,
                age / float(output["belief_trajectory"].shape[1]),
            ),
            dim=-1,
        ).unsqueeze(1).expand(-1, horizon_count, -1)
        horizon_feature = (
            torch.log1p(horizons) / torch.log1p(horizons.max().clamp_min(1.0))
        ).reshape(1, horizon_count, 1).expand(batch, -1, -1)
        context = torch.cat(
            (
                belief,
                belief - base,
                metadata,
                log_variance.unsqueeze(-1),
                horizon_feature,
            ),
            dim=-1,
        )
        hidden = self.shared(context)
        stale_gate = (1.0 - freshness).amax(dim=-1).reshape(batch, 1, 1)
        stage_delta = stale_gate * self.maximum_stage_delta * torch.tanh(
            self.stage_head(hidden)
        )
        physiology_delta = (
            stale_gate
            * self.maximum_physiology_delta
            * torch.tanh(self.physiology_head(hidden))
        )
        return {
            "stage_logits": output["stage_logits"] + stage_delta,
            "future_physiology": output["future_physiology"] + physiology_delta,
            "stage_delta": stage_delta,
            "physiology_delta": physiology_delta,
            "stale_gate": stale_gate.squeeze(-1).squeeze(-1),
        }
