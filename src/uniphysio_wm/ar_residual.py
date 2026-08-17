from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

from .linear_dynamics import fit_ridge_map


class FeatureAR(nn.Module):
    """Direct horizon-specific AR maps stored as fixed torch buffers."""

    def __init__(
        self,
        x_mean: torch.Tensor,
        x_scale: torch.Tensor,
        y_mean: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> None:
        super().__init__()
        expected = coefficients.shape
        if coefficients.ndim != 3 or x_mean.shape != expected or x_scale.shape != expected:
            raise ValueError("AR statistics must be [horizons, features, history]")
        if y_mean.shape != expected[:2]:
            raise ValueError("AR target means must be [horizons, features]")
        self.register_buffer("x_mean", x_mean.to(torch.float32))
        self.register_buffer("x_scale", x_scale.to(torch.float32))
        self.register_buffer("y_mean", y_mean.to(torch.float32))
        self.register_buffer("coefficients", coefficients.to(torch.float32))

    @property
    def horizon_count(self) -> int:
        return int(self.y_mean.shape[0])

    @property
    def feature_count(self) -> int:
        return int(self.y_mean.shape[1])

    @property
    def history_epochs(self) -> int:
        return int(self.x_mean.shape[2])

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.shape[1:] != (self.history_epochs, self.feature_count):
            raise ValueError("AR history must be [batch, history, features]")
        values = history.transpose(1, 2).unsqueeze(1)
        normalized = (values - self.x_mean.unsqueeze(0)) / self.x_scale.unsqueeze(0)
        return (normalized * self.coefficients.unsqueeze(0)).sum(dim=-1) + self.y_mean


def fit_feature_ar(
    history: torch.Tensor,
    target: torch.Tensor,
    target_valid: torch.Tensor,
    alpha: float = 1.0,
) -> FeatureAR:
    if history.ndim != 3 or target.ndim != 3 or target_valid.shape != target.shape:
        raise ValueError("feature AR tensors must be aligned rank-three arrays")
    if history.shape[0] != target.shape[0] or history.shape[2] != target.shape[2]:
        raise ValueError("feature AR history and target dimensions do not align")
    x = history.detach().cpu().numpy().astype(np.float64)
    y = target.detach().cpu().numpy().astype(np.float64)
    valid = target_valid.detach().cpu().numpy().astype(bool)
    horizon_count, feature_count = y.shape[1:]
    history_epochs = x.shape[1]
    x_mean = np.empty((horizon_count, feature_count, history_epochs), dtype=np.float32)
    x_scale = np.empty_like(x_mean)
    y_mean = np.empty((horizon_count, feature_count), dtype=np.float32)
    coefficients = np.empty_like(x_mean)
    for horizon in range(horizon_count):
        for feature in range(feature_count):
            selected = valid[:, horizon, feature]
            model = fit_ridge_map(
                x[selected, :, feature],
                y[selected, horizon, feature : feature + 1],
                alpha,
            )
            x_mean[horizon, feature] = model.x_mean
            x_scale[horizon, feature] = model.x_scale
            y_mean[horizon, feature] = model.y_mean[0]
            coefficients[horizon, feature] = model.coefficients[:, 0]
    return FeatureAR(
        torch.from_numpy(x_mean),
        torch.from_numpy(x_scale),
        torch.from_numpy(y_mean),
        torch.from_numpy(coefficients),
    )


def impute_feature_history(
    history: torch.Tensor,
    valid: torch.Tensor,
    standardized_median: torch.Tensor,
) -> torch.Tensor:
    if history.shape != valid.shape or standardized_median.shape != history.shape[-1:]:
        raise ValueError("history, validity, and median shapes do not align")
    return torch.where(
        valid,
        history,
        standardized_median.to(history).reshape(1, 1, -1),
    )


class CrossModalResidual(nn.Module):
    """Nonlinear residual conditioned on all feature groups and latent context."""

    def __init__(
        self,
        history_epochs: int,
        feature_count: int,
        state_dim: int,
        modality_count: int,
        horizon_count: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.history_epochs = int(history_epochs)
        self.feature_count = int(feature_count)
        self.horizon_count = int(horizon_count)
        input_dim = (
            2 * self.history_epochs * self.feature_count
            + self.feature_count
            + int(state_dim)
            + 2 * int(modality_count)
        )
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.horizon_embedding = nn.Parameter(
            torch.empty(self.horizon_count, hidden_dim)
        )
        nn.init.normal_(self.horizon_embedding, std=0.02)
        self.residual_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.feature_count),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(
        self,
        history: torch.Tensor,
        history_valid: torch.Tensor,
        ar_prediction: torch.Tensor,
        predicted_states: torch.Tensor,
        reliability: torch.Tensor,
        availability: torch.Tensor,
    ) -> torch.Tensor:
        batch = history.shape[0]
        if history.shape[1:] != (self.history_epochs, self.feature_count):
            raise ValueError("residual history shape is incompatible")
        if ar_prediction.shape[1:] != (self.horizon_count, self.feature_count):
            raise ValueError("residual AR prediction shape is incompatible")
        static = torch.cat(
            (
                history.reshape(batch, -1),
                history_valid.to(history.dtype).reshape(batch, -1),
                reliability,
                availability,
            ),
            dim=-1,
        ).unsqueeze(1).expand(-1, self.horizon_count, -1)
        features = torch.cat(
            (static, ar_prediction, predicted_states), dim=-1
        )
        hidden = self.input_projection(features) + self.horizon_embedding.unsqueeze(0)
        return self.residual_head(hidden)


class ARAnchoredCrossModalResidual(nn.Module):
    """A fixed AR base plus a gated nonlinear cross-modal residual."""

    def __init__(
        self,
        ar_model: FeatureAR,
        state_dim: int,
        modality_count: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        minimum_modality_fraction: float = 1e-3,
    ) -> None:
        super().__init__()
        self.ar_model = ar_model
        self.minimum_modality_fraction = float(minimum_modality_fraction)
        self.residual = CrossModalResidual(
            ar_model.history_epochs,
            ar_model.feature_count,
            state_dim,
            modality_count,
            ar_model.horizon_count,
            hidden_dim,
            dropout,
        )

    def residual_parameters(self):
        yield from self.residual.parameters()

    def forward(
        self,
        history: torch.Tensor,
        history_valid: torch.Tensor,
        predicted_states: torch.Tensor,
        reliability: torch.Tensor,
        availability: torch.Tensor,
        base_prediction: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ar_prediction = self.ar_model(history)
        residual = self.residual(
            history,
            history_valid,
            ar_prediction,
            predicted_states,
            reliability,
            availability,
        )
        active = (availability >= self.minimum_modality_fraction).all(dim=-1)
        prediction = torch.where(
            active.reshape(-1, 1, 1),
            ar_prediction + residual,
            base_prediction,
        )
        return {
            "future_physiology": prediction,
            "ar_future_physiology": ar_prediction,
            "cross_modal_residual": residual,
            "ar_residual_active": active,
        }


def group_masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    group_sizes: Sequence[int],
) -> torch.Tensor:
    losses = []
    offset = 0
    for size in group_sizes:
        selected = slice(offset, offset + int(size))
        element = torch.nn.functional.smooth_l1_loss(
            prediction[..., selected], target[..., selected], reduction="none"
        )
        weight = valid[..., selected].to(element.dtype)
        losses.append((element * weight).sum() / weight.sum().clamp_min(1.0))
        offset += int(size)
    return torch.stack(losses).mean()
