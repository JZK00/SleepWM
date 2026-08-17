from __future__ import annotations

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
