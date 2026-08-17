from __future__ import annotations

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
