from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from .models import sinusoidal_positions


class EpochConvEncoder(nn.Module):
    """Lightweight raw-waveform encoder shared by causal sequence baselines."""

    def __init__(self, modalities: int, d_model: int, dropout: float) -> None:
        super().__init__()
        hidden = max(32, d_model // 2)
        self.signal_encoder = nn.Sequential(
            nn.Conv1d(modalities, hidden, 25, stride=8, padding=12),
            nn.GroupNorm(4, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 9, stride=4, padding=4),
            nn.GroupNorm(4, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, d_model, 7, stride=4, padding=3),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
        )
        self.presence_projection = nn.Sequential(
            nn.Linear(modalities, d_model),
            nn.Tanh(),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, signals: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        if signals.ndim != 4 or present.shape != signals.shape[:3]:
            raise ValueError("signals/present must be [batch, history, modalities, samples]/[batch, history, modalities]")
        batch, history, modalities, samples = signals.shape
        masked = signals * present.to(signals.dtype).unsqueeze(-1)
        encoded = self.signal_encoder(masked.reshape(batch * history, modalities, samples))
        encoded = encoded + self.presence_projection(present.reshape(batch * history, modalities).to(signals.dtype))
        return self.norm(encoded).reshape(batch, history, -1)


class CausalSequenceBaseline(nn.Module):
    """History-only GRU or Transformer baseline with direct multi-horizon heads."""

    def __init__(
        self,
        modalities: int,
        horizons: Sequence[int],
        architecture: str = "gru",
        d_model: int = 96,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        if architecture not in {"gru", "transformer"}:
            raise ValueError("architecture must be 'gru' or 'transformer'")
        self.architecture = architecture
        self.horizons = tuple(int(value) for value in horizons)
        self.encoder = EpochConvEncoder(modalities, d_model, dropout)
        if architecture == "gru":
            self.temporal = nn.GRU(
                d_model,
                d_model,
                num_layers=layers,
                dropout=dropout if layers > 1 else 0.0,
                batch_first=True,
            )
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=layers)
        self.current_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))
        self.future_heads = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes)) for _ in self.horizons]
        )

    def forward(self, history_signals: torch.Tensor, history_present: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.encoder(history_signals, history_present)
        if self.architecture == "transformer":
            encoded = encoded + sinusoidal_positions(
                encoded.shape[1], encoded.shape[2], encoded.device, encoded.dtype
            ).unsqueeze(0)
            temporal = self.temporal(encoded)
        else:
            temporal, _ = self.temporal(encoded)
        state = temporal[:, -1]
        return {
            "current_logits": self.current_head(state),
            "future_logits": torch.stack([head(state) for head in self.future_heads], dim=1),
            "state": state,
        }


def build_sequence_baseline(config: Dict[str, Dict[str, object]]) -> CausalSequenceBaseline:
    data = config["data"]
    model = config["model"]
    return CausalSequenceBaseline(
        modalities=len(data["modalities"]),  # type: ignore[arg-type]
        horizons=data["future_horizons"],  # type: ignore[arg-type]
        architecture=str(model["name"]),
        d_model=int(model.get("d_model", 96)),
        layers=int(model.get("layers", 2)),
        heads=int(model.get("heads", 4)),
        dropout=float(model.get("dropout", 0.1)),
        num_classes=int(data.get("num_classes", 5)),
    )
