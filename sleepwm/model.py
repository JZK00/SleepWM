from __future__ import annotations

"""The six-component SleepWM inference system used by the final readout.

Checkpoint files are supplied separately. No data are loaded by this module.
"""
from pathlib import Path
import sys

import torch
from torch import nn

from sleepwm.readouts import build_direct_branch, observation_age
from sleepwm.readouts import DynamicEventHazardAdapter
from sleepwm.readouts import ReliabilityGatedEventCorrection
from sleepwm.training.recursive_belief_filter import build_student
from sleepwm.training.trajectory_outcome_adapter import build_adapter
from sleepwm.training.sleepwm_event_correction import build_po17
from sleepwm.engine import load_checkpoint


COMPONENTS = ("state", "outcome", "direct", "latent_hazard", "direct_event", "event_correction")


class SleepWM(nn.Module):
    def __init__(self, payloads):
        super().__init__()
        missing = set(COMPONENTS) - set(payloads)
        if missing:
            raise ValueError(f"Missing checkpoint components: {sorted(missing)}")
        state, outcome, direct, latent, event, correction = (payloads[k] for k in COMPONENTS)
        self.config = state["config"]
        self.horizons = tuple(int(h) for h in self.config["data"]["future_horizons"])
        if tuple(direct["config"]["data"]["future_horizons"]) != self.horizons:
            raise ValueError("State and direct-observation horizons differ")
        self.state_model = build_student(self.config)
        self.outcome_adapter = build_adapter(self.state_model, outcome["config"])
        self.direct_model = build_direct_branch(direct["config"])
        self.latent_adapter = build_po17(latent["config"], self.state_model)
        self.direct_event = DynamicEventHazardAdapter(
            state_dim=int(direct["config"]["model"]["hidden_dim"]),
            modality_count=len(self.config["data"]["modalities"]),
            num_classes=int(self.config["data"]["num_classes"]),
            physiology_features=len(self.config["physiology"]["feature_names"]),
            hidden_dim=int(direct["config"]["model"]["hidden_dim"]),
            dropout=float(direct["config"]["model"].get("dropout", 0.1)),
        )
        self.event_correction = ReliabilityGatedEventCorrection(
            modality_count=len(self.config["data"]["modalities"]),
            num_classes=int(self.config["data"]["num_classes"]),
        )
        for module, payload, key in (
            (self.state_model, state, "model_state"),
            (self.outcome_adapter, outcome, "adapter_state"),
            (self.direct_model, direct, "model_state"),
            (self.latent_adapter, latent, "adapter_state"),
            (self.direct_event, event, "adapter_state"),
            (self.event_correction, correction, "adapter_state"),
        ):
            module.load_state_dict(payload[key], strict=True)
        self.requires_grad_(False).eval()

    @classmethod
    def from_checkpoints(cls, paths, device="cpu"):
        """Load trusted research checkpoints, including their configuration mappings."""
        return cls({key: load_checkpoint(paths[key]) for key in COMPONENTS}).to(device)

    @torch.inference_mode()
    def forward(self, signals, present):
        """signals: [B,T,M,S], present: [B,T,M]; unavailable signals must be zero."""
        if signals.ndim != 4 or present.shape != signals.shape[:3]:
            raise ValueError("Expected signals [batch, history, modalities, samples] and matching mask")
        state = self.state_model.rollout_context_horizons(signals, present.bool(), self.horizons)
        adapted = self.outcome_adapter(state)
        direct = self.direct_model(signals, present.bool())
        latent = self.latent_adapter(state, adapted["stage_logits"], adapted["future_physiology"], direct["future_physiology"])
        direct_event = self.direct_event(
            direct, present.bool(), observation_age(present.bool()).to(signals.device),
            torch.tensor(self.horizons, device=signals.device),
        )
        corrected = self.event_correction(state, latent["stage_logits"], latent["interval_hazard"], direct_event["interval_hazard"])
        return {
            "stage_logits": latent["stage_logits"],
            "stage_probabilities": latent["stage_logits"].softmax(-1),
            "future_physiology": latent["future_physiology"],
            "transition_risk": corrected["transition_risk"],
            "interval_hazard": corrected["interval_hazard"],
        }
