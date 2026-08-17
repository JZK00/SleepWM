from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import GatedRecursiveTaskAwareWaveformWorldModel


def _inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("softplus initialization must be positive")
    return math.log(math.expm1(value))


class FreshnessAwareCarryCorrectWorldModel(
    GatedRecursiveTaskAwareWaveformWorldModel
):
    """Carry latent state through interruptions and correct it with fresh observations."""

    uses_freshness_aware_carry_correct = True

    def __init__(
        self,
        *args,
        partial_filter_hidden_dim: int = 128,
        initial_freshness_decay: float = 0.35,
        initial_uncertainty_age_scale: float = 0.15,
        initial_uncertainty_horizon_scale: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        state_dim = self.encoder.config.d_model
        modality_count = len(self.encoder.config.modalities)
        hidden_dim = int(partial_filter_hidden_dim)
        if hidden_dim < 1:
            raise ValueError("partial-observation hidden dimensions must be positive")

        self.log_freshness_decay = nn.Parameter(
            torch.full(
                (modality_count,), _inverse_softplus(initial_freshness_decay)
            )
        )
        carry_input_dim = 2 * state_dim + 2 * modality_count
        self.carry_state_filter = nn.Sequential(
            nn.LayerNorm(carry_input_dim),
            nn.Linear(carry_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
        nn.init.zeros_(self.carry_state_filter[-1].weight)
        nn.init.zeros_(self.carry_state_filter[-1].bias)

        uncertainty_input_dim = state_dim + 2 * modality_count + 1
        self.partial_uncertainty_head = nn.Sequential(
            nn.LayerNorm(uncertainty_input_dim),
            nn.Linear(uncertainty_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.partial_uncertainty_head[-1].weight)
        nn.init.zeros_(self.partial_uncertainty_head[-1].bias)
        self.partial_uncertainty_state_norm = nn.LayerNorm(state_dim)
        self.uncertainty_bias = nn.Parameter(torch.tensor(-2.0))
        self.uncertainty_age_log_scale = nn.Parameter(
            torch.full(
                (modality_count,),
                _inverse_softplus(initial_uncertainty_age_scale),
            )
        )
        self.uncertainty_horizon_log_scale = nn.Parameter(
            torch.tensor(_inverse_softplus(initial_uncertainty_horizon_scale))
        )

    def partial_observation_parameters(self):
        modules = (
            self.carry_state_filter,
            self.partial_uncertainty_head,
            self.partial_uncertainty_state_norm,
        )
        yield self.log_freshness_decay
        yield self.uncertainty_bias
        yield self.uncertainty_age_log_scale
        yield self.uncertainty_horizon_log_scale
        for module in modules:
            yield from module.parameters()

    def _observation_metadata(
        self,
        history_signals: torch.Tensor,
        history_present: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if history_present is None:
            present = torch.ones(
                history_signals.shape[:3],
                device=history_signals.device,
                dtype=torch.bool,
            )
        else:
            present = history_present.bool()
        epoch_count = present.shape[1]
        indices = torch.arange(epoch_count, device=present.device).reshape(1, -1, 1)
        last_indices = torch.where(present, indices, -torch.ones_like(indices)).amax(dim=1)
        has_observation = last_indices >= 0
        age_epochs = torch.where(
            has_observation,
            (epoch_count - 1 - last_indices).to(history_signals.dtype),
            history_signals.new_full(last_indices.shape, float(epoch_count)),
        )
        decay = F.softplus(self.log_freshness_decay).reshape(1, -1)
        freshness = torch.exp(-decay * age_epochs) * has_observation.to(
            history_signals.dtype
        )
        availability = present.to(history_signals.dtype).mean(dim=1)
        return age_epochs, freshness, availability, last_indices.clamp_min(0)

    @staticmethod
    def _last_observed_states(
        modality_states: torch.Tensor,
        last_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, modalities, state_dim = modality_states.shape
        gather_indices = last_indices.reshape(batch, 1, modalities, 1).expand(
            -1, 1, -1, state_dim
        )
        return modality_states.gather(dim=1, index=gather_indices).squeeze(1)

    def _observation_adjustment(
        self,
        output: Dict[str, torch.Tensor],
        history_signals: torch.Tensor,
        history_present: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        age_epochs, freshness, availability, last_indices = self._observation_metadata(
            history_signals, history_present
        )
        last_states = self._last_observed_states(
            output["history_modality_states"], last_indices
        )
        signal_scale = history_signals.float().std(dim=-1).mean(dim=1)
        quality = signal_scale / (1.0 + signal_scale)
        weighted_adjustment = torch.zeros_like(output["history_state"])
        weighted_last_state = torch.zeros_like(output["history_state"])
        reliability_values = []
        denominator = availability.new_zeros(availability.shape[0], 1)
        for modality_index, modality in enumerate(self.encoder.config.modalities):
            state = last_states[:, modality_index]
            metadata = torch.stack(
                (availability[:, modality_index], quality[:, modality_index]), dim=-1
            ).to(dtype=state.dtype)
            base_reliability = self.observation_reliability_heads[modality](
                torch.cat((state, metadata), dim=-1)
            ).sigmoid()
            reliability = (
                base_reliability
                * freshness[:, modality_index].unsqueeze(-1)
                * (availability[:, modality_index] > 0).to(state.dtype).unsqueeze(-1)
            )
            weighted_adjustment = weighted_adjustment + reliability * self.observation_state_adapters[
                modality
            ](state)
            weighted_last_state = weighted_last_state + reliability * state
            denominator = denominator + reliability
            reliability_values.append(reliability.squeeze(-1))

        observation_correction = weighted_adjustment / denominator.clamp_min(1e-3)
        carried_observation = weighted_last_state / denominator.clamp_min(1e-3)
        age_normalized = age_epochs / float(history_signals.shape[1])
        carry_context = torch.cat(
            (
                output["history_state"],
                carried_observation,
                age_normalized,
                freshness,
            ),
            dim=-1,
        )
        stale_gate = (1.0 - freshness).amax(dim=-1, keepdim=True)
        carry_delta = stale_gate * self.carry_state_filter(carry_context)
        adjustment = carry_delta + observation_correction
        return adjustment, torch.stack(reliability_values, dim=-1), quality

    def _partial_log_variance(
        self,
        corrected_state: torch.Tensor,
        age_epochs: torch.Tensor,
        reliability: torch.Tensor,
        horizons: Sequence[int],
    ) -> torch.Tensor:
        selected = torch.as_tensor(
            tuple(int(value) for value in horizons),
            device=corrected_state.device,
            dtype=corrected_state.dtype,
        )
        age_features = torch.log1p(age_epochs)
        age_term = (
            age_features * F.softplus(self.uncertainty_age_log_scale).reshape(1, -1)
        ).sum(dim=-1, keepdim=True)
        horizon_features = torch.log1p(selected).reshape(1, -1)
        horizon_term = F.softplus(self.uncertainty_horizon_log_scale) * horizon_features
        batch = corrected_state.shape[0]
        count = len(selected)
        context = torch.cat(
            (
                self.partial_uncertainty_state_norm(corrected_state),
                age_features,
                1.0 - reliability,
            ),
            dim=-1,
        ).unsqueeze(1).expand(-1, count, -1)
        horizon_context = horizon_features.expand(batch, -1).unsqueeze(-1)
        residual = 0.5 * torch.tanh(
            self.partial_uncertainty_head(
                torch.cat((context, horizon_context), dim=-1)
            ).squeeze(-1)
        )
        return (
            self.uncertainty_bias
            + age_term
            + horizon_term
            + residual
        ).clamp(-6.0, 3.0)

    def rollout_context_horizons(
        self,
        history_signals: torch.Tensor,
        history_present: Optional[torch.Tensor] = None,
        horizons: Optional[Sequence[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        selected_horizons = self.rollout_horizons if horizons is None else tuple(horizons)
        output = super().rollout_context_horizons(
            history_signals, history_present, selected_horizons
        )
        age_epochs, freshness, _, _ = self._observation_metadata(
            history_signals, history_present
        )
        output["observation_age_epochs"] = age_epochs
        output["observation_freshness"] = freshness
        output["recursive_log_variance"] = self._partial_log_variance(
            output["corrected_history_state"],
            age_epochs,
            output["observation_reliability"],
            selected_horizons,
        )
        return output


class RecursiveBeliefCarryCorrectWorldModel(
    FreshnessAwareCarryCorrectWorldModel
):
    """Explicit epoch-wise latent belief prediction and observation correction."""

    uses_recursive_belief_filter = True

    def __init__(
        self,
        *args,
        belief_hidden_dim: int = 128,
        belief_max_delta: float = 0.5,
        belief_use_dynamics: bool = True,
        belief_correction_mode: str = "learned",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        state_dim = self.encoder.config.d_model
        modality_count = len(self.encoder.config.modalities)
        hidden_dim = int(belief_hidden_dim)
        if hidden_dim < 1 or float(belief_max_delta) <= 0.0:
            raise ValueError("belief dimensions and maximum delta must be positive")
        if belief_correction_mode not in {"learned", "ungated"}:
            raise ValueError(
                "belief correction mode must be 'learned' or 'ungated'"
            )
        self.belief_max_delta = float(belief_max_delta)
        self.belief_use_dynamics = bool(belief_use_dynamics)
        self.belief_correction_mode = str(belief_correction_mode)
        self.belief_transition = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
        nn.init.zeros_(self.belief_transition[-1].weight)
        nn.init.zeros_(self.belief_transition[-1].bias)
        correction_input_dim = 2 * state_dim + 2 * modality_count
        self.belief_correction_gate = nn.Sequential(
            nn.LayerNorm(correction_input_dim),
            nn.Linear(correction_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Sigmoid(),
        )

    def belief_parameters(self):
        yield from self.belief_transition.parameters()
        yield from self.belief_correction_gate.parameters()

    def _belief_trajectory(
        self,
        epoch_states: torch.Tensor,
        history_present: Optional[torch.Tensor],
        use_dynamics: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if epoch_states.ndim != 3:
            raise ValueError("epoch states must be [batch, time, dimensions]")
        batch, time, _ = epoch_states.shape
        modality_count = len(self.encoder.config.modalities)
        if history_present is None:
            present = torch.ones(
                batch,
                time,
                modality_count,
                device=epoch_states.device,
                dtype=torch.bool,
            )
        else:
            present = history_present.bool()
        if present.shape != (batch, time, modality_count):
            raise ValueError("history presence does not match epoch states")

        dynamics_enabled = (
            self.belief_use_dynamics if use_dynamics is None else bool(use_dynamics)
        )
        state = epoch_states[:, 0]
        age = (~present[:, 0]).to(epoch_states.dtype)
        states = [state]
        priors = [state]
        gates = [epoch_states.new_ones(state.shape)]
        corrections = [epoch_states.new_zeros(state.shape)]
        for index in range(1, time):
            if dynamics_enabled:
                delta = self.belief_max_delta * torch.tanh(
                    self.belief_transition(state)
                )
                prior = state + delta
            else:
                prior = state
            observed = epoch_states[:, index]
            present_now = present[:, index]
            age = torch.where(
                present_now,
                torch.zeros_like(age),
                age + 1.0,
            )
            age_feature = torch.log1p(age) / math.log1p(float(time))
            availability = present_now.to(epoch_states.dtype)
            any_observed = present_now.any(dim=-1, keepdim=True)
            all_observed = present_now.all(dim=-1, keepdim=True)
            if self.belief_correction_mode == "learned":
                gate = self.belief_correction_gate(
                    torch.cat((prior, observed, availability, age_feature), dim=-1)
                )
                gate = torch.where(
                    all_observed,
                    torch.ones_like(gate),
                    gate * any_observed.to(gate.dtype),
                )
            else:
                gate = any_observed.to(prior.dtype).expand_as(prior)
            correction = gate * (observed - prior)
            state = prior + correction
            states.append(state)
            priors.append(prior)
            gates.append(gate)
            corrections.append(correction)
        return (
            torch.stack(states, dim=1),
            torch.stack(priors, dim=1),
            torch.stack(gates, dim=1),
            torch.stack(corrections, dim=1),
        )

    def rollout_context_horizons(
        self,
        history_signals: torch.Tensor,
        history_present: Optional[torch.Tensor] = None,
        horizons: Optional[Sequence[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        selected_horizons = self.rollout_horizons if horizons is None else tuple(horizons)
        output = super().rollout_context_horizons(
            history_signals, history_present, selected_horizons
        )
        epoch_states = output["history_epoch_states"]
        belief, prior, correction_gate, correction = self._belief_trajectory(
            epoch_states, history_present
        )
        persistence, _, _, _ = self._belief_trajectory(
            epoch_states, history_present, use_dynamics=False
        )
        final_belief = belief[:, -1]
        endpoint_delta = final_belief - epoch_states[:, -1]
        base_predicted_states = output["predicted_states"]
        predicted_states = base_predicted_states + endpoint_delta.unsqueeze(1)
        if self.factorized_transition_head or self.stage_head is None:
            raise ValueError("recursive belief filtering requires the standard stage head")
        stage_logits = output["stage_logits"] + self.stage_head(
            predicted_states
        ) - self.stage_head(base_predicted_states)
        if self.current_stage_head is not None:
            filtered_history_state = output["corrected_history_state"] + endpoint_delta
            current_stage_logits = self.current_stage_head(filtered_history_state)
            base_current_stage_logits = self.current_stage_head(
                output["corrected_history_state"]
            )
            stage_logits = stage_logits + (
                current_stage_logits - base_current_stage_logits
            ).unsqueeze(1)
            output["belief_current_stage_logits"] = current_stage_logits
        else:
            filtered_history_state = output["corrected_history_state"] + endpoint_delta
        task_stage, task_physiology = self._task_residuals(
            predicted_states, output["observation_reliability"]
        )
        base_task_stage, base_task_physiology = self._task_residuals(
            base_predicted_states, output["observation_reliability"]
        )
        stage_logits = stage_logits + task_stage - base_task_stage

        future_by_group = {}
        offset = 0
        for group, size in self.physiology_group_sizes.items():
            current = output["current_physiology"][:, offset : offset + size]
            future_by_group[group] = current.unsqueeze(1) + self.future_physiology_delta_heads[
                group
            ](predicted_states)
            offset += size
        belief_future_physiology = torch.cat(
            [future_by_group[group] for group in self.physiology_group_sizes], dim=-1
        ) + task_physiology
        base_future_by_group = {}
        offset = 0
        for group, size in self.physiology_group_sizes.items():
            current = output["current_physiology"][:, offset : offset + size]
            base_future_by_group[group] = current.unsqueeze(1) + self.future_physiology_delta_heads[
                group
            ](base_predicted_states)
            offset += size
        base_future_physiology = torch.cat(
            [base_future_by_group[group] for group in self.physiology_group_sizes], dim=-1
        ) + base_task_physiology
        future_physiology = (
            output["future_physiology"]
            + belief_future_physiology
            - base_future_physiology
        )

        age_epochs, _, _, _ = self._observation_metadata(
            history_signals, history_present
        )
        output["belief_trajectory"] = belief
        output["belief_prior_trajectory"] = prior
        output["belief_correction_gate"] = correction_gate
        output["belief_correction"] = correction
        output["belief_persistence_trajectory"] = persistence
        output["belief_endpoint_delta"] = endpoint_delta
        output["belief_base_corrected_history_state"] = output[
            "corrected_history_state"
        ]
        output["belief_base_predicted_states"] = base_predicted_states
        output["belief_base_stage_logits"] = output["stage_logits"]
        output["corrected_history_state"] = filtered_history_state
        output["predicted_states"] = predicted_states
        output["stage_logits"] = stage_logits
        output["future_physiology"] = future_physiology
        output["recursive_log_variance"] = self._partial_log_variance(
            filtered_history_state,
            age_epochs,
            output["observation_reliability"],
            selected_horizons,
        )
        return output
