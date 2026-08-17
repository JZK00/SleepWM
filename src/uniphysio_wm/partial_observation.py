from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


@dataclass(frozen=True)
class DynamicObservationSpec:
    name: str
    missing_epochs: Mapping[str, int]
    recovery_epochs: int = 0
    profile: str = "hard"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("dynamic observation condition requires a name")
        if self.recovery_epochs < 0:
            raise ValueError("recovery_epochs must be nonnegative")
        if self.profile not in {"hard", "linear_decay"}:
            raise ValueError(f"unsupported observation profile: {self.profile}")
        if not self.missing_epochs:
            raise ValueError("at least one modality interruption is required")
        if any(int(value) < 0 for value in self.missing_epochs.values()):
            raise ValueError("missing durations must be nonnegative")


def dynamic_observation_view(
    history_signals: torch.Tensor,
    history_present: torch.Tensor,
    modalities: Sequence[str],
    spec: DynamicObservationSpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a causal block interruption and optional pre-cutoff recovery."""

    if history_signals.ndim != 4:
        raise ValueError("history_signals must be [batch, epochs, modalities, samples]")
    if history_present.shape != history_signals.shape[:3]:
        raise ValueError("history_present must match the first three signal dimensions")
    if len(modalities) != history_signals.shape[2]:
        raise ValueError("modality names do not match the signal tensor")
    unknown = set(spec.missing_epochs) - set(modalities)
    if unknown:
        raise ValueError(f"unknown interrupted modalities: {sorted(unknown)}")

    epoch_count = history_signals.shape[1]
    stop = max(0, epoch_count - int(spec.recovery_epochs))
    quality = history_present.to(dtype=history_signals.dtype).clone()
    for modality, requested_duration in spec.missing_epochs.items():
        duration = min(int(requested_duration), stop)
        if duration == 0:
            continue
        modality_index = modalities.index(modality)
        start = stop - duration
        if spec.profile == "hard":
            quality[:, start:stop, modality_index] = 0.0
        else:
            decay = torch.linspace(
                1.0 - 1.0 / duration,
                0.0,
                duration,
                device=history_signals.device,
                dtype=history_signals.dtype,
            )
            quality[:, start:stop, modality_index] = decay.reshape(1, -1)

    observed = history_present.bool() & (quality > 0.0)
    signals = history_signals * quality.unsqueeze(-1)
    signals = signals.masked_fill(~observed.unsqueeze(-1), 0.0)
    return signals, observed, quality


def primary_dynamic_observation_specs(
    modalities: Sequence[str],
    duration_epochs: Sequence[int],
    recovery_epochs: Sequence[int],
) -> tuple[DynamicObservationSpec, ...]:
    specs = []
    for modality in modalities:
        for duration in duration_epochs:
            specs.append(
                DynamicObservationSpec(
                    name=f"tail_{modality.lower()}_{int(duration)}ep",
                    missing_epochs={modality: int(duration)},
                )
            )
    for duration in duration_epochs:
        specs.append(
            DynamicObservationSpec(
                name=f"tail_all_{int(duration)}ep",
                missing_epochs={modality: int(duration) for modality in modalities},
            )
        )

    recovery_duration = max(1, min(4, max(int(value) for value in duration_epochs)))
    for recovered in recovery_epochs:
        specs.append(
            DynamicObservationSpec(
                name=f"recover_all_{recovery_duration}ep_after_{int(recovered)}ep",
                missing_epochs={modality: recovery_duration for modality in modalities},
                recovery_epochs=int(recovered),
            )
        )
    specs.append(
        DynamicObservationSpec(
            name="linear_decay_all_4ep",
            missing_epochs={modality: 4 for modality in modalities},
            profile="linear_decay",
        )
    )
    asynchronous = {
        modality: max(1, 4 // (index + 1))
        for index, modality in enumerate(modalities)
    }
    specs.append(
        DynamicObservationSpec(
            name="asynchronous_tail",
            missing_epochs=asynchronous,
        )
    )
    return tuple(specs)

