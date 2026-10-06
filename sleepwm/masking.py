from __future__ import annotations

from itertools import combinations
from typing import List, Optional, Tuple

import torch


def random_span_mask(
    batch_size: int,
    num_tokens: int,
    mask_ratio: float,
    min_span: int = 1,
    max_span: int = 8,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if not 0.0 < mask_ratio <= 1.0:
        raise ValueError("mask_ratio must be in (0, 1]")
    if num_tokens < 1 or min_span < 1 or max_span < min_span:
        raise ValueError("invalid token or span configuration")
    target = max(1, int(round(num_tokens * mask_ratio)))
    mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device)
    for batch_index in range(batch_size):
        attempts = 0
        while int(mask[batch_index].sum()) < target and attempts < num_tokens * 8:
            attempts += 1
            span = int(torch.randint(min_span, max_span + 1, (1,), device=device, generator=generator).item())
            span = min(span, num_tokens)
            start = int(
                torch.randint(0, num_tokens - span + 1, (1,), device=device, generator=generator).item()
            )
            mask[batch_index, start : start + span] = True
    return mask


def random_modality_presence(
    batch_size: int,
    modality_count: int,
    drop_probability: float,
    device: torch.device,
    protected_modality: int | None = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if not 0.0 <= drop_probability < 1.0:
        raise ValueError("drop_probability must be in [0, 1)")
    present = torch.rand(batch_size, modality_count, device=device, generator=generator) >= drop_probability
    if protected_modality is not None:
        present[:, protected_modality] = True
    empty = ~present.any(dim=1)
    if empty.any():
        fallback = torch.randint(
            0,
            modality_count,
            (int(empty.sum()),),
            device=device,
            generator=generator,
        )
        present[empty] = False
        present[empty, fallback] = True
    return present


def random_natural_modality_subset(
    natural_present: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample uniformly from each row's nonempty naturally available subsets."""

    if natural_present.ndim != 2:
        raise ValueError("natural_present must be [batch, modalities]")
    natural_present = natural_present.to(dtype=torch.bool)
    if not natural_present.any(dim=1).all():
        raise ValueError("every sample must contain at least one natural modality")
    batch_size, modality_count = natural_present.shape
    selected = torch.zeros_like(natural_present)
    pending = torch.arange(batch_size, device=natural_present.device)
    bit_positions = torch.arange(modality_count, device=natural_present.device)
    while len(pending):
        codes = torch.randint(
            1,
            2**modality_count,
            (len(pending), 1),
            device=natural_present.device,
            generator=generator,
        )
        candidates = codes.bitwise_right_shift(bit_positions).bitwise_and(1).bool()
        candidates = candidates & natural_present[pending]
        accepted = candidates.any(dim=1)
        selected[pending[accepted]] = candidates[accepted]
        pending = pending[~accepted]
    return selected


def random_strict_natural_modality_subset(
    natural_present: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample a proper nonempty subset whenever at least two modalities exist."""

    if natural_present.ndim != 2:
        raise ValueError("natural_present must be [batch, modalities]")
    natural_present = natural_present.to(dtype=torch.bool)
    if not natural_present.any(dim=1).all():
        raise ValueError("every sample must contain at least one natural modality")
    batch_size, modality_count = natural_present.shape
    selected = natural_present.clone()
    pending = torch.nonzero(natural_present.sum(dim=1) > 1, as_tuple=False).flatten()
    bit_positions = torch.arange(modality_count, device=natural_present.device)
    while len(pending):
        codes = torch.randint(
            1,
            2**modality_count,
            (len(pending), 1),
            device=natural_present.device,
            generator=generator,
        )
        candidates = codes.bitwise_right_shift(bit_positions).bitwise_and(1).bool()
        candidates = candidates & natural_present[pending]
        accepted = candidates.any(dim=1) & (candidates != natural_present[pending]).any(dim=1)
        selected[pending[accepted]] = candidates[accepted]
        pending = pending[~accepted]
    return selected


def random_full_biased_natural_modality_subset(
    natural_present: torch.Tensor,
    full_modality_probability: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Choose a full-observation batch or a batch of strict nonempty subsets."""

    if not 0.0 <= full_modality_probability <= 1.0:
        raise ValueError("full_modality_probability must be in [0, 1]")
    keep_full = bool(
        torch.rand((), device=natural_present.device, generator=generator).item()
        < full_modality_probability
    )
    if keep_full:
        return natural_present.to(dtype=torch.bool).clone()
    return random_strict_natural_modality_subset(natural_present, generator=generator)


def nonempty_modality_subsets(modalities: Tuple[str, ...]) -> List[Tuple[str, ...]]:
    return [subset for size in range(1, len(modalities) + 1) for subset in combinations(modalities, size)]


from typing import Dict, Sequence

import torch



def forced_history_view(
    history_signals: torch.Tensor,
    history_present: torch.Tensor,
    subset: Sequence[str],
    modalities: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    if history_signals.ndim != 4 or history_present.shape != history_signals.shape[:3]:
        raise ValueError("history tensors must be [batch, epochs, modalities, samples]")
    selected = torch.tensor(
        [modality in subset for modality in modalities],
        dtype=torch.bool,
        device=history_present.device,
    )
    if not selected.any():
        raise ValueError("forced modality subset must be nonempty")
    forced_present = history_present.bool() & selected.reshape(1, 1, -1)
    forced_signals = history_signals.masked_fill(
        ~forced_present.unsqueeze(-1), 0.0
    )
    return forced_signals, forced_present


def sampled_history_view(
    history_signals: torch.Tensor,
    history_present: torch.Tensor,
    full_modality_probability: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if history_signals.ndim != 4 or history_present.shape != history_signals.shape[:3]:
        raise ValueError("history tensors must be [batch, epochs, modalities, samples]")
    naturally_stable = history_present.bool().all(dim=1)
    selected = random_full_biased_natural_modality_subset(
        naturally_stable,
        full_modality_probability=float(full_modality_probability),
    )
    forced_present = history_present.bool() & selected.unsqueeze(1)
    forced_signals = history_signals.masked_fill(
        ~forced_present.unsqueeze(-1), 0.0
    )
    return forced_signals, forced_present, selected


def missing_rollout_summary(
    results: Dict[str, Dict[str, object]], modalities: Sequence[str]
) -> Dict[str, object]:
    full_key = "+".join(modalities)
    if full_key not in results:
        raise ValueError("missing-rollout results require the full-modality reference")
    full = results[full_key]
    full_stage = float(full["stage"]["all_horizons"]["macro_f1"])
    full_physiology = float(
        full["future_physiology"]["all_features"]["mean_normalized_mae"]
    )
    full_waveform = float(full["waveform"]["all"]["mean_standardized_mae"])
    nonfull = [key for key in results if key != full_key]
    if not nonfull:
        raise ValueError("missing-rollout results require nonfull modality subsets")
    degradation = {}
    for key in nonfull:
        values = results[key]
        stage = float(values["stage"]["all_horizons"]["macro_f1"])
        physiology = float(
            values["future_physiology"]["all_features"]["mean_normalized_mae"]
        )
        waveform = float(values["waveform"]["all"]["mean_standardized_mae"])
        degradation[key] = {
            "stage_macro_f1_drop": full_stage - stage,
            "future_physiology_mae_ratio": physiology / max(full_physiology, 1e-12),
            "waveform_mae_ratio": waveform / max(full_waveform, 1e-12),
        }
    uncertainty_response = {}
    for modality in modalities:
        absent = [
            float(results[key]["uncertainty"][modality]["mean_scale"])
            for key in nonfull
            if modality not in key.split("+")
        ]
        full_uncertainty = float(full["uncertainty"][modality]["mean_scale"])
        absent_mean = sum(absent) / len(absent)
        uncertainty_response[modality] = {
            "full_mean_scale": full_uncertainty,
            "absent_mean_scale": absent_mean,
            "ratio": absent_mean / max(full_uncertainty, 1e-12),
            "increased": absent_mean > full_uncertainty,
        }
    stage_drops = [values["stage_macro_f1_drop"] for values in degradation.values()]
    physiology_ratios = [
        values["future_physiology_mae_ratio"] for values in degradation.values()
    ]
    waveform_ratios = [values["waveform_mae_ratio"] for values in degradation.values()]
    return {
        "full_modality": {
            "stage_macro_f1": full_stage,
            "future_physiology_mae": full_physiology,
            "waveform_mae": full_waveform,
        },
        "by_subset_degradation": degradation,
        "nonfull_mean_stage_macro_f1_drop": sum(stage_drops) / len(stage_drops),
        "nonfull_worst_stage_macro_f1_drop": max(stage_drops),
        "nonfull_mean_future_physiology_mae_ratio": sum(physiology_ratios)
        / len(physiology_ratios),
        "nonfull_mean_waveform_mae_ratio": sum(waveform_ratios) / len(waveform_ratios),
        "uncertainty_response": uncertainty_response,
    }


def missing_rollout_gate_result(
    summary: Dict[str, object],
    mean_stage_drop_limit: float = 0.10,
    worst_stage_drop_limit: float = 0.20,
    physiology_mae_ratio_limit: float = 1.25,
    waveform_mae_ratio_limit: float = 1.25,
    uncertainty_modalities_required: int = 2,
) -> Dict[str, object]:
    uncertainty_increased = [
        modality
        for modality, values in summary["uncertainty_response"].items()
        if bool(values["increased"])
    ]
    result = {
        "mean_stage_retained": float(summary["nonfull_mean_stage_macro_f1_drop"])
        <= float(mean_stage_drop_limit),
        "worst_stage_retained": float(summary["nonfull_worst_stage_macro_f1_drop"])
        <= float(worst_stage_drop_limit),
        "future_physiology_retained": float(
            summary["nonfull_mean_future_physiology_mae_ratio"]
        )
        <= float(physiology_mae_ratio_limit),
        "waveform_retained": float(summary["nonfull_mean_waveform_mae_ratio"])
        <= float(waveform_mae_ratio_limit),
        "uncertainty_increased_modalities": uncertainty_increased,
        "uncertainty_response_passed": len(uncertainty_increased)
        >= int(uncertainty_modalities_required),
    }
    result["passed"] = bool(
        result["mean_stage_retained"]
        and result["worst_stage_retained"]
        and result["future_physiology_retained"]
        and result["waveform_retained"]
        and result["uncertainty_response_passed"]
    )
    return result


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
