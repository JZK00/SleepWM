from __future__ import annotations

from typing import Optional, Sequence
import torch
from sleepwm.masking import DynamicObservationSpec, dynamic_observation_view


def dynamic_view(
    signals: torch.Tensor,
    present: torch.Tensor,
    modalities: Sequence[str],
    spec: Optional[DynamicObservationSpec],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if spec is None:
        return signals, present, present.to(signals.dtype)
    return dynamic_observation_view(signals, present, modalities, spec)


def route_outputs(model, output: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    base_states = output["belief_base_predicted_states"]
    persistence_delta = (
        output["belief_persistence_trajectory"][:, -1]
        - output["history_epoch_states"][:, -1]
    )
    persistence_states = base_states + persistence_delta.unsqueeze(1)

    persistence_logits = output["belief_base_stage_logits"] + model.stage_head(
        persistence_states
    ) - model.stage_head(base_states)
    direct_current = output["current_stage_logits"]
    persistence_current = direct_current
    if model.current_stage_head is not None:
        base_history = output["belief_base_corrected_history_state"]
        persistence_history = base_history + persistence_delta
        current_delta = model.current_stage_head(
            persistence_history
        ) - model.current_stage_head(base_history)
        persistence_current = direct_current + current_delta
        persistence_logits = persistence_logits + current_delta.unsqueeze(1)

    persistence_task, persistence_physiology_task = model._task_residuals(
        persistence_states, output["observation_reliability"]
    )
    base_task, base_physiology_task = model._task_residuals(
        base_states, output["observation_reliability"]
    )
    persistence_logits = persistence_logits + persistence_task - base_task

    def physiology_from_states(
        states: torch.Tensor, task_residual: torch.Tensor
    ) -> torch.Tensor:
        values = []
        offset = 0
        for group, size in model.physiology_group_sizes.items():
            current = output["current_physiology"][:, offset : offset + size]
            values.append(
                current.unsqueeze(1)
                + model.future_physiology_delta_heads[group](states)
            )
            offset += size
        return torch.cat(values, dim=-1) + task_residual

    base_physiology = physiology_from_states(base_states, base_physiology_task)
    belief_physiology_task = model._task_residuals(
        output["predicted_states"], output["observation_reliability"]
    )[1]
    belief_physiology = physiology_from_states(
        output["predicted_states"], belief_physiology_task
    )
    direct_physiology = (
        output["future_physiology"] - belief_physiology + base_physiology
    )
    persistence_physiology = direct_physiology + physiology_from_states(
        persistence_states, persistence_physiology_task
    ) - base_physiology

    belief_current = output.get("belief_current_stage_logits", direct_current)
    return {
        "direct_incomplete": {
            "states": base_states,
            "stage_logits": output["belief_base_stage_logits"],
            "current_stage_logits": direct_current,
            "future_physiology": direct_physiology,
        },
        "static_persistence": {
            "states": persistence_states,
            "stage_logits": persistence_logits,
            "current_stage_logits": persistence_current,
            "future_physiology": persistence_physiology,
        },
        "recursive_belief": {
            "states": output["predicted_states"],
            "stage_logits": output["stage_logits"],
            "current_stage_logits": belief_current,
            "future_physiology": output["future_physiology"],
        },
    }


import numpy as np
import torch


def event_at(event: str, previous: int, current: int) -> bool:
    if previous < 0 or current < 0:
        return False
    if event == "transition":
        return previous != current
    if event == "sleep_onset":
        return previous == 0 and current != 0
    if event == "wake_onset":
        return previous != 0 and current == 0
    if event == "rem_onset":
        return previous != 4 and current == 4
    if event == "n3_onset":
        return previous != 3 and current == 3
    raise ValueError(f"unknown event: {event}")


def next_event_offset(labels: np.ndarray, current_index: int, event: str, max_horizon: int) -> int:
    end = min(len(labels) - 1, current_index + max_horizon)
    for index in range(current_index + 1, end + 1):
        if event_at(event, int(labels[index - 1]), int(labels[index])):
            return index - current_index
    return max_horizon + 1


def event_risk_scores(current_probabilities: torch.Tensor, future_probabilities: torch.Tensor) -> dict[str, torch.Tensor]:
    overlap = (current_probabilities.unsqueeze(1) * future_probabilities).sum(dim=-1)
    raw = {
        "transition": 1.0 - overlap,
        "sleep_onset": current_probabilities[:, 0:1] * (1.0 - future_probabilities[:, :, 0]),
        "wake_onset": (1.0 - current_probabilities[:, 0:1]) * future_probabilities[:, :, 0],
        "rem_onset": (1.0 - current_probabilities[:, 4:5]) * future_probabilities[:, :, 4],
        "n3_onset": (1.0 - current_probabilities[:, 3:4]) * future_probabilities[:, :, 3],
    }
    return {name: torch.cummax(value.clamp(0.0, 1.0), dim=1).values for name, value in raw.items()}


def average_precision(target: np.ndarray, score: np.ndarray) -> float | None:
    positive = int(target.sum())
    if positive == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    sorted_target = target[order].astype(np.float64)
    precision = np.cumsum(sorted_target) / np.arange(1, target.size + 1)
    return float(precision[sorted_target.astype(bool)].sum() / positive)


def load_label_map(dataset) -> dict[str, np.ndarray]:
    result = {}
    for record in dataset.records:
        with np.load(record["path"], allow_pickle=False) as archive:
            result[str(record["record_id"])] = archive["labels"].astype(np.int64, copy=True)
    return result


import math
import torch
from sleepwm.engine import data_loader
from sleepwm.masking import nonempty_modality_subsets
from sleepwm.masking import forced_history_view


def _maximum_abs_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return float("inf")
    return float((left - right).abs().max().detach().cpu())


def state_identity_check(
    model,
    recursive_model,
    dataset,
    config: dict,
    device: torch.device,
) -> dict:
    modalities = tuple(config["data"]["modalities"])
    horizons = tuple(int(value) for value in config["data"]["future_horizons"])
    batch = next(iter(data_loader(dataset, config, shuffle=False)))
    history = batch["history_signals"].to(device, torch.float32)
    present = batch["history_present"].to(device, torch.bool)
    compared_keys = (
        "predicted_states",
        "stage_logits",
        "future_physiology",
        "corrected_history_state",
        "observation_reliability",
        "recursive_update_gate",
        "recursive_log_variance",
        "physiology_dynamics_states",
    )
    by_subset = {}
    model.eval()
    recursive_model.eval()
    with torch.no_grad():
        for subset in nonempty_modality_subsets(modalities):
            forced, forced_present = forced_history_view(
                history, present, subset, modalities
            )
            integrated = model.rollout_context_horizons(
                forced, forced_present, horizons
            )
            reference = recursive_model.rollout_context_horizons(
                forced, forced_present, horizons
            )
            by_subset["+".join(subset)] = {
                key: _maximum_abs_difference(integrated[key], reference[key])
                for key in compared_keys
            }
    maximum = max(
        value
        for subset_values in by_subset.values()
        for value in subset_values.values()
    )
    tolerance = float(config["mainline"].get("state_identity_tolerance", 0.0))
    return {
        "maximum_absolute_difference": maximum,
        "tolerance": tolerance,
        "by_subset": by_subset,
        "passed": maximum <= tolerance,
    }


def extract_subset_waveform_cache(
    model,
    dataset,
    config: dict,
    device: torch.device,
    subset: tuple[str, ...],
) -> dict:
    modalities = tuple(config["data"]["modalities"])
    maximum_samples = model.waveform_decoder.max_samples
    cached = {
        "shared_state": [],
        "dynamics_state": [],
        "recent_waveform": [],
        "target_waveform": [],
        "valid": [],
        "modality_availability": [],
    }
    model.eval()
    with torch.no_grad():
        for batch in data_loader(dataset, config, shuffle=False):
            history = batch["history_signals"].to(device, torch.float32)
            present = batch["history_present"].to(device, torch.bool)
            forced, forced_present = forced_history_view(
                history, present, subset, modalities
            )
            output = model.rollout_context(forced, forced_present)
            cached["shared_state"].append(output["predicted_states"][:, 0].cpu())
            cached["dynamics_state"].append(
                output["physiology_dynamics_states"][:, 0].cpu()
            )
            cached["recent_waveform"].append(
                forced[:, -1, :, -maximum_samples:].half().cpu()
            )
            cached["target_waveform"].append(
                batch["future_signals"][:, 0, :, :maximum_samples].half()
            )
            cached["valid"].append(batch["future_present"][:, 0].bool())
            cached["modality_availability"].append(
                forced_present.to(torch.float32).mean(dim=1).cpu()
            )
    return {key: torch.cat(values) for key, values in cached.items()}


def _all_finite(value, key: str = "") -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item, name) for name, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (int, float)):
        if key == "baseline_rr_mae_ms":
            return True
        return math.isfinite(float(value))
    return True


def integration_gate(
    component_identity: dict,
    state_identity: dict,
    waveform_results: dict,
    w5_validation: dict,
    config: dict,
    smoke: bool,
) -> dict:
    modalities = tuple(config["data"]["modalities"])
    full_name = "+".join(modalities)
    full_mae = float(
        waveform_results[full_name]["model_waveform"]["all"][
            "mean_standardized_mae"
        ]
    )
    w5_mae = float(
        w5_validation["model_waveform"]["all"]["mean_standardized_mae"]
    )
    amplitude_ratios = {
        subset: {
            modality: float(values["generated_to_target_ratio"])
            for modality, values in metrics["amplitude"].items()
        }
        for subset, metrics in waveform_results.items()
    }
    minimum = float(config["mainline"].get("minimum_amplitude_ratio", 0.25))
    maximum = float(config["mainline"].get("maximum_amplitude_ratio", 2.5))
    amplitude_retained = all(
        minimum <= ratio <= maximum
        for values in amplitude_ratios.values()
        for ratio in values.values()
    )
    waveform_ratio = full_mae / max(w5_mae, 1e-12)
    maximum_waveform_ratio = float(
        config["mainline"].get("maximum_w5_waveform_ratio", 1.05)
    )
    result = {
        "scientific_evaluation": not smoke,
        "component_identity": bool(component_identity["passed"]),
        "r3_state_identity": bool(state_identity["passed"]),
        "all_subset_metrics_finite": _all_finite(waveform_results),
        "all_subset_amplitudes_retained": amplitude_retained,
        "amplitude_range": [minimum, maximum],
        "amplitude_ratios": amplitude_ratios,
        "full_waveform_mae": full_mae,
        "w5_waveform_mae": w5_mae,
        "full_waveform_ratio_to_w5": waveform_ratio,
        "maximum_w5_waveform_ratio": maximum_waveform_ratio,
        "full_waveform_retained": waveform_ratio <= maximum_waveform_ratio,
    }
    result["passed"] = bool(
        not smoke
        and result["component_identity"]
        and result["r3_state_identity"]
        and result["all_subset_metrics_finite"]
        and result["all_subset_amplitudes_retained"]
        and result["full_waveform_retained"]
    )
    return result


import torch
from sleepwm.engine import data_loader
from sleepwm.metrics import classification_metrics, forecast_subgroup_metrics
from sleepwm.masking import forced_history_view
from sleepwm.metrics import standardized_physiology_metrics
from sleepwm.rollout import recursive_latent_metrics, state_alignment_metrics
from sleepwm.metrics import waveform_forecast_metrics


def probability_uncertainty(
    probabilities: dict, horizons: tuple[int, ...], sample_rate: int, patch_samples: int
) -> dict:
    result = {}
    for modality, values in probabilities.items():
        if modality == "EEG":
            scale = values["spectral_scale"]
        elif modality == "ECG":
            scale = values["rr_scale_seconds"]
        elif modality == "EMG":
            scale = 0.5 * (values["envelope_scale"] + values["rms_scale"])
        else:
            continue
        by_horizon = {}
        for seconds in horizons:
            patch_count = int(seconds) * int(sample_rate) // int(patch_samples)
            by_horizon[str(seconds)] = float(scale[:, :patch_count].mean().detach().cpu())
        result[modality] = {
            "mean_scale": float(scale.mean().detach().cpu()),
            "by_horizon_seconds": by_horizon,
        }
    return result


def evaluate_subset(model, dataset, config: dict, device: torch.device, subset: tuple[str, ...]) -> dict:
    modalities = tuple(config["data"]["modalities"])
    horizons = tuple(int(value) for value in config["data"]["future_horizons"])
    waveform_horizons = tuple(
        int(value) for value in config["waveform"]["horizons_seconds"]
    )
    stage_logits = []
    future_labels = []
    current_labels = []
    physiology_prediction = []
    physiology_target = []
    physiology_valid = []
    waveform_prediction = []
    waveform_target = []
    waveform_valid = []
    recursive_prediction = []
    recursive_direct_prediction = []
    recursive_target = []
    recursive_correction = []
    recursive_direct_stage = []
    recursive_direct_physiology = []
    recursive_direct_waveform = []
    observation_direct_state = []
    observation_corrected_state = []
    observation_teacher_state = []
    observation_reliability = []
    uncertainty_sums = {
        modality: {"mean_scale": 0.0, "samples": 0, "by_horizon_seconds": {}}
        for modality in modalities
    }
    model.eval()
    with torch.no_grad():
        for batch in data_loader(dataset, config, shuffle=False):
            history_signals = batch["history_signals"].to(
                device=device, dtype=torch.float32
            )
            history_present = batch["history_present"].to(
                device=device, dtype=torch.bool
            )
            forced_signals, forced_present = forced_history_view(
                history_signals, history_present, subset, modalities
            )
            output = model.rollout(forced_signals, forced_present)
            batch_size = len(history_signals)
            stage_logits.append(output["stage_logits"].cpu())
            future_labels.append(batch["future_labels"])
            current_labels.append(batch["history_labels"][:, -1])
            physiology_prediction.append(output["future_physiology"].cpu())
            physiology_target.append(batch["future_physiology"])
            physiology_valid.append(batch["future_physiology_valid"])
            waveform_prediction.append(output["future_waveforms"].half().cpu())
            waveform_target.append(
                batch["future_signals"][:, 0, :, : model.waveform_decoder.max_samples].half()
            )
            waveform_valid.append(batch["future_present"][:, 0].bool())
            if "direct_predicted_states" in output:
                if model.target_encoder is None:
                    raise ValueError("recursive evaluation requires a target encoder")
                future_signals = batch["future_signals"].to(
                    device=device, dtype=torch.float32
                )
                future_present = batch["future_present"].to(
                    device=device, dtype=torch.bool
                )
                target_states = model._encode_epochs(
                    future_signals,
                    future_present,
                    require_gradient=False,
                    encoder=model.target_encoder,
                )
                direct_decoded = model.waveform_decoder(
                    forced_signals[:, -1],
                    output["direct_predicted_states"][:, 0],
                    output["physiology_dynamics_states"][:, 0],
                    return_structure=model.waveform_decoder.structured_event_heads,
                    return_probabilities=model.waveform_decoder.probabilistic_event_heads,
                )
                direct_waveform = (
                    direct_decoded[0]
                    if isinstance(direct_decoded, tuple)
                    else direct_decoded
                )
                recursive_prediction.append(output["predicted_states"].cpu())
                recursive_direct_prediction.append(
                    output["direct_predicted_states"].cpu()
                )
                recursive_target.append(target_states.cpu())
                correction = output.get("recursive_state_correction")
                if correction is None:
                    correction = output["observation_state_correction"].unsqueeze(1)
                    correction = correction.expand_as(output["predicted_states"])
                recursive_correction.append(correction.cpu())
                recursive_direct_stage.append(output["direct_stage_logits"].cpu())
                recursive_direct_physiology.append(
                    output["direct_future_physiology"].cpu()
                )
                recursive_direct_waveform.append(direct_waveform.half().cpu())
                if "corrected_history_state" in output:
                    teacher_output = model.direct_rollout_context(
                        history_signals, history_present
                    )
                    observation_direct_state.append(
                        output["direct_history_state"].cpu()
                    )
                    observation_corrected_state.append(
                        output["corrected_history_state"].cpu()
                    )
                    observation_teacher_state.append(
                        teacher_output["history_state"].cpu()
                    )
                    observation_reliability.append(
                        output["observation_reliability"].cpu()
                    )
            uncertainty = probability_uncertainty(
                output["future_waveform_probabilities"],
                waveform_horizons,
                int(config["data"]["sample_rate"]),
                int(config["waveform"]["patch_samples"]),
            )
            for modality, values in uncertainty.items():
                uncertainty_sums[modality]["mean_scale"] += (
                    float(values["mean_scale"]) * batch_size
                )
                uncertainty_sums[modality]["samples"] += batch_size
                for horizon, value in values["by_horizon_seconds"].items():
                    uncertainty_sums[modality]["by_horizon_seconds"][horizon] = (
                        uncertainty_sums[modality]["by_horizon_seconds"].get(horizon, 0.0)
                        + float(value) * batch_size
                    )
    logits = torch.cat(stage_logits)
    labels = torch.cat(future_labels)
    current = torch.cat(current_labels)
    num_classes = int(config["data"].get("num_classes", 5))
    uncertainty_metrics = {}
    for modality, values in uncertainty_sums.items():
        count = max(int(values["samples"]), 1)
        uncertainty_metrics[modality] = {
            "mean_scale": float(values["mean_scale"]) / count,
            "by_horizon_seconds": {
                horizon: float(value) / count
                for horizon, value in values["by_horizon_seconds"].items()
            },
        }
    feature_names = tuple(config["physiology"]["feature_names"])
    feature_groups = {
        group: tuple(names)
        for group, names in config["physiology"]["feature_groups"].items()
    }
    result = {
        "stage": {
            "all_horizons": classification_metrics(logits, labels, num_classes),
            "by_horizon": {
                str(horizon): classification_metrics(
                    logits[:, index], labels[:, index], num_classes
                )
                for index, horizon in enumerate(horizons)
            },
            "subgroups": forecast_subgroup_metrics(
                logits, labels, current, horizons, num_classes
            ),
        },
        "future_physiology": standardized_physiology_metrics(
            torch.cat(physiology_prediction),
            torch.cat(physiology_target),
            torch.cat(physiology_valid),
            feature_names,
            feature_groups,
            horizons,
        ),
        "waveform": waveform_forecast_metrics(
            torch.cat(waveform_prediction),
            torch.cat(waveform_target),
            torch.cat(waveform_valid),
            modalities,
            int(config["data"]["sample_rate"]),
            waveform_horizons,
        ),
        "uncertainty": uncertainty_metrics,
    }
    if recursive_prediction:
        direct_logits = torch.cat(recursive_direct_stage)
        direct_physiology = torch.cat(recursive_direct_physiology)
        direct_waveform = torch.cat(recursive_direct_waveform)
        target_physiology = torch.cat(physiology_target)
        target_valid = torch.cat(physiology_valid)
        target_waveform = torch.cat(waveform_target)
        target_waveform_valid = torch.cat(waveform_valid)
        result["recursive_comparison"] = {
            "latent": recursive_latent_metrics(
                torch.cat(recursive_prediction),
                torch.cat(recursive_direct_prediction),
                torch.cat(recursive_target),
                torch.cat(recursive_correction),
                horizons,
            ),
            "direct_stage": {
                "all_horizons": classification_metrics(
                    direct_logits, labels, num_classes
                ),
                "by_horizon": {
                    str(horizon): classification_metrics(
                        direct_logits[:, index], labels[:, index], num_classes
                    )
                    for index, horizon in enumerate(horizons)
                },
            },
            "direct_future_physiology": standardized_physiology_metrics(
                direct_physiology,
                target_physiology,
                target_valid,
                feature_names,
                feature_groups,
                horizons,
            ),
            "direct_waveform": waveform_forecast_metrics(
                direct_waveform,
                target_waveform,
                target_waveform_valid,
                modalities,
                int(config["data"]["sample_rate"]),
                waveform_horizons,
            ),
        }
        if observation_direct_state:
            reliability = torch.cat(observation_reliability)
            result["recursive_comparison"]["observation_state"] = (
                state_alignment_metrics(
                    torch.cat(observation_direct_state),
                    torch.cat(observation_corrected_state),
                    torch.cat(observation_teacher_state),
                )
            )
            result["recursive_comparison"]["mean_reliability"] = {
                modality: float(reliability[:, index].mean())
                for index, modality in enumerate(modalities)
            }
    return result
