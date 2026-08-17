from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Subset

from evaluate_sleep_events import (
    EVENTS,
    at_risk,
    evaluate_event_bundle,
    event_risk_scores,
    load_label_map,
    next_event_offset,
)
from train_recursive_belief_filter import build_student
from uniphysio_wm.engine import (
    data_loader,
    load_checkpoint,
    physio_feature_sequence_dataset,
    resolve_device,
    seed_everything,
)
from uniphysio_wm.metrics import classification_metrics
from uniphysio_wm.outcome_adapter import TrajectoryOutcomeAdapter
from uniphysio_wm.partial_observation import (
    DynamicObservationSpec,
    dynamic_observation_view,
)
from uniphysio_wm.physiology_metrics import standardized_physiology_metrics
from uniphysio_wm.waveform_metrics import (
    structured_head_metrics,
    waveform_forecast_metrics,
)


ROUTES = ("direct_incomplete", "static_persistence", "recursive_belief")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SleepWM belief trajectories on downstream outcomes."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--adapter-checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--unlock-test", action="store_true")
    parser.add_argument("--event-thresholds")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--condition", action="append")
    return parser.parse_args()


def protocol_specs(modalities: Sequence[str]) -> tuple[Optional[DynamicObservationSpec], ...]:
    all_for = lambda duration: {modality: duration for modality in modalities}
    return (
        None,
        DynamicObservationSpec("hard_eeg_1ep", {"EEG": 1}),
        DynamicObservationSpec("hard_eeg_4ep", {"EEG": 4}),
        DynamicObservationSpec("hard_eeg_10ep", {"EEG": 10}),
        DynamicObservationSpec("hard_all_1ep", all_for(1)),
        DynamicObservationSpec("hard_all_4ep", all_for(4)),
        DynamicObservationSpec("hard_all_10ep", all_for(10)),
        DynamicObservationSpec(
            "linear_decay_all_4ep", all_for(4), profile="linear_decay"
        ),
        DynamicObservationSpec(
            "asynchronous_eeg4_ecg2_emg1",
            {"EEG": 4, "ECG": 2, "EMG": 1},
        ),
        DynamicObservationSpec(
            "recover_all_4ep_after_1ep", all_for(4), recovery_epochs=1
        ),
        DynamicObservationSpec(
            "recover_all_4ep_after_2ep", all_for(4), recovery_epochs=2
        ),
    )


def condition_name(spec: Optional[DynamicObservationSpec]) -> str:
    return "full_observation" if spec is None else spec.name


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


def decode_route(
    model,
    signals: torch.Tensor,
    present: torch.Tensor,
    output: dict[str, torch.Tensor],
    route: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, Optional[dict[str, dict[str, torch.Tensor]]]]:
    availability = present.to(signals.dtype).mean(dim=1)
    decoded = model.waveform_decoder(
        signals[:, -1],
        route["states"][:, 0],
        output["physiology_dynamics_states"][:, 0],
        return_structure=model.waveform_decoder.structured_event_heads,
        return_probabilities=model.waveform_decoder.probabilistic_event_heads,
        modality_availability=availability,
    )
    if model.waveform_decoder.probabilistic_event_heads:
        waveform, structure, _ = decoded
        return waveform, structure
    if model.waveform_decoder.structured_event_heads:
        waveform, structure = decoded
        return waveform, structure
    return decoded, None


def empty_route() -> dict[str, Any]:
    return {
        "stage_logits": [],
        "future_physiology": [],
        "future_waveforms": [],
        "structures": {},
        "event_scores": {event: [] for event in EVENTS},
    }


def append_structures(
    target: dict[str, dict[str, list[torch.Tensor]]],
    structures: Optional[dict[str, dict[str, torch.Tensor]]],
) -> None:
    if structures is None:
        return
    for modality, values in structures.items():
        target.setdefault(modality, {})
        for name, value in values.items():
            target[modality].setdefault(name, []).append(value.half().cpu())


def stack_structures(
    values: dict[str, dict[str, list[torch.Tensor]]]
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        modality: {name: torch.cat(parts).float() for name, parts in fields.items()}
        for modality, fields in values.items()
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x = x.astype(np.float64) - float(x.mean())
    y = y.astype(np.float64) - float(y.mean())
    denominator = float(np.sqrt(np.square(x).sum() * np.square(y).sum()))
    return float(np.dot(x, y) / denominator) if denominator > 1e-12 else float("nan")


def subject_calibration_mask(subjects: np.ndarray) -> np.ndarray:
    selected = []
    for subject in subjects:
        digest = hashlib.sha256(str(subject).encode("utf-8")).digest()
        selected.append(int.from_bytes(digest[:4], "little") % 5 < 2)
    mask = np.asarray(selected, dtype=bool)
    if not mask.any() or mask.all():
        unique = sorted(set(str(value) for value in subjects))
        calibration = set(unique[: max(1, len(unique) // 3)])
        mask = np.asarray([str(value) in calibration for value in subjects], dtype=bool)
    return mask


def select_f1_threshold(target: np.ndarray, score: np.ndarray) -> float:
    """Select the exact best observed threshold in O(n log n)."""

    if target.size == 0 or np.unique(target).size < 2:
        return 0.5
    target = target.astype(bool, copy=False)
    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_target = target[order]
    true_positive = np.cumsum(sorted_target, dtype=np.int64)
    predicted_positive = np.arange(1, len(score) + 1, dtype=np.int64)
    false_positive = predicted_positive - true_positive
    false_negative = int(sorted_target.sum()) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = np.divide(
        2.0 * true_positive,
        np.maximum(denominator, 1),
        dtype=np.float64,
    )
    group_end = np.r_[sorted_score[1:] != sorted_score[:-1], True]
    candidate_indices = np.flatnonzero(group_end)
    best = candidate_indices[int(np.argmax(f1[candidate_indices]))]
    return float(sorted_score[best])


def event_metadata(
    dataset,
    record_ids: np.ndarray,
    origins: np.ndarray,
    current_labels: np.ndarray,
    horizons: Sequence[int],
) -> dict[str, Any]:
    labels = load_label_map(dataset)
    maximum = max(horizons)
    return {
        "tte": {
            event: np.asarray(
                [
                    next_event_offset(labels[record], int(origin), event, maximum)
                    for record, origin in zip(record_ids, origins)
                ],
                dtype=np.int64,
            )
            for event in EVENTS
        },
        "at_risk": {
            event: np.asarray(
                [at_risk(event, int(label)) for label in current_labels], dtype=bool
            )
            for event in EVENTS
        },
    }


def subset_bundle(bundle: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    return {
        "record_ids": bundle["record_ids"][mask],
        "origins": bundle["origins"][mask],
        "scores": {event: values[mask] for event, values in bundle["scores"].items()},
        "tte": {event: values[mask] for event, values in bundle["tte"].items()},
        "at_risk": {
            event: values[mask] for event, values in bundle["at_risk"].items()
        },
    }


def route_event_metrics(
    route: dict[str, Any],
    metadata: dict[str, Any],
    record_ids: np.ndarray,
    origins: np.ndarray,
    subjects: np.ndarray,
    horizons: Sequence[int],
    epoch_seconds: int,
    fixed_thresholds: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    bundle = {
        "record_ids": record_ids,
        "origins": origins,
        "scores": {
            event: torch.cat(values).numpy()
            for event, values in route["event_scores"].items()
        },
        **metadata,
    }
    if fixed_thresholds is not None:
        missing = sorted(set(EVENTS) - set(fixed_thresholds))
        if missing:
            raise ValueError(f"fixed event thresholds are incomplete: {missing}")
        thresholds = {event: float(fixed_thresholds[event]) for event in EVENTS}
        return {
            "threshold_source": "frozen_validation_calibration",
            "calibration_subjects": 0,
            "evaluation_subjects": len(set(subjects.tolist())),
            "thresholds": thresholds,
            "evaluation": evaluate_event_bundle(
                bundle, horizons, thresholds, epoch_seconds
            ),
        }

    calibration_mask = subject_calibration_mask(subjects)
    calibration = subset_bundle(bundle, calibration_mask)
    evaluation = subset_bundle(bundle, ~calibration_mask)
    thresholds = {}
    for event in EVENTS:
        risk = calibration["at_risk"][event]
        target = calibration["tte"][event][risk] <= max(horizons)
        thresholds[event] = select_f1_threshold(
            target,
            calibration["scores"][event][risk, -1],
        )
    return {
        "threshold_source": "validation_subject_internal_calibration",
        "calibration_subjects": len(set(subjects[calibration_mask].tolist())),
        "evaluation_subjects": len(set(subjects[~calibration_mask].tolist())),
        "thresholds": thresholds,
        "evaluation": evaluate_event_bundle(
            evaluation, horizons, thresholds, epoch_seconds
        ),
    }


def evaluate_condition(
    model,
    adapter: Optional[TrajectoryOutcomeAdapter],
    dataset,
    config: dict,
    device: torch.device,
    spec: Optional[DynamicObservationSpec],
    event_thresholds: Optional[dict[str, dict[str, float]]] = None,
) -> dict[str, Any]:
    modalities = tuple(config["data"]["modalities"])
    horizons = tuple(int(value) for value in config["data"]["future_horizons"])
    waveform_horizons = tuple(
        int(value) for value in config["waveform"]["horizons_seconds"]
    )
    routes = {name: empty_route() for name in ROUTES}
    labels = []
    physiology_target = []
    physiology_valid = []
    waveform_target = []
    waveform_valid = []
    subjects: list[str] = []
    record_ids: list[str] = []
    origins: list[int] = []
    current_labels = []
    uncertainty = []
    uncertainty_stage_error = []
    uncertainty_physiology_error = []
    quality = []
    availability = []
    reliability = []
    age = []

    model.eval()
    with torch.inference_mode():
        for batch_index, batch in enumerate(data_loader(dataset, config, shuffle=False)):
            natural_signals = batch["history_signals"].to(
                device=device, dtype=torch.float32
            )
            natural_present = batch["history_present"].to(
                device=device, dtype=torch.bool
            )
            signals, present, observation_quality = dynamic_view(
                natural_signals, natural_present, modalities, spec
            )
            output = model.rollout_context_horizons(signals, present, horizons)
            route_values = route_outputs(model, output)
            if adapter is not None:
                adapted = adapter(output)
                route_values["recursive_belief"]["stage_logits"] = adapted[
                    "stage_logits"
                ]
                route_values["recursive_belief"]["future_physiology"] = adapted[
                    "future_physiology"
                ]
            target_labels = batch["future_labels"][:, : len(horizons)]
            labels.append(target_labels.cpu())
            physiology_target.append(batch["future_physiology"][:, : len(horizons)])
            physiology_valid.append(batch["future_physiology_valid"][:, : len(horizons)])
            maximum_samples = model.waveform_decoder.max_samples
            waveform_target.append(
                batch["future_signals"][:, 0, :, :maximum_samples].half()
            )
            waveform_valid.append(batch["future_present"][:, 0].bool())

            for route_name, values in route_values.items():
                waveforms, structures = decode_route(
                    model, signals, present, output, values
                )
                target = routes[route_name]
                target["stage_logits"].append(values["stage_logits"].cpu())
                target["future_physiology"].append(
                    values["future_physiology"].cpu()
                )
                target["future_waveforms"].append(waveforms.half().cpu())
                append_structures(target["structures"], structures)
                risks = event_risk_scores(
                    values["current_stage_logits"].softmax(dim=-1),
                    values["stage_logits"].softmax(dim=-1),
                )
                for event in EVENTS:
                    target["event_scores"][event].append(risks[event].cpu())

            belief_probabilities = route_values["recursive_belief"][
                "stage_logits"
            ].softmax(dim=-1)
            label_device = target_labels.to(device)
            stage_error = 1.0 - belief_probabilities.gather(
                -1, label_device.unsqueeze(-1)
            ).squeeze(-1)
            phys_target_device = batch["future_physiology"][:, : len(horizons)].to(
                device=device, dtype=torch.float32
            )
            phys_valid_device = batch["future_physiology_valid"][:, : len(horizons)].to(
                device=device, dtype=torch.bool
            )
            phys_error = (
                (
                    route_values["recursive_belief"]["future_physiology"]
                    - phys_target_device
                )
                .abs()
                .masked_fill(~phys_valid_device, 0.0)
                .sum(dim=-1)
                / phys_valid_device.sum(dim=-1).clamp_min(1)
            )
            uncertainty.append(output["recursive_log_variance"].mul(0.5).exp().cpu())
            uncertainty_stage_error.append(stage_error.cpu())
            uncertainty_physiology_error.append(phys_error.cpu())
            quality.append(observation_quality.mean(dim=1).cpu())
            availability.append(present.float().mean(dim=1).cpu())
            reliability.append(output["observation_reliability"].cpu())
            age.append(output["observation_age_epochs"].cpu())
            subjects.extend(str(value) for value in batch["subject"])
            record_ids.extend(str(value) for value in batch["record_id"])
            origins.extend(
                (
                    batch["start_epoch"]
                    + int(config["data"]["history_epochs"])
                    - 1
                ).tolist()
            )
            current_labels.append(batch["history_labels"][:, -1].cpu())
            if (batch_index + 1) % 25 == 0:
                print(
                    f"condition={condition_name(spec)} batches={batch_index + 1}",
                    flush=True,
                )

    labels_tensor = torch.cat(labels)
    physiology_target_tensor = torch.cat(physiology_target)
    physiology_valid_tensor = torch.cat(physiology_valid)
    waveform_target_tensor = torch.cat(waveform_target).float()
    waveform_valid_tensor = torch.cat(waveform_valid)
    record_array = np.asarray(record_ids)
    origin_array = np.asarray(origins, dtype=np.int64)
    subject_array = np.asarray(subjects)
    current_array = torch.cat(current_labels).numpy()
    metadata = event_metadata(
        dataset.dataset if isinstance(dataset, Subset) else dataset,
        record_array,
        origin_array,
        current_array,
        horizons,
    )
    feature_names = tuple(config["physiology"]["feature_names"])
    feature_groups = {
        group: tuple(names)
        for group, names in config["physiology"]["feature_groups"].items()
    }
    route_results = {}
    for route_name, values in routes.items():
        logits = torch.cat(values["stage_logits"])
        waveforms = torch.cat(values["future_waveforms"]).float()
        result = {
            "stage": {
                "all_horizons": classification_metrics(
                    logits, labels_tensor, int(config["data"].get("num_classes", 5))
                ),
                "by_horizon": {
                    str(horizon): classification_metrics(
                        logits[:, index],
                        labels_tensor[:, index],
                        int(config["data"].get("num_classes", 5)),
                    )
                    for index, horizon in enumerate(horizons)
                },
            },
            "future_physiology": standardized_physiology_metrics(
                torch.cat(values["future_physiology"]),
                physiology_target_tensor,
                physiology_valid_tensor,
                feature_names,
                feature_groups,
                horizons,
            ),
            "waveform": waveform_forecast_metrics(
                waveforms,
                waveform_target_tensor,
                waveform_valid_tensor,
                modalities,
                int(config["data"]["sample_rate"]),
                waveform_horizons,
            ),
            "sleep_events": route_event_metrics(
                values,
                metadata,
                record_array,
                origin_array,
                subject_array,
                horizons,
                int(config["data"]["epoch_seconds"]),
                None if event_thresholds is None else event_thresholds[route_name],
            ),
        }
        if values["structures"]:
            result["waveform_structure"] = structured_head_metrics(
                stack_structures(values["structures"]),
                waveform_target_tensor,
                waveform_valid_tensor,
                int(config["data"]["sample_rate"]),
                waveform_horizons,
                int(config["waveform"]["patch_samples"]),
            )
        route_results[route_name] = result

    uncertainty_array = torch.cat(uncertainty).numpy().reshape(-1)
    stage_error_array = torch.cat(uncertainty_stage_error).numpy().reshape(-1)
    physiology_error_array = torch.cat(uncertainty_physiology_error).numpy().reshape(-1)
    uncertainty_result = {
        "mean_scale": float(uncertainty_array.mean()),
        "by_horizon": {
            str(horizon): float(torch.cat(uncertainty)[:, index].mean())
            for index, horizon in enumerate(horizons)
        },
        "stage_error": {
            "pearson_r": correlation(uncertainty_array, stage_error_array),
            "spearman_r": correlation(
                rankdata(uncertainty_array), rankdata(stage_error_array)
            ),
        },
        "physiology_error": {
            "pearson_r": correlation(uncertainty_array, physiology_error_array),
            "spearman_r": correlation(
                rankdata(uncertainty_array), rankdata(physiology_error_array)
            ),
        },
    }
    return {
        "observation": {
            "mean_quality": dict(zip(modalities, torch.cat(quality).mean(dim=0).tolist())),
            "mean_availability": dict(
                zip(modalities, torch.cat(availability).mean(dim=0).tolist())
            ),
            "mean_reliability": dict(
                zip(modalities, torch.cat(reliability).mean(dim=0).tolist())
            ),
            "mean_age_epochs": dict(zip(modalities, torch.cat(age).mean(dim=0).tolist())),
        },
        "uncertainty": uncertainty_result,
        "routes": route_results,
        "sequences": len(subjects),
        "subjects": len(set(subjects)),
    }


def primary_summary(conditions: dict[str, Any], primary_horizons: Sequence[int]) -> dict[str, Any]:
    summary = {}
    for name, condition in conditions.items():
        routes = condition["routes"]
        route_summary = {}
        for route_name, values in routes.items():
            f1 = np.mean(
                [
                    values["stage"]["by_horizon"][str(horizon)]["macro_f1"]
                    for horizon in primary_horizons
                ]
            )
            physiology = np.mean(
                [
                    values["future_physiology"]["by_horizon"][str(horizon)][
                        "mean_normalized_mae"
                    ]
                    for horizon in primary_horizons
                ]
            )
            transition = values["sleep_events"]["evaluation"]["transition"][
                "max_horizon"
            ]
            route_summary[route_name] = {
                "primary_stage_macro_f1": float(f1),
                "primary_physiology_mae": float(physiology),
                "transition_auroc": transition["auroc"],
                "transition_auprc": transition["auprc"],
                "waveform_mae": values["waveform"]["all"][
                    "mean_standardized_mae"
                ],
            }
        belief = route_summary["recursive_belief"]
        persistence = route_summary["static_persistence"]
        direct = route_summary["direct_incomplete"]
        route_summary["belief_gain"] = {
            "stage_f1_vs_direct": belief["primary_stage_macro_f1"]
            - direct["primary_stage_macro_f1"],
            "stage_f1_vs_persistence": belief["primary_stage_macro_f1"]
            - persistence["primary_stage_macro_f1"],
            "physiology_mae_reduction_vs_direct": direct["primary_physiology_mae"]
            - belief["primary_physiology_mae"],
            "physiology_mae_reduction_vs_persistence": persistence[
                "primary_physiology_mae"
            ]
            - belief["primary_physiology_mae"],
            "transition_auprc_gain_vs_direct": (
                None
                if belief["transition_auprc"] is None
                or direct["transition_auprc"] is None
                else belief["transition_auprc"] - direct["transition_auprc"]
            ),
            "transition_auprc_gain_vs_persistence": (
                None
                if belief["transition_auprc"] is None
                or persistence["transition_auprc"] is None
                else belief["transition_auprc"] - persistence["transition_auprc"]
            ),
        }
        summary[name] = route_summary
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    split = payload["protocol"]["split"]
    lines = [
        "# SleepWM trajectory-conditioned outcomes",
        "",
        f"Frozen {split} evaluation using a fixed SleepWM checkpoint.",
        "",
        "| Condition | Route | Stage Macro-F1 | Physiology MAE | Transition AUPRC | Waveform MAE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for condition, values in payload["summary"].items():
        for route in ROUTES:
            row = values[route]
            auprc = row["transition_auprc"]
            auprc_text = "NA" if auprc is None else f"{float(auprc):.4f}"
            lines.append(
                f"| {condition} | {route} | {row['primary_stage_macro_f1']:.4f} | "
                f"{row['primary_physiology_mae']:.4f} | "
                f"{auprc_text} | {row['waveform_mae']:.4f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.split == "test" and not args.unlock_test:
        raise PermissionError("test evaluation requires --unlock-test")
    if args.split == "test" and not args.event_thresholds:
        raise ValueError("test evaluation requires frozen validation event thresholds")
    with Path(args.config).open(encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    checkpoint_config = protocol.get("sleepwm", protocol.get("po3"))
    if not isinstance(checkpoint_config, dict) or "checkpoint_path" not in checkpoint_config:
        raise ValueError("config requires sleepwm.checkpoint_path")
    checkpoint_path = Path(args.checkpoint or checkpoint_config["checkpoint_path"])
    checkpoint = load_checkpoint(checkpoint_path)
    config = copy.deepcopy(checkpoint["config"])
    for section in ("experiment", "data", "physiology", "train"):
        config.setdefault(section, {}).update(protocol.get(section, {}))
    config["train"]["batch_size"] = int(protocol["train"].get("batch_size", 16))
    config["train"]["num_workers"] = int(protocol["train"].get("num_workers", 4))
    seed_everything(int(config["experiment"]["seed"]))
    device = resolve_device(str(config["train"].get("device", "cuda")))
    model = build_student(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.requires_grad_(False)
    model.eval()
    adapter = None
    adapter_checkpoint_path = None
    if args.adapter_checkpoint:
        adapter_checkpoint_path = Path(args.adapter_checkpoint)
        adapter_checkpoint = load_checkpoint(adapter_checkpoint_path)
        adapter_config = adapter_checkpoint["config"]["outcome_adapter"]
        adapter = TrajectoryOutcomeAdapter(
            state_dim=model.encoder.config.d_model,
            modality_count=len(model.encoder.config.modalities),
            num_classes=int(config["data"].get("num_classes", 5)),
            physiology_features=len(config["physiology"]["feature_names"]),
            hidden_dim=int(adapter_config.get("hidden_dim", 128)),
            maximum_stage_delta=float(
                adapter_config.get("maximum_stage_delta", 2.0)
            ),
            maximum_physiology_delta=float(
                adapter_config.get("maximum_physiology_delta", 1.0)
            ),
        ).to(device)
        adapter.load_state_dict(adapter_checkpoint["adapter_state"], strict=True)
        adapter.requires_grad_(False)
        adapter.eval()
    threshold_payload = None
    threshold_path = None
    if args.event_thresholds:
        threshold_path = Path(args.event_thresholds)
        threshold_payload = json.loads(threshold_path.read_text(encoding="utf-8"))
        if threshold_payload.get("split") != "validation":
            raise ValueError("event thresholds must originate from validation")

    dataset = physio_feature_sequence_dataset(config, args.split)
    if args.smoke:
        dataset = Subset(dataset, range(min(48, len(dataset))))
        config["train"]["num_workers"] = 0

    requested = set(args.condition or [])
    specs = [
        spec
        for spec in protocol_specs(tuple(config["data"]["modalities"]))
        if not requested or condition_name(spec) in requested
    ]
    if requested - {condition_name(spec) for spec in specs}:
        raise ValueError(f"unknown conditions: {sorted(requested)}")

    conditions = {}
    for spec in specs:
        name = condition_name(spec)
        print(f"evaluating condition={name}", flush=True)
        fixed = None
        if threshold_payload is not None:
            fixed = threshold_payload["conditions"].get(name)
            if fixed is None:
                raise ValueError(f"event thresholds do not contain condition={name}")
        conditions[name] = evaluate_condition(
            model, adapter, dataset, config, device, spec, fixed
        )

    primary_horizons = tuple(
        int(value) for value in protocol["evaluation"]["primary_horizons_epochs"]
    )
    payload = {
        "protocol": {
            "id": config["experiment"]["id"],
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "outcome_adapter_checkpoint": (
                None if adapter_checkpoint_path is None else str(adapter_checkpoint_path)
            ),
            "outcome_adapter_sha256": (
                None
                if adapter_checkpoint_path is None
                else file_sha256(adapter_checkpoint_path)
            ),
            "split": args.split,
            "test_accessed": args.split == "test",
            "routes": list(ROUTES),
            "primary_horizons_epochs": list(primary_horizons),
            "primary_horizons_seconds": [
                horizon * int(config["data"]["epoch_seconds"])
                for horizon in primary_horizons
            ],
            "event_threshold_protocol": (
                "validation subjects hash-split 40/60 calibration/evaluation"
                if threshold_payload is None
                else "frozen validation thresholds applied unchanged"
            ),
            "event_threshold_path": None if threshold_path is None else str(threshold_path),
            "event_threshold_sha256": (
                None if threshold_path is None else file_sha256(threshold_path)
            ),
            "smoke": args.smoke,
        },
        "summary": primary_summary(conditions, primary_horizons),
        "conditions": conditions,
    }
    output_dir = Path(args.output_dir or protocol["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(output_dir / "summary.md", payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {output_dir / 'metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
