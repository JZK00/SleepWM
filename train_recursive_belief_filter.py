from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Subset

from uniphysio_wm.data import balanced_class_weights
from uniphysio_wm.engine import (
    data_loader,
    load_checkpoint,
    physio_feature_sequence_dataset,
    resolve_device,
    save_checkpoint,
    seed_everything,
)
from uniphysio_wm.mainline import build_mainline_model
from uniphysio_wm.metrics import classification_metrics
from uniphysio_wm.partial_observation import (
    DynamicObservationSpec,
    dynamic_observation_view,
)
from uniphysio_wm.partial_observation_model import (
    RecursiveBeliefCarryCorrectWorldModel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the SleepWM causal latent belief model."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def model_kwargs(config: dict) -> dict:
    partial = config["partial_observation"]
    belief = config["belief_filter"]
    return {
        "partial_filter_hidden_dim": int(partial.get("hidden_dim", 128)),
        "initial_freshness_decay": float(partial.get("initial_freshness_decay", 0.35)),
        "initial_uncertainty_age_scale": float(
            partial.get("initial_uncertainty_age_scale", 0.15)
        ),
        "initial_uncertainty_horizon_scale": float(
            partial.get("initial_uncertainty_horizon_scale", 0.05)
        ),
        "belief_hidden_dim": int(belief.get("hidden_dim", 128)),
        "belief_max_delta": float(belief.get("maximum_delta", 0.5)),
        "belief_use_dynamics": bool(belief.get("use_dynamics", True)),
        "belief_correction_mode": str(
            belief.get("correction_mode", "learned")
        ),
    }


def build_student(config: dict) -> RecursiveBeliefCarryCorrectWorldModel:
    return build_mainline_model(
        config,
        model_class=RecursiveBeliefCarryCorrectWorldModel,
        extra_model_kwargs=model_kwargs(config),
    )


def load_models(config: dict, device: torch.device):
    teacher_checkpoint = load_checkpoint(config["baseline"]["checkpoint_path"])
    teacher = build_mainline_model(config).to(device)
    teacher.load_state_dict(teacher_checkpoint["model_state"], strict=True)
    teacher.requires_grad_(False)
    teacher.eval()

    student_checkpoint = load_checkpoint(config["belief_filter"]["initial_checkpoint"])
    student = build_student(config).to(device)
    incompatible = student.load_state_dict(student_checkpoint["model_state"], strict=False)
    expected = ("belief_transition.", "belief_correction_gate.")
    invalid_missing = [key for key in incompatible.missing_keys if not key.startswith(expected)]
    if invalid_missing or incompatible.unexpected_keys:
        raise ValueError(
            "belief initialization mismatch: "
            f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
        )
    student.requires_grad_(False)
    for parameter in student.belief_parameters():
        parameter.requires_grad = True
    return teacher, student, student_checkpoint


def set_belief_train_mode(model: RecursiveBeliefCarryCorrectWorldModel) -> None:
    model.eval()
    model.belief_transition.train()
    model.belief_correction_gate.train()


def protocol_specs(modalities: tuple[str, ...]):
    all_for = lambda duration: {modality: duration for modality in modalities}
    specs = [None] * 5
    for duration in (1, 2, 4, 10):
        specs.append(DynamicObservationSpec(f"tail_eeg_{duration}ep", {"EEG": duration}))
        specs.append(DynamicObservationSpec(f"tail_all_{duration}ep", all_for(duration)))
    specs.extend(
        [
            DynamicObservationSpec("tail_ecg_4ep", {"ECG": 4}),
            DynamicObservationSpec("tail_emg_4ep", {"EMG": 4}),
            DynamicObservationSpec(
                "recover_all_4ep_after_1ep", all_for(4), recovery_epochs=1
            ),
            DynamicObservationSpec(
                "recover_all_4ep_after_2ep", all_for(4), recovery_epochs=2
            ),
        ]
    )
    return tuple(specs)


def dynamic_view(signals, present, modalities, spec: Optional[DynamicObservationSpec]):
    if spec is None:
        return signals, present, torch.zeros_like(present)
    interrupted, interrupted_present, _ = dynamic_observation_view(
        signals, present, modalities, spec
    )
    artificial_missing = present & ~interrupted_present
    return interrupted, interrupted_present, artificial_missing


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(values.dtype)
    return (values * weight).sum() / weight.sum().clamp_min(1.0)


def belief_losses(output: dict, teacher: dict, artificial_missing: torch.Tensor) -> dict:
    belief = output["belief_trajectory"]
    teacher_states = teacher["history_epoch_states"]
    missing_epoch = artificial_missing.any(dim=-1)
    recovery_epoch = torch.zeros_like(missing_epoch)
    recovery_epoch[:, 1:] = missing_epoch[:, :-1] & ~missing_epoch[:, 1:]
    supervised_epoch = missing_epoch | recovery_epoch

    trajectory_error = F.smooth_l1_loss(
        belief, teacher_states, reduction="none"
    ).mean(dim=-1)
    trajectory_cosine = 1.0 - F.cosine_similarity(
        belief, teacher_states, dim=-1
    )
    trajectory = masked_mean(
        trajectory_error + trajectory_cosine, supervised_epoch
    )

    belief_delta = belief[:, 1:] - belief[:, :-1]
    teacher_delta = teacher_states[:, 1:] - teacher_states[:, :-1]
    delta_error = F.smooth_l1_loss(
        belief_delta, teacher_delta, reduction="none"
    ).mean(dim=-1)
    delta_cosine = 1.0 - F.cosine_similarity(
        belief_delta, teacher_delta, dim=-1
    )
    dynamics = masked_mean(
        delta_error + 0.25 * delta_cosine, supervised_epoch[:, 1:]
    )
    return {
        "trajectory": trajectory,
        "dynamics": dynamics,
        "supervised_epochs": supervised_epoch.sum(),
    }


def train_epoch(teacher, student, dataset, config, optimizer, class_weights, device, epoch):
    set_belief_train_mode(student)
    modalities = tuple(config["data"]["modalities"])
    specs = protocol_specs(modalities)
    weights = config["belief_filter"]["losses"]
    totals = {"loss": 0.0, "trajectory": 0.0, "dynamics": 0.0, "future": 0.0, "stage": 0.0}
    samples = 0
    for batch_index, batch in enumerate(data_loader(dataset, config, shuffle=True)):
        natural_signals = batch["history_signals"].to(device=device, dtype=torch.float32)
        natural_present = batch["history_present"].to(device=device, dtype=torch.bool)
        spec = specs[(batch_index + epoch) % len(specs)]
        signals, present, artificial_missing = dynamic_view(
            natural_signals, natural_present, modalities, spec
        )
        with torch.no_grad():
            teacher_output = teacher.rollout_context(natural_signals, natural_present)
        output = student.rollout_context(signals, present)
        path = belief_losses(output, teacher_output, artificial_missing)
        future = F.smooth_l1_loss(
            output["predicted_states"], teacher_output["predicted_states"]
        )
        labels = batch["future_labels"].to(device=device, dtype=torch.long)
        stage = F.cross_entropy(
            output["stage_logits"].flatten(0, 1),
            labels[:, : output["stage_logits"].shape[1]].flatten(),
            weight=class_weights,
        )
        total = (
            float(weights.get("trajectory", 4.0)) * path["trajectory"]
            + float(weights.get("dynamics", 2.0)) * path["dynamics"]
            + float(weights.get("future", 1.0)) * future
            + float(weights.get("stage", 0.25)) * stage
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(student.belief_parameters()),
            float(config["train"].get("grad_clip", 1.0)),
        )
        optimizer.step()
        batch_size = len(signals)
        samples += batch_size
        values = {
            "loss": total,
            "trajectory": path["trajectory"],
            "dynamics": path["dynamics"],
            "future": future,
            "stage": stage,
        }
        for name, value in values.items():
            totals[name] += float(value.detach().cpu()) * batch_size
    return {name: value / samples for name, value in totals.items()}


def validation_specs(modalities):
    all_for = lambda duration: {modality: duration for modality in modalities}
    return (
        None,
        DynamicObservationSpec("tail_eeg_1ep", {"EEG": 1}),
        DynamicObservationSpec("tail_eeg_2ep", {"EEG": 2}),
        DynamicObservationSpec("tail_eeg_4ep", {"EEG": 4}),
        DynamicObservationSpec("tail_eeg_10ep", {"EEG": 10}),
        DynamicObservationSpec("tail_all_1ep", all_for(1)),
        DynamicObservationSpec("tail_all_2ep", all_for(2)),
        DynamicObservationSpec("tail_all_4ep", all_for(4)),
        DynamicObservationSpec("tail_all_10ep", all_for(10)),
    )


def evaluate(teacher, student, dataset, config, device):
    teacher.eval()
    student.eval()
    modalities = tuple(config["data"]["modalities"])
    specs = validation_specs(modalities)
    names = ["full_observation" if spec is None else spec.name for spec in specs]
    accumulators = {
        name: {
            "logits": [],
            "observation_logits": [],
            "persistence_logits": [],
            "labels": [],
            "belief": [],
            "persistence": [],
            "observation": [],
            "teacher": [],
            "belief_delta": [],
            "persistence_delta": [],
            "teacher_delta": [],
            "missing_age": [],
            "delta_age": [],
        }
        for name in names
    }
    with torch.inference_mode():
        for batch in data_loader(dataset, config, shuffle=False):
            natural_signals = batch["history_signals"].to(device=device, dtype=torch.float32)
            natural_present = batch["history_present"].to(device=device, dtype=torch.bool)
            teacher_output = teacher.rollout_context(natural_signals, natural_present)
            labels = batch["future_labels"]
            for name, spec in zip(names, specs):
                signals, present, artificial_missing = dynamic_view(
                    natural_signals, natural_present, modalities, spec
                )
                output = student.rollout_context(signals, present)
                acc = accumulators[name]
                count = output["stage_logits"].shape[1]
                acc["logits"].append(output["stage_logits"].cpu())
                acc["observation_logits"].append(
                    output["belief_base_stage_logits"].cpu()
                )
                persistence_delta = (
                    output["belief_persistence_trajectory"][:, -1]
                    - output["history_epoch_states"][:, -1]
                )
                persistence_states = (
                    output["belief_base_predicted_states"]
                    + persistence_delta.unsqueeze(1)
                )
                persistence_logits = output["belief_base_stage_logits"] + student.stage_head(
                    persistence_states
                ) - student.stage_head(output["belief_base_predicted_states"])
                if student.current_stage_head is not None:
                    persistence_history = (
                        output["belief_base_corrected_history_state"]
                        + persistence_delta
                    )
                    persistence_logits = persistence_logits + (
                        student.current_stage_head(persistence_history)
                        - student.current_stage_head(
                            output["belief_base_corrected_history_state"]
                        )
                    ).unsqueeze(1)
                persistence_task, _ = student._task_residuals(
                    persistence_states, output["observation_reliability"]
                )
                base_task, _ = student._task_residuals(
                    output["belief_base_predicted_states"],
                    output["observation_reliability"],
                )
                persistence_logits = persistence_logits + persistence_task - base_task
                acc["persistence_logits"].append(persistence_logits.cpu())
                acc["labels"].append(labels[:, :count].cpu())
                missing_epoch = artificial_missing.any(dim=-1)
                if missing_epoch.any():
                    missing_age = torch.zeros_like(missing_epoch, dtype=torch.long)
                    for time_index in range(missing_epoch.shape[1]):
                        previous = (
                            torch.zeros_like(missing_age[:, 0])
                            if time_index == 0
                            else missing_age[:, time_index - 1]
                        )
                        missing_age[:, time_index] = torch.where(
                            missing_epoch[:, time_index],
                            previous + 1,
                            torch.zeros_like(previous),
                        )
                    acc["belief"].append(output["belief_trajectory"][missing_epoch].cpu())
                    acc["persistence"].append(
                        output["belief_persistence_trajectory"][missing_epoch].cpu()
                    )
                    acc["observation"].append(output["history_epoch_states"][missing_epoch].cpu())
                    acc["teacher"].append(teacher_output["history_epoch_states"][missing_epoch].cpu())
                    acc["missing_age"].append(missing_age[missing_epoch].cpu())
                    transition_mask = missing_epoch[:, 1:]
                    belief_delta = (
                        output["belief_trajectory"][:, 1:]
                        - output["belief_trajectory"][:, :-1]
                    )
                    persistence_delta_path = (
                        output["belief_persistence_trajectory"][:, 1:]
                        - output["belief_persistence_trajectory"][:, :-1]
                    )
                    teacher_delta = (
                        teacher_output["history_epoch_states"][:, 1:]
                        - teacher_output["history_epoch_states"][:, :-1]
                    )
                    acc["belief_delta"].append(
                        belief_delta[transition_mask].cpu()
                    )
                    acc["persistence_delta"].append(
                        persistence_delta_path[transition_mask].cpu()
                    )
                    acc["teacher_delta"].append(
                        teacher_delta[transition_mask].cpu()
                    )
                    acc["delta_age"].append(
                        missing_age[:, 1:][transition_mask].cpu()
                    )

    results = {}
    classes = int(config["data"].get("num_classes", 5))
    for name, acc in accumulators.items():
        labels = torch.cat(acc["labels"])
        metrics = classification_metrics(torch.cat(acc["logits"]), labels, classes)
        result = {
            "macro_f1": metrics["macro_f1"],
            "observation_macro_f1": classification_metrics(
                torch.cat(acc["observation_logits"]), labels, classes
            )["macro_f1"],
            "persistence_macro_f1": classification_metrics(
                torch.cat(acc["persistence_logits"]), labels, classes
            )["macro_f1"],
        }
        if acc["teacher"]:
            teacher_states = torch.cat(acc["teacher"])
            for route in ("belief", "persistence", "observation"):
                states = torch.cat(acc[route])
                result[f"{route}_smooth_l1"] = float(
                    F.smooth_l1_loss(states, teacher_states)
                )
                result[f"{route}_cosine"] = float(
                    F.cosine_similarity(states, teacher_states, dim=-1).mean()
                )
            belief_state_error = F.smooth_l1_loss(
                torch.cat(acc["belief"]), teacher_states, reduction="none"
            ).mean(dim=-1)
            persistence_state_error = F.smooth_l1_loss(
                torch.cat(acc["persistence"]), teacher_states, reduction="none"
            ).mean(dim=-1)
            result["belief_win_rate_vs_persistence"] = float(
                (belief_state_error < persistence_state_error).float().mean()
            )
            belief_delta = torch.cat(acc["belief_delta"])
            persistence_delta = torch.cat(acc["persistence_delta"])
            teacher_delta = torch.cat(acc["teacher_delta"])
            result["belief_delta_smooth_l1"] = float(
                F.smooth_l1_loss(belief_delta, teacher_delta)
            )
            result["persistence_delta_smooth_l1"] = float(
                F.smooth_l1_loss(persistence_delta, teacher_delta)
            )
            result["belief_delta_cosine"] = float(
                F.cosine_similarity(belief_delta, teacher_delta, dim=-1).mean()
            )
            result["belief_step_norm"] = float(belief_delta.norm(dim=-1).mean())
            result["persistence_step_norm"] = float(
                persistence_delta.norm(dim=-1).mean()
            )
            result["teacher_step_norm"] = float(teacher_delta.norm(dim=-1).mean())
            missing_age = torch.cat(acc["missing_age"])
            delta_age = torch.cat(acc["delta_age"])
            route_states = {
                "belief": torch.cat(acc["belief"]),
                "persistence": torch.cat(acc["persistence"]),
                "observation": torch.cat(acc["observation"]),
            }
            result["by_missing_age"] = {}
            for age_value in sorted(int(value) for value in missing_age.unique()):
                selected_state = missing_age == age_value
                selected_delta = delta_age == age_value
                age_result = {}
                for route, states in route_states.items():
                    age_result[f"{route}_smooth_l1"] = float(
                        F.smooth_l1_loss(
                            states[selected_state], teacher_states[selected_state]
                        )
                    )
                if selected_delta.any():
                    age_result["belief_delta_smooth_l1"] = float(
                        F.smooth_l1_loss(
                            belief_delta[selected_delta], teacher_delta[selected_delta]
                        )
                    )
                    age_result["persistence_delta_smooth_l1"] = float(
                        F.smooth_l1_loss(
                            persistence_delta[selected_delta], teacher_delta[selected_delta]
                        )
                    )
                result["by_missing_age"][str(age_value)] = age_result
        results[name] = result

    dynamic_names = names[1:]
    results["selection"] = {
        "full_macro_f1": results["full_observation"]["macro_f1"],
        "dynamic_macro_f1": sum(results[name]["macro_f1"] for name in dynamic_names) / len(dynamic_names),
        "belief_smooth_l1": sum(results[name]["belief_smooth_l1"] for name in dynamic_names) / len(dynamic_names),
        "persistence_smooth_l1": sum(results[name]["persistence_smooth_l1"] for name in dynamic_names) / len(dynamic_names),
        "observation_smooth_l1": sum(results[name]["observation_smooth_l1"] for name in dynamic_names) / len(dynamic_names),
    }
    return results


def main() -> int:
    args = parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    teacher_checkpoint = load_checkpoint(protocol["baseline"]["checkpoint_path"])
    config = copy.deepcopy(teacher_checkpoint["config"])
    for section in ("experiment", "baseline", "data", "physiology", "train", "partial_observation", "belief_filter"):
        config.setdefault(section, {}).update(protocol.get(section, {}))
    if args.seed is not None:
        config["experiment"]["seed"] = int(args.seed)
    if args.initial_checkpoint:
        config["belief_filter"]["initial_checkpoint"] = args.initial_checkpoint
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.output_dir:
        config["train"]["output_dir"] = args.output_dir
    if args.smoke:
        config["train"]["epochs"] = 1
        config["train"]["num_workers"] = 0

    seed_everything(int(config["experiment"]["seed"]))
    device = resolve_device(str(config["train"].get("device", "cuda")))
    output_dir = Path(config["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher, student, initialization = load_models(config, device)
    train_data = physio_feature_sequence_dataset(config, "train")
    val_data = physio_feature_sequence_dataset(config, "val")
    if args.smoke:
        train_data = Subset(train_data, range(min(64, len(train_data))))
        val_data = Subset(val_data, range(min(64, len(val_data))))
    counts = train_data.dataset.future_label_counts if isinstance(train_data, Subset) else train_data.future_label_counts
    class_weights = balanced_class_weights(counts).to(device)
    optimizer = torch.optim.AdamW(
        list(student.belief_parameters()),
        lr=float(config["belief_filter"].get("learning_rate", 2e-4)),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )

    best_key = (0.0, float("-inf"), float("-inf"))
    best_epoch = 0
    curve = []
    full_reference = float(config["belief_filter"]["full_reference_macro_f1"])
    maximum_full_drop = float(config["belief_filter"].get("maximum_full_drop", 0.005))
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_metrics = train_epoch(
            teacher, student, train_data, config, optimizer, class_weights, device, epoch
        )
        validation = evaluate(teacher, student, val_data, config, device)
        selected = validation["selection"]
        trajectory_pass = (
            selected["belief_smooth_l1"] < selected["persistence_smooth_l1"]
            and selected["belief_smooth_l1"] < selected["observation_smooth_l1"]
        )
        eligible = (
            selected["full_macro_f1"] >= full_reference - maximum_full_drop
            and trajectory_pass
        )
        key = (
            1.0 if eligible else 0.0,
            float(selected["dynamic_macro_f1"]),
            -float(selected["belief_smooth_l1"]),
        )
        curve.append({"epoch": epoch, "train": train_metrics, "validation": validation, "eligible": eligible})
        print(
            f"epoch={epoch:03d} loss={train_metrics['loss']:.5f} "
            f"dynamic_f1={selected['dynamic_macro_f1']:.4f} "
            f"belief={selected['belief_smooth_l1']:.5f} "
            f"persist={selected['persistence_smooth_l1']:.5f} "
            f"observe={selected['observation_smooth_l1']:.5f} eligible={eligible}",
            flush=True,
        )
        if key > best_key:
            best_key = key
            best_epoch = epoch
            save_checkpoint(
                output_dir / "best.pt",
                {
                    "experiment_id": config["experiment"]["id"],
                    "epoch": epoch,
                    "model_state": student.state_dict(),
                    "validation": validation,
                    "config": config,
                    "initial_checkpoint": config["belief_filter"]["initial_checkpoint"],
                    "initial_epoch": initialization.get("epoch"),
                },
            )
    if best_epoch < 1:
        raise RuntimeError("no recursive belief checkpoint was selected")
    selected_checkpoint = load_checkpoint(output_dir / "best.pt")
    student.load_state_dict(selected_checkpoint["model_state"], strict=True)
    validation = evaluate(teacher, student, val_data, config, device)
    metrics = {
        "best_epoch": best_epoch,
        "validation": validation,
        "training_curve": curve,
        "initial_checkpoint": config["belief_filter"]["initial_checkpoint"],
        "train_sequences": len(train_data),
        "validation_sequences": len(val_data),
        "test_split_accessed": False,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "resolved_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, **validation["selection"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
