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
    primary_dynamic_observation_specs,
)
from uniphysio_wm.partial_observation_model import (
    FreshnessAwareCarryCorrectWorldModel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the recency-aware carry-and-correct latent filter."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def partial_model_kwargs(config: dict) -> dict:
    partial = config["partial_observation"]
    return {
        "partial_filter_hidden_dim": int(partial.get("hidden_dim", 128)),
        "initial_freshness_decay": float(
            partial.get("initial_freshness_decay", 0.35)
        ),
        "initial_uncertainty_age_scale": float(
            partial.get("initial_uncertainty_age_scale", 0.15)
        ),
        "initial_uncertainty_horizon_scale": float(
            partial.get("initial_uncertainty_horizon_scale", 0.05)
        ),
    }


def build_student(config: dict) -> FreshnessAwareCarryCorrectWorldModel:
    return build_mainline_model(
        config,
        model_class=FreshnessAwareCarryCorrectWorldModel,
        extra_model_kwargs=partial_model_kwargs(config),
    )


def load_models(config: dict, device: torch.device):
    checkpoint = load_checkpoint(config["baseline"]["checkpoint_path"])
    teacher = build_mainline_model(config).to(device)
    teacher.load_state_dict(checkpoint["model_state"], strict=True)
    teacher.requires_grad_(False)
    teacher.eval()

    student = build_student(config).to(device)
    student_checkpoint_path = config["partial_observation"].get(
        "initial_checkpoint"
    )
    if student_checkpoint_path:
        student_initialization = load_checkpoint(student_checkpoint_path)
        student.load_state_dict(student_initialization["model_state"], strict=True)
        incompatible = None
    else:
        student_initialization = checkpoint
        incompatible = student.load_state_dict(checkpoint["model_state"], strict=False)
    expected_prefixes = (
        "log_freshness_decay",
        "carry_state_filter.",
        "partial_uncertainty_head.",
        "partial_uncertainty_state_norm.",
        "uncertainty_bias",
        "uncertainty_age_log_scale",
        "uncertainty_horizon_log_scale",
    )
    if incompatible is not None:
        unexpected_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(expected_prefixes)
        ]
        if incompatible.unexpected_keys or unexpected_missing:
            raise ValueError(
                "partial-observation initialization mismatch: "
                f"missing={unexpected_missing}, unexpected={incompatible.unexpected_keys}"
            )
    student.requires_grad_(False)
    for parameter in student.partial_observation_parameters():
        parameter.requires_grad = True
    return teacher, student, student_initialization


def set_partial_train_mode(model: FreshnessAwareCarryCorrectWorldModel) -> None:
    model.eval()
    model.carry_state_filter.train()
    model.partial_uncertainty_head.train()
    model.partial_uncertainty_state_norm.train()


def training_specs(config: dict, modalities: tuple[str, ...]):
    dynamic = config["dynamic_observation"]
    specs = list(
        primary_dynamic_observation_specs(
            modalities,
            tuple(int(value) for value in dynamic["duration_epochs"]),
            tuple(int(value) for value in dynamic["recovery_epochs"]),
        )
    )
    full_repeats = int(dynamic.get("full_observation_repeats", 5))
    return [None] * full_repeats + specs


def dynamic_view(
    signals: torch.Tensor,
    present: torch.Tensor,
    modalities: tuple[str, ...],
    spec: Optional[DynamicObservationSpec],
):
    if spec is None:
        return signals, present
    interrupted, interrupted_present, _ = dynamic_observation_view(
        signals, present, modalities, spec
    )
    return interrupted, interrupted_present


def train_epoch(
    teacher,
    student,
    dataset,
    config: dict,
    optimizer,
    class_weights,
    device,
    epoch: int,
) -> dict:
    set_partial_train_mode(student)
    modalities = tuple(config["data"]["modalities"])
    specs = training_specs(config, modalities)
    losses = config["partial_observation"]["losses"]
    totals = {
        "loss": 0.0,
        "history_distillation": 0.0,
        "future_distillation": 0.0,
        "stage": 0.0,
        "uncertainty_nll": 0.0,
    }
    samples = 0
    for batch_index, batch in enumerate(data_loader(dataset, config, shuffle=True)):
        natural_signals = batch["history_signals"].to(
            device=device, dtype=torch.float32
        )
        natural_present = batch["history_present"].to(
            device=device, dtype=torch.bool
        )
        spec = specs[(batch_index + epoch) % len(specs)]
        signals, present = dynamic_view(
            natural_signals, natural_present, modalities, spec
        )
        with torch.no_grad():
            teacher_output = teacher.rollout_context(
                natural_signals, natural_present
            )
        output = student.rollout_context(signals, present)
        labels = batch["future_labels"].to(device=device, dtype=torch.long)
        count = output["stage_logits"].shape[1]

        max_duration = max(
            int(value) for value in config["dynamic_observation"]["duration_epochs"]
        )
        duration_weight = 1.0 + float(
            config["partial_observation"].get("duration_weight_scale", 0.0)
        ) * output["observation_age_epochs"].amax(dim=-1) / float(max_duration)
        history_error = F.smooth_l1_loss(
            output["corrected_history_state"],
            teacher_output["corrected_history_state"],
            reduction="none",
        ).mean(dim=-1) + 1.0 - F.cosine_similarity(
            output["corrected_history_state"],
            teacher_output["corrected_history_state"],
            dim=-1,
        )
        history_distillation = (
            history_error * duration_weight
        ).sum() / duration_weight.sum().clamp_min(1e-6)
        future_error = F.smooth_l1_loss(
            output["predicted_states"][:, :count],
            teacher_output["predicted_states"][:, :count],
            reduction="none",
        ).mean(dim=(-1, -2))
        future_distillation = (
            future_error * duration_weight
        ).sum() / duration_weight.sum().clamp_min(1e-6)
        stage_loss = F.cross_entropy(
            output["stage_logits"].flatten(0, 1),
            labels[:, :count].flatten(),
            weight=class_weights,
        )
        state_error = (
            output["predicted_states"][:, :count]
            - teacher_output["predicted_states"][:, :count]
        ).square().mean(dim=-1).detach()
        log_variance = output["recursive_log_variance"][:, :count]
        uncertainty_nll = 0.5 * (
            torch.exp(-log_variance) * state_error + log_variance
        ).mean()
        total = (
            float(losses.get("history_distillation", 1.0))
            * history_distillation
            + float(losses.get("future_distillation", 1.0))
            * future_distillation
            + float(losses.get("stage", 0.5)) * stage_loss
            + float(losses.get("uncertainty_nll", 0.1)) * uncertainty_nll
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(student.partial_observation_parameters()),
            float(config["train"].get("grad_clip", 1.0)),
        )
        optimizer.step()

        batch_size = len(signals)
        samples += batch_size
        values = {
            "loss": total,
            "history_distillation": history_distillation,
            "future_distillation": future_distillation,
            "stage": stage_loss,
            "uncertainty_nll": uncertainty_nll,
        }
        for name, value in values.items():
            totals[name] += float(value.detach().cpu()) * batch_size
    return {name: value / samples for name, value in totals.items()}


def validation_specs(modalities: tuple[str, ...]):
    all_missing = {modality: 1 for modality in modalities}
    all_missing_long = {modality: 4 for modality in modalities}
    return (
        None,
        DynamicObservationSpec("tail_eeg_1ep", {"EEG": 1}),
        DynamicObservationSpec("tail_eeg_4ep", {"EEG": 4}),
        DynamicObservationSpec("tail_eeg_10ep", {"EEG": 10}),
        DynamicObservationSpec("tail_emg_4ep", {"EMG": 4}),
        DynamicObservationSpec("tail_all_1ep", all_missing),
        DynamicObservationSpec("tail_all_4ep", all_missing_long),
        DynamicObservationSpec(
            "tail_all_10ep", {modality: 10 for modality in modalities}
        ),
    )


def evaluate(teacher, student, dataset, config: dict, device) -> dict:
    teacher.eval()
    student.eval()
    modalities = tuple(config["data"]["modalities"])
    specs = validation_specs(modalities)
    names = ["full_observation" if spec is None else spec.name for spec in specs]
    values = {
        name: {"logits": [], "labels": [], "latent_error": [], "uncertainty": []}
        for name in names
    }
    with torch.inference_mode():
        for batch in data_loader(dataset, config, shuffle=False):
            natural_signals = batch["history_signals"].to(
                device=device, dtype=torch.float32
            )
            natural_present = batch["history_present"].to(
                device=device, dtype=torch.bool
            )
            teacher_output = teacher.rollout_context(
                natural_signals, natural_present
            )
            labels = batch["future_labels"]
            for name, spec in zip(names, specs):
                signals, present = dynamic_view(
                    natural_signals, natural_present, modalities, spec
                )
                output = student.rollout_context(signals, present)
                count = output["stage_logits"].shape[1]
                values[name]["logits"].append(output["stage_logits"].cpu())
                values[name]["labels"].append(labels[:, :count].cpu())
                values[name]["latent_error"].append(
                    F.smooth_l1_loss(
                        output["predicted_states"][:, :count],
                        teacher_output["predicted_states"][:, :count],
                        reduction="none",
                    ).mean(dim=-1).cpu()
                )
                values[name]["uncertainty"].append(
                    output["recursive_log_variance"][:, :count]
                    .mul(0.5)
                    .exp()
                    .cpu()
                )

    results = {}
    classes = int(config["data"].get("num_classes", 5))
    for name, accumulated in values.items():
        logits = torch.cat(accumulated["logits"])
        labels = torch.cat(accumulated["labels"])
        results[name] = {
            "macro_f1": classification_metrics(logits, labels, classes)["macro_f1"],
            "future_latent_smooth_l1": float(
                torch.cat(accumulated["latent_error"]).mean()
            ),
            "uncertainty_scale": float(
                torch.cat(accumulated["uncertainty"]).mean()
            ),
        }
    dynamic_names = [name for name in names if name != "full_observation"]
    results["selection"] = {
        "dynamic_macro_f1": sum(results[name]["macro_f1"] for name in dynamic_names)
        / len(dynamic_names),
        "dynamic_latent_smooth_l1": sum(
            results[name]["future_latent_smooth_l1"] for name in dynamic_names
        )
        / len(dynamic_names),
        "full_macro_f1": results["full_observation"]["macro_f1"],
    }
    return results


def main() -> int:
    args = parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    checkpoint = load_checkpoint(protocol["baseline"]["checkpoint_path"])
    config = copy.deepcopy(checkpoint["config"])
    for section in (
        "experiment",
        "baseline",
        "data",
        "physiology",
        "train",
        "dynamic_observation",
        "partial_observation",
    ):
        config.setdefault(section, {}).update(protocol.get(section, {}))
    if args.seed is not None:
        config["experiment"]["seed"] = int(args.seed)
    if args.initial_checkpoint:
        config["partial_observation"]["initial_checkpoint"] = (
            args.initial_checkpoint
        )
    if args.epochs is not None:
        config["train"]["epochs"] = int(args.epochs)
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
    class_weights = balanced_class_weights(
        train_data.dataset.future_label_counts
        if isinstance(train_data, Subset)
        else train_data.future_label_counts
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(student.partial_observation_parameters()),
        lr=float(config["partial_observation"].get("learning_rate", 2e-4)),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )

    best_key = (0.0, float("-inf"), float("-inf"))
    best_epoch = 0
    curve = []
    full_reference = float(config["partial_observation"]["full_reference_macro_f1"])
    maximum_full_drop = float(
        config["partial_observation"].get("maximum_full_drop", 0.005)
    )
    maximum_dynamic_latent = float(
        config["partial_observation"].get(
            "maximum_dynamic_latent_smooth_l1", float("inf")
        )
    )
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_metrics = train_epoch(
            teacher,
            student,
            train_data,
            config,
            optimizer,
            class_weights,
            device,
            epoch,
        )
        validation = evaluate(teacher, student, val_data, config, device)
        selection = validation["selection"]
        eligible = (
            selection["full_macro_f1"] >= full_reference - maximum_full_drop
            and selection["dynamic_latent_smooth_l1"] <= maximum_dynamic_latent
        )
        key = (
            1.0 if eligible else 0.0,
            float(selection["dynamic_macro_f1"]),
            -float(selection["dynamic_latent_smooth_l1"]),
        )
        curve.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation,
                "eligible": eligible,
            }
        )
        print(
            f"epoch={epoch:03d} loss={train_metrics['loss']:.5f} "
            f"dynamic_f1={selection['dynamic_macro_f1']:.4f} "
            f"full_f1={selection['full_macro_f1']:.4f} "
            f"latent={selection['dynamic_latent_smooth_l1']:.5f} "
            f"eligible={eligible}",
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
                    "initial_checkpoint": config["baseline"]["checkpoint_path"],
                    "initial_epoch": initialization.get("epoch"),
                },
            )
    if best_epoch < 1:
        raise RuntimeError("no partial-observation checkpoint was selected")

    selected = load_checkpoint(output_dir / "best.pt")
    student.load_state_dict(selected["model_state"], strict=True)
    final_validation = evaluate(teacher, student, val_data, config, device)
    metrics = {
        "best_epoch": best_epoch,
        "validation": final_validation,
        "training_curve": curve,
        "initial_checkpoint": config["baseline"]["checkpoint_path"],
        "train_sequences": len(train_data),
        "validation_sequences": len(val_data),
        "test_split_accessed": False,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"best_epoch": best_epoch, **final_validation["selection"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
