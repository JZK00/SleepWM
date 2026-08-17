from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Subset

from evaluate_belief_outcomes import dynamic_view, route_outputs
from train_recursive_belief_filter import build_student
from uniphysio_wm.data import balanced_class_weights
from uniphysio_wm.engine import (
    data_loader,
    load_checkpoint,
    physio_feature_sequence_dataset,
    resolve_device,
    save_checkpoint,
    seed_everything,
)
from uniphysio_wm.metrics import classification_metrics
from uniphysio_wm.outcome_adapter import TrajectoryOutcomeAdapter
from uniphysio_wm.partial_observation import DynamicObservationSpec
from uniphysio_wm.physiology_metrics import standardized_physiology_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train downstream readouts over a frozen SleepWM belief trajectory."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--po3-checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def build_adapter(model, config: dict) -> TrajectoryOutcomeAdapter:
    section = config["outcome_adapter"]
    return TrajectoryOutcomeAdapter(
        state_dim=model.encoder.config.d_model,
        modality_count=len(model.encoder.config.modalities),
        num_classes=int(config["data"].get("num_classes", 5)),
        physiology_features=len(config["physiology"]["feature_names"]),
        hidden_dim=int(section.get("hidden_dim", 128)),
        maximum_stage_delta=float(section.get("maximum_stage_delta", 2.0)),
        maximum_physiology_delta=float(
            section.get("maximum_physiology_delta", 1.0)
        ),
    )


def training_specs(modalities: Sequence[str]) -> tuple[DynamicObservationSpec, ...]:
    all_for = lambda duration: {modality: duration for modality in modalities}
    return (
        DynamicObservationSpec("hard_eeg_1ep", {"EEG": 1}),
        DynamicObservationSpec("hard_all_1ep", all_for(1)),
        DynamicObservationSpec("hard_eeg_4ep", {"EEG": 4}),
        DynamicObservationSpec("hard_all_4ep", all_for(4)),
        DynamicObservationSpec("hard_eeg_10ep", {"EEG": 10}),
        DynamicObservationSpec("hard_all_10ep", all_for(10)),
        DynamicObservationSpec("hard_ecg_4ep", {"ECG": 4}),
        DynamicObservationSpec("hard_emg_4ep", {"EMG": 4}),
        DynamicObservationSpec(
            "linear_decay_all_4ep", all_for(4), profile="linear_decay"
        ),
        DynamicObservationSpec(
            "asynchronous_eeg4_ecg2_emg1",
            {"EEG": 4, "ECG": 2, "EMG": 1},
        ),
    )


def validation_specs(
    modalities: Sequence[str],
) -> tuple[Optional[DynamicObservationSpec], ...]:
    all_for = lambda duration: {modality: duration for modality in modalities}
    return (
        None,
        DynamicObservationSpec("hard_eeg_4ep", {"EEG": 4}),
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
    )


def masked_smooth_l1(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    errors = F.smooth_l1_loss(prediction, target, reduction="none")
    weights = valid.to(errors.dtype)
    return (errors * weights).sum() / weights.sum().clamp_min(1.0)


def train_epoch(
    model,
    adapter,
    dataset,
    config: dict,
    optimizer,
    class_weights: torch.Tensor,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.eval()
    adapter.train()
    modalities = tuple(config["data"]["modalities"])
    specs = training_specs(modalities)
    totals = {"loss": 0.0, "stage": 0.0, "physiology": 0.0, "regularizer": 0.0}
    samples = 0
    section = config["outcome_adapter"]
    for batch_index, batch in enumerate(data_loader(dataset, config, shuffle=True)):
        natural_signals = batch["history_signals"].to(
            device=device, dtype=torch.float32
        )
        natural_present = batch["history_present"].to(
            device=device, dtype=torch.bool
        )
        spec = specs[(batch_index + epoch - 1) % len(specs)]
        signals, present, _ = dynamic_view(
            natural_signals, natural_present, modalities, spec
        )
        with torch.no_grad():
            output = model.rollout_context_horizons(
                signals, present, config["data"]["future_horizons"]
            )
        adapted = adapter(output)
        labels = batch["future_labels"].to(device=device, dtype=torch.long)
        physiology = batch["future_physiology"].to(
            device=device, dtype=torch.float32
        )
        physiology_valid = batch["future_physiology_valid"].to(
            device=device, dtype=torch.bool
        )
        stage_loss = F.cross_entropy(
            adapted["stage_logits"].flatten(0, 1),
            labels.flatten(),
            weight=class_weights,
        )
        physiology_loss = masked_smooth_l1(
            adapted["future_physiology"], physiology, physiology_valid
        )
        regularizer = (
            adapted["stage_delta"].square().mean()
            + adapted["physiology_delta"].square().mean()
        )
        loss = (
            float(section.get("stage_weight", 1.0)) * stage_loss
            + float(section.get("physiology_weight", 1.0)) * physiology_loss
            + float(section.get("regularizer_weight", 0.001)) * regularizer
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            adapter.parameters(), float(config["train"].get("grad_clip", 1.0))
        )
        optimizer.step()
        batch_size = len(signals)
        samples += batch_size
        for name, value in (
            ("loss", loss),
            ("stage", stage_loss),
            ("physiology", physiology_loss),
            ("regularizer", regularizer),
        ):
            totals[name] += float(value.detach().cpu()) * batch_size
    return {name: value / samples for name, value in totals.items()}


@torch.inference_mode()
def evaluate(model, adapter, dataset, config: dict, device: torch.device) -> dict:
    model.eval()
    adapter.eval()
    modalities = tuple(config["data"]["modalities"])
    specs = validation_specs(modalities)
    horizons = tuple(int(value) for value in config["data"]["future_horizons"])
    accumulators = {}
    for spec in specs:
        name = "full_observation" if spec is None else spec.name
        accumulators[name] = {
            route: {"logits": [], "physiology": []}
            for route in ("direct_incomplete", "static_persistence", "recursive_belief")
        }
    labels = []
    physiology = []
    physiology_valid = []
    for batch in data_loader(dataset, config, shuffle=False):
        natural_signals = batch["history_signals"].to(
            device=device, dtype=torch.float32
        )
        natural_present = batch["history_present"].to(
            device=device, dtype=torch.bool
        )
        labels.append(batch["future_labels"].cpu())
        physiology.append(batch["future_physiology"].cpu())
        physiology_valid.append(batch["future_physiology_valid"].cpu())
        for spec in specs:
            name = "full_observation" if spec is None else spec.name
            signals, present, _ = dynamic_view(
                natural_signals, natural_present, modalities, spec
            )
            output = model.rollout_context_horizons(signals, present, horizons)
            routes = route_outputs(model, output)
            adapted = adapter(output)
            routes["recursive_belief"]["stage_logits"] = adapted["stage_logits"]
            routes["recursive_belief"]["future_physiology"] = adapted[
                "future_physiology"
            ]
            for route, values in routes.items():
                accumulators[name][route]["logits"].append(
                    values["stage_logits"].cpu()
                )
                accumulators[name][route]["physiology"].append(
                    values["future_physiology"].cpu()
                )

    labels_tensor = torch.cat(labels)
    physiology_tensor = torch.cat(physiology)
    valid_tensor = torch.cat(physiology_valid)
    feature_names = tuple(config["physiology"]["feature_names"])
    feature_groups = {
        group: tuple(names)
        for group, names in config["physiology"]["feature_groups"].items()
    }
    results = {}
    for condition, route_values in accumulators.items():
        results[condition] = {}
        for route, values in route_values.items():
            stage = classification_metrics(
                torch.cat(values["logits"]),
                labels_tensor,
                int(config["data"].get("num_classes", 5)),
            )
            phys = standardized_physiology_metrics(
                torch.cat(values["physiology"]),
                physiology_tensor,
                valid_tensor,
                feature_names,
                feature_groups,
                horizons,
            )
            results[condition][route] = {
                "stage_macro_f1": stage["macro_f1"],
                "physiology_mae": phys["all_features"]["mean_normalized_mae"],
            }
    dynamic = [name for name in results if name != "full_observation"]
    selection = {
        "full_stage_macro_f1": results["full_observation"]["recursive_belief"][
            "stage_macro_f1"
        ],
        "full_physiology_mae": results["full_observation"]["recursive_belief"][
            "physiology_mae"
        ],
        "dynamic_stage_macro_f1": float(
            np.mean(
                [results[name]["recursive_belief"]["stage_macro_f1"] for name in dynamic]
            )
        ),
        "dynamic_physiology_mae": float(
            np.mean(
                [results[name]["recursive_belief"]["physiology_mae"] for name in dynamic]
            )
        ),
    }
    return {"selection": selection, "conditions": results}


def main() -> int:
    args = parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    checkpoint_path = Path(
        args.po3_checkpoint or protocol["po3"]["checkpoint_path"]
    )
    checkpoint = load_checkpoint(checkpoint_path)
    config = copy.deepcopy(checkpoint["config"])
    for section in (
        "experiment",
        "data",
        "physiology",
        "train",
        "outcome_adapter",
    ):
        config.setdefault(section, {}).update(protocol.get(section, {}))
    if args.seed is not None:
        config["experiment"]["seed"] = int(args.seed)
    if args.output_dir:
        config["train"]["output_dir"] = args.output_dir
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.smoke:
        config["train"]["epochs"] = 1
        config["train"]["num_workers"] = 0

    seed_everything(int(config["experiment"]["seed"]))
    device = resolve_device(str(config["train"].get("device", "cuda")))
    model = build_student(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.requires_grad_(False)
    model.eval()
    adapter = build_adapter(model, config).to(device)

    train_data = physio_feature_sequence_dataset(config, "train")
    validation_data = physio_feature_sequence_dataset(config, "val")
    if args.smoke:
        train_data = Subset(train_data, range(min(64, len(train_data))))
        validation_data = Subset(validation_data, range(min(64, len(validation_data))))
    counts = (
        train_data.dataset.future_label_counts
        if isinstance(train_data, Subset)
        else train_data.future_label_counts
    )
    class_weights = balanced_class_weights(counts).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(config["outcome_adapter"].get("learning_rate", 2e-4)),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    baseline = evaluate(model, adapter, validation_data, config, device)
    baseline_selection = baseline["selection"]
    best_key = (float("-inf"), float("-inf"))
    best_epoch = 0
    curve = []
    output_dir = Path(config["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_metrics = train_epoch(
            model,
            adapter,
            train_data,
            config,
            optimizer,
            class_weights,
            device,
            epoch,
        )
        validation = evaluate(model, adapter, validation_data, config, device)
        selected = validation["selection"]
        stage_gain = (
            selected["dynamic_stage_macro_f1"]
            - baseline_selection["dynamic_stage_macro_f1"]
        )
        physiology_gain = (
            baseline_selection["dynamic_physiology_mae"]
            - selected["dynamic_physiology_mae"]
        )
        eligible = stage_gain >= -0.005 and physiology_gain > 0.0
        key = (
            1.0 if eligible else 0.0,
            stage_gain + physiology_gain,
        )
        curve.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation,
                "stage_gain": stage_gain,
                "physiology_gain": physiology_gain,
                "eligible": eligible,
            }
        )
        print(
            f"epoch={epoch:03d} loss={train_metrics['loss']:.5f} "
            f"dynamic_f1={selected['dynamic_stage_macro_f1']:.4f} "
            f"physiology_mae={selected['dynamic_physiology_mae']:.4f} "
            f"stage_gain={stage_gain:+.4f} physiology_gain={physiology_gain:+.4f} "
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
                    "adapter_state": adapter.state_dict(),
                    "validation": validation,
                    "config": config,
                    "po3_checkpoint": str(checkpoint_path),
                    "po3_checkpoint_epoch": checkpoint.get("epoch"),
                },
            )
    selected_checkpoint = load_checkpoint(output_dir / "best.pt")
    adapter.load_state_dict(selected_checkpoint["adapter_state"], strict=True)
    validation = evaluate(model, adapter, validation_data, config, device)
    metrics = {
        "best_epoch": best_epoch,
        "baseline": baseline,
        "validation": validation,
        "training_curve": curve,
        "po3_checkpoint": str(checkpoint_path),
        "po3_modified": False,
        "train_sequences": len(train_data),
        "validation_sequences": len(validation_data),
        "test_split_accessed": False,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"best_epoch": best_epoch, **validation["selection"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
