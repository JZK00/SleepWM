from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Subset


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from dynamic_missing_baselines import build_dynamic_baseline
from evaluate_sleep_events import average_precision, event_risk_scores, load_label_map, next_event_offset
from uniphysio_wm.data import balanced_class_weights
from uniphysio_wm.engine import (
    data_loader,
    load_checkpoint,
    model_size,
    physio_feature_sequence_dataset,
    resolve_device,
    save_checkpoint,
    seed_everything,
)
from uniphysio_wm.metrics import classification_metrics
from uniphysio_wm.partial_observation import DynamicObservationSpec, dynamic_observation_view
from uniphysio_wm.physiology_metrics import standardized_physiology_metrics


PRIMARY_HORIZONS = (1, 2, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matched dynamic-missingness baseline suite.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    return parser.parse_args()


def dynamic_specs(modalities: Sequence[str]) -> tuple[DynamicObservationSpec, ...]:
    all_for = lambda duration: {modality: duration for modality in modalities}
    return (
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
    )


def training_specs(
    modalities: Sequence[str],
) -> tuple[Optional[DynamicObservationSpec], ...]:
    all_for = lambda duration: {modality: duration for modality in modalities}
    specs: list[Optional[DynamicObservationSpec]] = [None] * 5
    for duration in (1, 2, 4, 10):
        specs.append(
            DynamicObservationSpec(f"tail_eeg_{duration}ep", {"EEG": duration})
        )
        specs.append(
            DynamicObservationSpec(f"tail_all_{duration}ep", all_for(duration))
        )
    specs.extend(
        (
            DynamicObservationSpec("tail_ecg_4ep", {"ECG": 4}),
            DynamicObservationSpec("tail_emg_4ep", {"EMG": 4}),
            DynamicObservationSpec(
                "recover_all_4ep_after_1ep", all_for(4), recovery_epochs=1
            ),
            DynamicObservationSpec(
                "recover_all_4ep_after_2ep", all_for(4), recovery_epochs=2
            ),
        )
    )
    return tuple(specs)


def apply_spec(signals, present, modalities, spec: Optional[DynamicObservationSpec]):
    if spec is None:
        return signals, present
    interrupted, interrupted_present, _ = dynamic_observation_view(
        signals, present, modalities, spec
    )
    return interrupted, interrupted_present


def masked_smooth_l1(prediction, target, valid):
    error = F.smooth_l1_loss(prediction, target, reduction="none")
    weight = valid.to(error.dtype)
    return (error * weight).sum() / weight.sum().clamp_min(1.0)


def train_epoch(model, dataset, config, optimizer, class_weights, device, epoch):
    model.train()
    modalities = tuple(config["data"]["modalities"])
    specs = training_specs(modalities)
    stage_weight = float(config["losses"].get("stage", 1.0))
    current_weight = float(config["losses"].get("current_stage", 0.25))
    physiology_weight = float(config["losses"].get("physiology", 1.0))
    consistency_weight = float(config["losses"].get("consistency", 0.1))
    totals = {"loss": 0.0, "stage": 0.0, "current": 0.0, "physiology": 0.0}
    samples = 0
    for batch_index, batch in enumerate(data_loader(dataset, config, shuffle=True)):
        signals = batch["history_signals"].to(device=device, dtype=torch.float32)
        present = batch["history_present"].to(device=device, dtype=torch.bool)
        spec = specs[(batch_index + epoch) % len(specs)]
        signals, present = apply_spec(signals, present, modalities, spec)
        output = model(signals, present)
        current = batch["history_labels"][:, -1].to(device=device, dtype=torch.long)
        future = batch["future_labels"].to(device=device, dtype=torch.long)
        physiology = batch["future_physiology"].to(device=device, dtype=torch.float32)
        physiology_valid = batch["future_physiology_valid"].to(device=device, dtype=torch.bool)
        current_loss = F.cross_entropy(output["current_logits"], current, weight=class_weights)
        stage_loss = F.cross_entropy(
            output["future_logits"].flatten(0, 1), future.flatten(), weight=class_weights
        )
        physiology_loss = masked_smooth_l1(
            output["future_physiology"], physiology, physiology_valid
        )
        loss = (
            current_weight * current_loss
            + stage_weight * stage_loss
            + physiology_weight * physiology_loss
            + consistency_weight * output["consistency_loss"]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["train"].get("grad_clip", 1.0))
        )
        optimizer.step()
        batch_size = len(signals)
        samples += batch_size
        for name, value in (
            ("loss", loss),
            ("stage", stage_loss),
            ("current", current_loss),
            ("physiology", physiology_loss),
        ):
            totals[name] += float(value.detach().cpu()) * batch_size
    return {name: value / max(samples, 1) for name, value in totals.items()}


@torch.inference_mode()
def evaluate_condition(model, dataset, config, device, spec, include_event):
    model.eval()
    modalities = tuple(config["data"]["modalities"])
    horizons = tuple(int(value) for value in config["data"]["future_horizons"])
    primary_indices = tuple(horizons.index(value) for value in PRIMARY_HORIZONS)
    logits, labels = [], []
    physiology_prediction, physiology_target, physiology_valid = [], [], []
    current_probabilities, future_probabilities = [], []
    record_ids, origins = [], []
    history_epochs = int(config["data"]["history_epochs"])
    for batch in data_loader(dataset, config, shuffle=False):
        signals = batch["history_signals"].to(device=device, dtype=torch.float32)
        present = batch["history_present"].to(device=device, dtype=torch.bool)
        signals, present = apply_spec(signals, present, modalities, spec)
        output = model(signals, present)
        logits.append(output["future_logits"].cpu())
        labels.append(batch["future_labels"].cpu())
        physiology_prediction.append(output["future_physiology"].cpu())
        physiology_target.append(batch["future_physiology"].cpu())
        physiology_valid.append(batch["future_physiology_valid"].cpu())
        if include_event:
            current_probabilities.append(output["current_logits"].softmax(dim=-1).cpu())
            future_probabilities.append(output["future_logits"].softmax(dim=-1).cpu())
            record_ids.extend(str(value) for value in batch["record_id"])
            origins.extend((batch["start_epoch"] + history_epochs - 1).tolist())
    logit_tensor = torch.cat(logits)
    label_tensor = torch.cat(labels)
    prediction_tensor = torch.cat(physiology_prediction)
    target_tensor = torch.cat(physiology_target)
    valid_tensor = torch.cat(physiology_valid)
    feature_names = tuple(config["physiology"]["feature_names"])
    feature_groups = {
        key: tuple(value) for key, value in config["physiology"]["feature_groups"].items()
    }
    physiology_metrics = standardized_physiology_metrics(
        prediction_tensor,
        target_tensor,
        valid_tensor,
        feature_names,
        feature_groups,
        horizons,
    )
    by_horizon = {
        str(horizon): {
            "stage_macro_f1": classification_metrics(
                logit_tensor[:, index], label_tensor[:, index], int(config["data"]["num_classes"])
            )["macro_f1"],
            "physiology_mae": physiology_metrics["by_horizon"][str(horizon)][
                "mean_normalized_mae"
            ],
        }
        for index, horizon in enumerate(horizons)
    }
    result = {
        "stage_macro_f1": float(
            np.mean([by_horizon[str(horizons[index])]["stage_macro_f1"] for index in primary_indices])
        ),
        "future_physiology_mae": float(
            np.mean([by_horizon[str(horizons[index])]["physiology_mae"] for index in primary_indices])
        ),
        "by_horizon": by_horizon,
        "sequences": len(label_tensor),
    }
    if include_event:
        risks = event_risk_scores(
            torch.cat(current_probabilities), torch.cat(future_probabilities)
        )["transition"]
        source_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
        label_map = load_label_map(source_dataset)
        max_horizon = max(PRIMARY_HORIZONS)
        offsets = np.asarray(
            [
                next_event_offset(
                    label_map[record], int(origin), "transition", max_horizon
                )
                for record, origin in zip(record_ids, origins)
            ],
            dtype=np.int64,
        )
        event_by_horizon = {}
        for horizon in PRIMARY_HORIZONS:
            target = offsets <= horizon
            score = risks[:, horizons.index(horizon)].numpy()
            event_by_horizon[str(horizon)] = {
                "auprc": average_precision(target, score),
                "event_fraction": float(target.mean()),
            }
        result["transition_auprc_by_horizon"] = event_by_horizon
        result["transition_auprc"] = float(
            np.mean([event_by_horizon[str(horizon)]["auprc"] for horizon in PRIMARY_HORIZONS])
        )
    return result


def evaluate_protocol(model, dataset, config, device, include_event):
    modalities = tuple(config["data"]["modalities"])
    conditions = {
        "full_observation": evaluate_condition(
            model, dataset, config, device, None, include_event
        )
    }
    for spec in dynamic_specs(modalities):
        conditions[spec.name] = evaluate_condition(
            model, dataset, config, device, spec, include_event
        )
    dynamic = [value for key, value in conditions.items() if key != "full_observation"]
    aggregate = {
        "stage_macro_f1": float(np.mean([value["stage_macro_f1"] for value in dynamic])),
        "future_physiology_mae": float(
            np.mean([value["future_physiology_mae"] for value in dynamic])
        ),
    }
    if include_event:
        aggregate["transition_auprc"] = float(
            np.mean([value["transition_auprc"] for value in dynamic])
        )
    return {"full_observation": conditions["full_observation"], "dynamic": aggregate, "conditions": conditions}


def main() -> int:
    args = parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = copy.deepcopy(config)
    if args.seed is not None:
        config["experiment"]["seed"] = int(args.seed)
    if args.epochs is not None:
        config["train"]["epochs"] = int(args.epochs)
    if args.output_dir:
        config["train"]["output_dir"] = args.output_dir
    if args.device:
        config["train"]["device"] = args.device
    if args.smoke:
        config["train"]["epochs"] = 1
        config["train"]["num_workers"] = 0
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    device = resolve_device(str(config["train"].get("device", "cuda")))
    output_dir = Path(config["train"]["output_dir"]) / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    train_data = physio_feature_sequence_dataset(config, "train")
    validation_data = physio_feature_sequence_dataset(config, "val")
    if args.smoke:
        train_data = Subset(train_data, range(min(96, len(train_data))))
        validation_data = Subset(validation_data, range(min(96, len(validation_data))))
    counts = (
        train_data.dataset.future_label_counts
        if isinstance(train_data, Subset)
        else train_data.future_label_counts
    )
    class_weights = balanced_class_weights(counts).to(device)
    model = build_dynamic_baseline(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    best_score = float("-inf")
    curve = []
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        training = train_epoch(
            model, train_data, config, optimizer, class_weights, device, epoch
        )
        validation = evaluate_protocol(
            model, validation_data, config, device, include_event=False
        )
        score = (
            validation["dynamic"]["stage_macro_f1"]
            - validation["dynamic"]["future_physiology_mae"]
        )
        curve.append({"epoch": epoch, "train": training, "validation": validation, "score": score})
        print(
            f"epoch={epoch:03d} loss={training['loss']:.5f} "
            f"dynamic_f1={validation['dynamic']['stage_macro_f1']:.4f} "
            f"physiology_mae={validation['dynamic']['future_physiology_mae']:.4f} "
            f"selection={score:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            save_checkpoint(
                output_dir / "best.pt",
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "selection_score": score,
                    "validation": validation,
                    "config": config,
                },
            )
    checkpoint = load_checkpoint(output_dir / "best.pt")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    validation = evaluate_protocol(
        model, validation_data, config, device, include_event=True
    )
    result = {
        "architecture": config["model"]["architecture"],
        "seed": seed,
        "best_epoch": checkpoint["epoch"],
        "selection": "validation dynamic Macro-F1 minus physiology MAE",
        "validation": validation,
        "resources": model_size(model),
        "training_curve": curve,
        "test_split_accessed": False,
    }
    if not args.skip_test and not args.smoke:
        test_data = physio_feature_sequence_dataset(config, "test")
        result["test"] = evaluate_protocol(
            model, test_data, config, device, include_event=True
        )
        result["test_split_accessed"] = True
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "architecture": result["architecture"],
        "seed": seed,
        "best_epoch": result["best_epoch"],
        "validation": result["validation"]["dynamic"],
        "test": None if "test" not in result else result["test"]["dynamic"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
