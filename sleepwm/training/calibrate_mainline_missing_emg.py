from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
from torch.utils.data import Subset

from sleepwm.protocols import integration_gate, state_identity_check
from sleepwm.training.gated_long_rollout import build_model as build_recursive_model
from sleepwm.training.mainline_missing_waveform import evaluate_subsets
from sleepwm.config import load_config, with_manifest, with_normalization
from sleepwm.engine import data_loader, load_checkpoint, physio_feature_sequence_dataset, prepare_run, resolve_device, save_checkpoint, seed_everything, write_json
from sleepwm.mainline import build_mainline_model
from sleepwm.masking import nonempty_modality_subsets
from sleepwm.masking import forced_history_view


TEACHER_BIAS_KEY = "waveform_decoder.missing_emg_log_rms_bias"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit train-only conditional intercepts for missing EMG RMS."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--normalization")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_i2_initialization(model, checkpoint_path: str) -> dict:
    checkpoint = load_checkpoint(checkpoint_path)
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    invalid_missing = [
        key for key in incompatible.missing_keys if key != TEACHER_BIAS_KEY
    ]
    if incompatible.unexpected_keys or invalid_missing:
        raise ValueError(
            "I2 teacher calibration initialization mismatch: "
            f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
        )
    for parameter in model.parameters():
        parameter.requires_grad = False
    return checkpoint


def fit_teacher_scale(model, dataset, config: dict, device: torch.device) -> dict:
    modalities = tuple(config["data"]["modalities"])
    emg_index = modalities.index("EMG")
    missing_subsets = (("EEG",), ("ECG",), ("EEG", "ECG"))
    condition_names = tuple("+".join(subset) for subset in missing_subsets)
    statistics = {
        name: {"target_rms_sum": 0.0, "missing_rms_sum": 0.0, "sample_count": 0}
        for name in condition_names
    }
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader(dataset, config, shuffle=False)):
            history = batch["history_signals"].to(device, torch.float32)
            present = batch["history_present"].to(device, torch.bool)
            subset = missing_subsets[batch_index % len(missing_subsets)]
            missing_history, missing_present = forced_history_view(
                history, present, subset, modalities
            )
            missing_output = model.rollout(
                missing_history, missing_present
            )["future_waveforms"][:, emg_index]
            selected = present[:, :, emg_index].all(dim=1) & batch[
                "future_present"
            ][:, 0, emg_index].to(device, torch.bool)
            if selected.any():
                missing_rms = missing_output[selected].square().mean(dim=-1).sqrt()
                target = batch["future_signals"][
                    :, 0, emg_index, : missing_output.shape[-1]
                ].to(device, torch.float32)
                target_rms = target[selected].square().mean(dim=-1).sqrt()
                condition = statistics["+".join(subset)]
                condition["target_rms_sum"] += float(target_rms.sum().cpu())
                condition["missing_rms_sum"] += float(missing_rms.sum().cpu())
                condition["sample_count"] += int(selected.sum().cpu())
    minimum = float(config["teacher_calibration"].get("minimum_scale", 0.25))
    maximum = float(config["teacher_calibration"].get("maximum_scale", 4.0))
    if model.waveform_decoder.missing_emg_log_rms_bias is None:
        raise ValueError("missing EMG teacher calibration is not enabled")
    fitted_scales = []
    condition_results = {}
    for name in condition_names:
        condition = statistics[name]
        count = condition["sample_count"]
        missing_sum = condition["missing_rms_sum"]
        if count < 1 or missing_sum <= 0.0:
            raise ValueError(f"training split has no valid {name} EMG samples")
        raw_scale = condition["target_rms_sum"] / missing_sum
        fitted_scale = min(max(raw_scale, minimum), maximum)
        fitted_scales.append(fitted_scale)
        condition_results[name] = {
            "sample_count": count,
            "target_rms_mean": condition["target_rms_sum"] / count,
            "missing_generated_rms_mean": missing_sum / count,
            "raw_scale": raw_scale,
            "fitted_scale": fitted_scale,
        }
    model.waveform_decoder.missing_emg_log_rms_bias.data.copy_(
        torch.tensor(
            [math.log(scale) for scale in fitted_scales],
            device=model.waveform_decoder.missing_emg_log_rms_bias.device,
        )
    )
    return {
        "condition_order": list(condition_names),
        "conditions": condition_results,
        "sample_count": sum(item["sample_count"] for item in condition_results.values()),
        "scale_range": [minimum, maximum],
        "fit_split": "train",
        "training_targets_used": True,
        "validation_targets_used": False,
    }


def main() -> int:
    args = parse_args()
    config = with_normalization(
        with_manifest(load_config(args.config), args.manifest), args.normalization
    )
    config = copy.deepcopy(config)
    if args.seed is not None:
        config["experiment"]["seed"] = int(args.seed)
    if args.initial_checkpoint:
        config["teacher_calibration"]["initial_checkpoint"] = str(
            Path(args.initial_checkpoint)
        )
    if args.output_dir:
        config["train"]["output_dir"] = str(Path(args.output_dir))
    if args.smoke:
        config["train"]["num_workers"] = 0
    seed_everything(int(config["experiment"]["seed"]))
    device = resolve_device(str(config["train"].get("device", "cuda")))
    run_dir = prepare_run(config)
    model = build_mainline_model(config)
    initial_checkpoint = load_i2_initialization(
        model, str(config["teacher_calibration"]["initial_checkpoint"])
    )
    model.to(device)
    train_data = physio_feature_sequence_dataset(config, "train")
    val_data = physio_feature_sequence_dataset(config, "val")
    if args.smoke:
        minimum_conditions = len(("EEG", "ECG", "EEG+ECG"))
        smoke_train_size = int(config["train"]["batch_size"]) * minimum_conditions
        train_data = Subset(train_data, range(min(smoke_train_size, len(train_data))))
        val_data = Subset(val_data, range(min(64, len(val_data))))
    calibration = fit_teacher_scale(model, train_data, config, device)
    modalities = tuple(config["data"]["modalities"])
    results = evaluate_subsets(
        model,
        val_data,
        config,
        device,
        nonempty_modality_subsets(modalities),
    )
    recursive_checkpoint = load_checkpoint(config["mainline"]["recursive_checkpoint"])
    recursive_config = recursive_checkpoint.get("config")
    if not isinstance(recursive_config, dict):
        raise ValueError("R3 checkpoint does not contain its resolved config")
    recursive_model = build_recursive_model(recursive_config).to(device)
    recursive_model.load_state_dict(recursive_checkpoint["model_state"], strict=True)
    state_identity = state_identity_check(
        model, recursive_model, val_data, config, device
    )
    del recursive_model
    waveform_checkpoint = load_checkpoint(config["mainline"]["waveform_checkpoint"])
    w5_validation = waveform_checkpoint.get("validation_metrics")
    if not isinstance(w5_validation, dict):
        raise ValueError("W5 checkpoint does not contain validation_metrics")
    i1_checkpoint = load_checkpoint(config["teacher_calibration"]["i1_checkpoint"])
    component_identity = i1_checkpoint.get("component_identity", {})
    if not bool(component_identity.get("passed", False)):
        raise ValueError("I1 checkpoint lacks a passing component identity record")
    gate = integration_gate(
        component_identity,
        state_identity,
        results,
        w5_validation,
        config,
        args.smoke,
    )
    payload = {
        "initial_checkpoint": config["teacher_calibration"]["initial_checkpoint"],
        "initial_epoch": initial_checkpoint.get("epoch"),
        "teacher_calibration": calibration,
        "component_identity": component_identity,
        "state_identity": state_identity,
        "waveform_results": results,
        "w5_reference_validation": w5_validation,
        "gate": gate,
        "train_sequences": len(train_data),
        "validation_sequences": len(val_data),
        "smoke": bool(args.smoke),
        "test_split_accessed": False,
    }
    write_json(run_dir / "metrics.json", payload)
    write_json(run_dir / "mainline_missing_emg_calibration_val.json", payload)
    checkpoint_payload = {
        "experiment_id": config["experiment"]["id"],
        "model_state": model.state_dict(),
        "config": config,
        "initial_checkpoint": config["teacher_calibration"]["initial_checkpoint"],
        "teacher_calibration": calibration,
        "component_identity": component_identity,
        "state_identity": state_identity,
        "gate": gate,
        "test_split_accessed": False,
    }
    save_checkpoint(run_dir / "best.pt", checkpoint_payload)
    print(json.dumps({"teacher_calibration": calibration, "gate": gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
