from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from sleepwm.protocols import integration_gate, state_identity_check
from sleepwm.training.gated_long_rollout import build_model as build_recursive_model
from sleepwm.training.mainline_missing_waveform import evaluate_subsets, selection_key
from sleepwm.config import load_config, with_manifest, with_normalization
from sleepwm.engine import data_loader, load_checkpoint, physio_feature_sequence_dataset, prepare_run, resolve_device, save_checkpoint, seed_everything, write_json
from sleepwm.mainline import build_mainline_model
from sleepwm.masking import nonempty_modality_subsets
from sleepwm.masking import sampled_history_view
from sleepwm.waveforms import _eeg_targets, _emg_targets


CALIBRATION_PREFIX = "waveform_decoder.missing_physiology_calibration_heads."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate missing EEG spectral energy and EMG RMS."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--normalization")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_i1_initialization(model, checkpoint_path: str) -> dict:
    checkpoint = load_checkpoint(checkpoint_path)
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    invalid_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(CALIBRATION_PREFIX)
    ]
    if incompatible.unexpected_keys or invalid_missing:
        raise ValueError(
            "I1 calibration initialization mismatch: "
            f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
        )
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.waveform_decoder.missing_calibration_parameters():
        parameter.requires_grad = True
    return checkpoint


def trainable_parameters(model) -> list[torch.nn.Parameter]:
    return [
        parameter
        for parameter in model.waveform_decoder.missing_calibration_parameters()
        if parameter.requires_grad
    ]


def _patch_log_rms(waveform: torch.Tensor, patch_samples: int) -> torch.Tensor:
    patches = waveform.reshape(waveform.shape[0], -1, int(patch_samples))
    return patches.square().mean(dim=-1).clamp_min(1e-8).sqrt().log()


def train_epoch(model, dataset, config: dict, optimizer, device: torch.device) -> dict:
    model.eval()
    model.waveform_decoder.missing_physiology_calibration_heads.train()
    totals = {
        "loss": 0.0,
        "eeg_spectral_loss": 0.0,
        "eeg_energy_loss": 0.0,
        "emg_envelope_loss": 0.0,
        "emg_rms_loss": 0.0,
        "emg_generated_rms_loss": 0.0,
    }
    samples_seen = 0
    sample_rate = int(config["data"]["sample_rate"])
    patch_samples = int(config["waveform"]["patch_samples"])
    modalities = tuple(config["data"]["modalities"])
    eeg_index = modalities.index("EEG")
    emg_index = modalities.index("EMG")
    calibration = config["integration_calibration"]
    for batch in data_loader(dataset, config, shuffle=True):
        history = batch["history_signals"].to(device, torch.float32)
        present = batch["history_present"].to(device, torch.bool)
        history, present, _ = sampled_history_view(
            history,
            present,
            float(calibration.get("full_modality_probability", 0.35)),
        )
        availability = present.to(torch.float32).mean(dim=1)
        context = model.rollout_context(history, present)
        prediction, _, probabilities = model.waveform_decoder(
            history[:, -1],
            context["predicted_states"][:, 0],
            context["physiology_dynamics_states"][:, 0],
            return_structure=True,
            return_probabilities=True,
            modality_availability=availability,
        )
        target = batch["future_signals"][:, 0, :, : prediction.shape[-1]].to(
            device, torch.float32
        )
        valid = batch["future_present"][:, 0].to(device, torch.bool)
        zero = prediction.sum() * 0.0

        eeg_selected = (availability[:, eeg_index] < 0.5) & valid[:, eeg_index]
        if eeg_selected.any():
            eeg_target = _eeg_targets(
                target[eeg_selected, eeg_index], patch_samples, sample_rate
            )
            eeg_spectral_loss = F.smooth_l1_loss(
                probabilities["EEG"]["spectral_mean"][eeg_selected], eeg_target
            )
            eeg_energy_loss = F.smooth_l1_loss(
                _patch_log_rms(prediction[eeg_selected, eeg_index], patch_samples),
                _patch_log_rms(target[eeg_selected, eeg_index], patch_samples),
            )
        else:
            eeg_spectral_loss = zero
            eeg_energy_loss = zero

        emg_selected = (availability[:, emg_index] < 0.5) & valid[:, emg_index]
        if emg_selected.any():
            envelope, rms, _ = _emg_targets(
                target[emg_selected, emg_index], patch_samples, sample_rate
            )
            emg_envelope_loss = F.smooth_l1_loss(
                probabilities["EMG"]["envelope_mean"][emg_selected], envelope
            )
            emg_rms_loss = F.smooth_l1_loss(
                probabilities["EMG"]["rms_mean"][emg_selected].clamp_min(1e-8).log(),
                rms.clamp_min(1e-8).log(),
            )
            emg_generated_rms_loss = F.smooth_l1_loss(
                _patch_log_rms(prediction[emg_selected, emg_index], patch_samples),
                _patch_log_rms(target[emg_selected, emg_index], patch_samples),
            )
        else:
            emg_envelope_loss = zero
            emg_rms_loss = zero
            emg_generated_rms_loss = zero

        loss = (
            float(calibration.get("eeg_spectral_weight", 1.0)) * eeg_spectral_loss
            + float(calibration.get("eeg_energy_weight", 1.0)) * eeg_energy_loss
            + float(calibration.get("emg_envelope_weight", 0.5))
            * emg_envelope_loss
            + float(calibration.get("emg_rms_weight", 1.0)) * emg_rms_loss
            + float(calibration.get("emg_generated_rms_weight", 1.0))
            * emg_generated_rms_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            trainable_parameters(model),
            float(config["train"].get("grad_clip", 1.0)),
        )
        optimizer.step()
        batch_size = len(history)
        samples_seen += batch_size
        values = {
            "loss": loss,
            "eeg_spectral_loss": eeg_spectral_loss,
            "eeg_energy_loss": eeg_energy_loss,
            "emg_envelope_loss": emg_envelope_loss,
            "emg_rms_loss": emg_rms_loss,
            "emg_generated_rms_loss": emg_generated_rms_loss,
        }
        for name, value in values.items():
            totals[name] += float(value.detach().cpu()) * batch_size
    return {name: value / samples_seen for name, value in totals.items()}


def main() -> int:
    args = parse_args()
    config = with_normalization(
        with_manifest(load_config(args.config), args.manifest), args.normalization
    )
    config = copy.deepcopy(config)
    if args.seed is not None:
        config["experiment"]["seed"] = int(args.seed)
    if args.epochs is not None:
        config["train"]["epochs"] = int(args.epochs)
    if args.output_dir:
        config["train"]["output_dir"] = str(Path(args.output_dir))
    if args.smoke:
        config["train"]["epochs"] = 1
        config["train"]["num_workers"] = 0
    seed_everything(int(config["experiment"]["seed"]))
    device = resolve_device(str(config["train"].get("device", "cuda")))
    run_dir = prepare_run(config)
    model = build_mainline_model(config)
    initial_checkpoint = load_i1_initialization(
        model, str(config["integration_calibration"]["initial_checkpoint"])
    )
    parameters = trainable_parameters(model)
    if not parameters:
        raise ValueError("missing physiology calibration heads are not enabled")
    model.to(device)
    train_data = physio_feature_sequence_dataset(config, "train")
    val_data = physio_feature_sequence_dataset(config, "val")
    if args.smoke:
        train_data = Subset(train_data, range(min(64, len(train_data))))
        val_data = Subset(val_data, range(min(64, len(val_data))))
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["integration_calibration"].get("learning_rate", 1e-4)),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    modalities = tuple(config["data"]["modalities"])
    selection_subsets = tuple((modality,) for modality in modalities)
    best_key = (float("inf"), float("inf"), float("inf"))
    best_epoch = 0
    curve = []
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_metrics = train_epoch(model, train_data, config, optimizer, device)
        selection_results = evaluate_subsets(
            model, val_data, config, device, selection_subsets
        )
        key = selection_key(selection_results, config)
        curve.append(
            {"epoch": epoch, "train": train_metrics, "selection_key": list(key)}
        )
        print(
            f"epoch={epoch:03d} loss={train_metrics['loss']:.6f} "
            f"amplitude_deficit={key[0]:.6f} single_mae={key[1]:.6f}",
            flush=True,
        )
        if key < best_key:
            best_key = key
            best_epoch = epoch
            save_checkpoint(
                run_dir / "best.pt",
                {
                    "experiment_id": config["experiment"]["id"],
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "config": config,
                    "initial_checkpoint": config["integration_calibration"][
                        "initial_checkpoint"
                    ],
                    "initial_epoch": initial_checkpoint.get("epoch"),
                    "selection_results": selection_results,
                    "test_split_accessed": False,
                },
            )
    selected = load_checkpoint(run_dir / "best.pt")
    model.load_state_dict(selected["model_state"], strict=True)
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
    final_results = evaluate_subsets(
        model,
        val_data,
        config,
        device,
        nonempty_modality_subsets(modalities),
    )
    waveform_checkpoint = load_checkpoint(config["mainline"]["waveform_checkpoint"])
    w5_validation = waveform_checkpoint.get("validation_metrics")
    if not isinstance(w5_validation, dict):
        raise ValueError("W5 checkpoint does not contain validation_metrics")
    component_identity = initial_checkpoint.get("component_identity", {})
    if not bool(component_identity.get("passed", False)):
        raise ValueError("I1 checkpoint lacks a passing component identity record")
    gate = integration_gate(
        component_identity,
        state_identity,
        final_results,
        w5_validation,
        config,
        args.smoke,
    )
    payload = {
        "best_epoch": best_epoch,
        "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
        "initial_checkpoint": config["integration_calibration"]["initial_checkpoint"],
        "component_identity": component_identity,
        "state_identity": state_identity,
        "waveform_results": final_results,
        "w5_reference_validation": w5_validation,
        "gate": gate,
        "training_curve": curve,
        "train_sequences": len(train_data),
        "validation_sequences": len(val_data),
        "smoke": bool(args.smoke),
        "test_split_accessed": False,
    }
    write_json(run_dir / "metrics.json", payload)
    write_json(run_dir / "mainline_missing_calibration_val.json", payload)
    print(json.dumps({"best_epoch": best_epoch, "gate": gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
