from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.utils.data import Subset

from sleepwm.protocols import extract_subset_waveform_cache, integration_gate, state_identity_check
from sleepwm.training.gated_long_rollout import build_model as build_recursive_model
from sleepwm.config import load_config, with_manifest, with_normalization
from sleepwm.engine import data_loader, load_checkpoint, physio_feature_sequence_dataset, prepare_run, resolve_device, save_checkpoint, seed_everything, write_json
from sleepwm.mainline import assemble_mainline_components, build_mainline_model
from sleepwm.waveforms import evaluate_mainline_cache
from sleepwm.masking import nonempty_modality_subsets
from sleepwm.masking import sampled_history_view
from sleepwm.waveforms import probabilistic_waveform_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit zero-initialized missing-modality waveform conditions."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--normalization")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def trainable_parameters(model) -> list[torch.nn.Parameter]:
    return [
        parameter
        for parameter in model.waveform_decoder.missing_condition_parameters()
        if parameter.requires_grad
    ]


def train_epoch(model, dataset, config: dict, optimizer, device: torch.device) -> dict:
    model.eval()
    model.waveform_decoder.missing_context_adapters.train()
    totals = {"loss": 0.0, "waveform_loss": 0.0, "probability_loss": 0.0}
    samples_seen = 0
    waveform = config["waveform"]
    finetune = config["integration_finetune"]
    horizons = tuple(int(value) for value in waveform["horizons_seconds"])
    for batch in data_loader(dataset, config, shuffle=True):
        history = batch["history_signals"].to(device, torch.float32)
        present = batch["history_present"].to(device, torch.bool)
        history, present, _ = sampled_history_view(
            history,
            present,
            float(finetune.get("full_modality_probability", 0.35)),
        )
        context = model.rollout_context(history, present)
        availability = present.to(torch.float32).mean(dim=1)
        prediction, structure, probabilities = model.waveform_decoder(
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
        waveform_losses = model.waveform_decoder.waveform_loss(
            prediction,
            target,
            valid,
            horizons,
            float(waveform.get("time_weight", 1.0)),
            float(waveform.get("spectral_weight", 0.25)),
            float(waveform.get("structure_weight", 0.25)),
            structure,
            tuple(int(value) for value in waveform.get("multi_resolution_fft_sizes", ())),
            float(waveform.get("auxiliary_structure_weight", 0.25)),
        )
        probability_losses = probabilistic_waveform_loss(
            probabilities,
            target,
            valid,
            tuple(config["data"]["modalities"]),
            int(config["data"]["sample_rate"]),
            horizons,
            int(waveform["patch_samples"]),
        )
        loss = waveform_losses["loss"] + float(
            waveform.get("probability_weight", 0.25)
        ) * probability_losses["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            trainable_parameters(model),
            float(config["train"].get("grad_clip", 1.0)),
        )
        optimizer.step()
        batch_size = len(history)
        samples_seen += batch_size
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["waveform_loss"] += (
            float(waveform_losses["loss"].detach().cpu()) * batch_size
        )
        totals["probability_loss"] += (
            float(probability_losses["loss"].detach().cpu()) * batch_size
        )
    return {name: value / samples_seen for name, value in totals.items()}


def evaluate_subsets(model, dataset, config: dict, device, subsets) -> dict:
    results = {}
    for subset in subsets:
        name = "+".join(subset)
        cache = extract_subset_waveform_cache(
            model, dataset, config, device, tuple(subset)
        )
        results[name] = evaluate_mainline_cache(model, cache, config, device)
        del cache
    return results


def selection_key(results: dict, config: dict) -> tuple[float, float, float]:
    minimum = float(config["mainline"].get("minimum_amplitude_ratio", 0.25))
    maximum = float(config["mainline"].get("maximum_amplitude_ratio", 2.5))
    ratios = [
        float(values["generated_to_target_ratio"])
        for metrics in results.values()
        for values in metrics["amplitude"].values()
    ]
    amplitude_deficit = sum(
        max(minimum - value, 0.0) + max(value - maximum, 0.0)
        for value in ratios
    )
    mean_mae = sum(
        float(metrics["model_waveform"]["all"]["mean_standardized_mae"])
        for metrics in results.values()
    ) / len(results)
    mean_probability = sum(
        float(metrics["mean_probability_loss"]) for metrics in results.values()
    ) / len(results)
    return amplitude_deficit, mean_mae, mean_probability


def main() -> int:
    args = parse_args()
    config = with_normalization(
        with_manifest(load_config(args.config), args.manifest), args.normalization
    )
    config = copy.deepcopy(config)
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
    component_paths = {
        name: str(config["mainline"][name])
        for name in (
            "observation_checkpoint",
            "recursive_checkpoint",
            "waveform_checkpoint",
        )
    }
    observation_checkpoint = load_checkpoint(component_paths["observation_checkpoint"])
    recursive_checkpoint = load_checkpoint(component_paths["recursive_checkpoint"])
    waveform_checkpoint = load_checkpoint(component_paths["waveform_checkpoint"])
    model = build_mainline_model(config)
    component_identity = assemble_mainline_components(
        model, observation_checkpoint, recursive_checkpoint, waveform_checkpoint
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.waveform_decoder.missing_condition_parameters():
        parameter.requires_grad = True
    parameters = trainable_parameters(model)
    if not parameters:
        raise ValueError("missing-modality waveform adapters are not enabled")
    model.to(device)
    train_data = physio_feature_sequence_dataset(config, "train")
    val_data = physio_feature_sequence_dataset(config, "val")
    if args.smoke:
        train_data = Subset(train_data, range(min(64, len(train_data))))
        val_data = Subset(val_data, range(min(64, len(val_data))))
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["integration_finetune"].get("learning_rate", 1e-4)),
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
            {
                "epoch": epoch,
                "train": train_metrics,
                "selection_key": list(key),
            }
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
                    "components": component_paths,
                    "component_identity": component_identity,
                    "selection_results": selection_results,
                    "test_split_accessed": False,
                },
            )
    selected = load_checkpoint(run_dir / "best.pt")
    model.load_state_dict(selected["model_state"], strict=True)
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
    w5_validation = waveform_checkpoint.get("validation_metrics")
    if not isinstance(w5_validation, dict):
        raise ValueError("W5 checkpoint does not contain validation_metrics")
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
        "components": component_paths,
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
    write_json(run_dir / "mainline_missing_waveform_val.json", payload)
    print(json.dumps({"best_epoch": best_epoch, "gate": gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
