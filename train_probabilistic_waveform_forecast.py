from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from train_waveform_forecast import build_model, cache_loader, extract_cache
from uniphysio_wm.config import load_config, with_manifest, with_normalization
from uniphysio_wm.engine import (
    load_checkpoint,
    physio_feature_sequence_dataset,
    prepare_run,
    resolve_device,
    save_checkpoint,
    seed_everything,
    write_json,
)
from uniphysio_wm.probabilistic_waveform import (
    probabilistic_waveform_gate_result,
    probabilistic_waveform_loss,
    probabilistic_waveform_metrics,
    refractory_ecg_gate_result,
)
from uniphysio_wm.waveform_metrics import waveform_forecast_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train probabilistic events and conditioned waveform residuals."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--normalization")
    parser.add_argument("--checkpoint")
    parser.add_argument("--feature-manifest")
    parser.add_argument("--feature-statistics")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def trainable_probability_modules(model) -> tuple:
    decoder = model.waveform_decoder
    if decoder.ecg_refractory_event_enabled:
        return (
            decoder.ecg_point_process_head,
            decoder.ecg_point_process_adapter,
            decoder.ecg_point_process_output_head,
        )
    return (
        decoder.probability_heads,
        decoder.probability_adapters,
        decoder.event_output_heads,
    )


def load_probability_initialization(model, checkpoint_path: str) -> dict:
    checkpoint = load_checkpoint(checkpoint_path)
    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("F5-W2 checkpoint does not contain model_state")
    incompatible = model.load_state_dict(state, strict=False)
    if model.waveform_decoder.ecg_refractory_event_enabled:
        allowed_prefixes = (
            "waveform_decoder.ecg_point_process_head.",
            "waveform_decoder.ecg_point_process_adapter.",
            "waveform_decoder.ecg_point_process_output_head.",
        )
    else:
        allowed_prefixes = (
            "waveform_decoder.probability_heads.",
            "waveform_decoder.probability_adapters.",
            "waveform_decoder.event_output_heads.",
        )
    unexpected_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(allowed_prefixes)
    ]
    if incompatible.unexpected_keys or unexpected_missing:
        raise ValueError(
            f"probability initialization mismatch: missing={unexpected_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in trainable_probability_modules(model):
        if module is None:
            raise RuntimeError("configured probability module was not initialized")
        for parameter in module.parameters():
            parameter.requires_grad = True
    return checkpoint


def predict_cache(model, cache: dict, batch_size: int, device: torch.device) -> tuple:
    model.waveform_decoder.eval()
    waveforms = []
    accumulated = {}
    with torch.no_grad():
        for shared, dynamics, recent, _, _ in cache_loader(
            cache, batch_size, False, 0
        ):
            prediction, _, probabilities = model.waveform_decoder(
                recent.to(device=device, dtype=torch.float32),
                shared.to(device=device, dtype=torch.float32),
                dynamics.to(device=device, dtype=torch.float32),
                return_structure=True,
                return_probabilities=True,
            )
            waveforms.append(prediction.cpu())
            for modality, values in probabilities.items():
                accumulated.setdefault(modality, {})
                for name, value in values.items():
                    accumulated[modality].setdefault(name, []).append(value.cpu())
    probabilities = {
        modality: {
            name: torch.cat(values) for name, values in modality_values.items()
        }
        for modality, modality_values in accumulated.items()
    }
    return torch.cat(waveforms), probabilities


def train_epoch(model, cache: dict, config: dict, optimizer, device: torch.device) -> dict:
    waveform = config["waveform"]
    model.eval()
    for module in trainable_probability_modules(model):
        if module is None:
            raise RuntimeError("configured probability module was not initialized")
        module.train()
    total = {
        "loss": 0.0,
        "waveform_loss": 0.0,
        "probability_loss": 0.0,
        "eeg_probability_loss": 0.0,
        "ecg_probability_loss": 0.0,
        "emg_probability_loss": 0.0,
    }
    samples_seen = 0
    loader = cache_loader(
        cache,
        int(config["train"]["batch_size"]),
        True,
        int(config["experiment"]["seed"]),
    )
    for shared, dynamics, recent, target, valid in loader:
        shared = shared.to(device=device, dtype=torch.float32)
        dynamics = dynamics.to(device=device, dtype=torch.float32)
        recent = recent.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.float32)
        valid = valid.to(device=device, dtype=torch.bool)
        prediction, _, probabilities = model.waveform_decoder(
            recent,
            shared,
            dynamics,
            return_structure=True,
            return_probabilities=True,
        )
        waveform_losses = model.waveform_decoder.waveform_loss(
            prediction,
            target,
            valid,
            waveform["horizons_seconds"],
            float(waveform.get("time_weight", 1.0)),
            float(waveform.get("spectral_weight", 0.25)),
            float(waveform.get("structure_weight", 0.25)),
            None,
            tuple(
                int(value)
                for value in waveform.get("multi_resolution_fft_sizes", ())
            ),
            0.0,
        )
        probability_losses = probabilistic_waveform_loss(
            probabilities,
            target,
            valid,
            tuple(config["data"]["modalities"]),
            int(config["data"]["sample_rate"]),
            tuple(int(value) for value in waveform["horizons_seconds"]),
            int(waveform["patch_samples"]),
        )
        loss = waveform_losses["loss"] + float(
            waveform.get("probability_weight", 0.25)
        ) * probability_losses["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            float(config["train"].get("grad_clip", 1.0)),
        )
        optimizer.step()
        batch_size = len(shared)
        samples_seen += batch_size
        values = {
            "loss": loss,
            "waveform_loss": waveform_losses["loss"],
            "probability_loss": probability_losses["loss"],
            "eeg_probability_loss": probability_losses["eeg_loss"],
            "ecg_probability_loss": probability_losses["ecg_loss"],
            "emg_probability_loss": probability_losses["emg_loss"],
        }
        for name, value in values.items():
            total[name] += float(value.detach().cpu()) * batch_size
    return {name: value / samples_seen for name, value in total.items()}


def evaluate(model, cache: dict, config: dict, device: torch.device) -> dict:
    waveform = config["waveform"]
    prediction, probabilities = predict_cache(
        model, cache, int(config["train"]["batch_size"]), device
    )
    target = cache["target_waveform"].float()
    recent = cache["recent_waveform"].float()
    valid = cache["valid"]
    modalities = tuple(config["data"]["modalities"])
    sample_rate = int(config["data"]["sample_rate"])
    horizons = tuple(int(value) for value in waveform["horizons_seconds"])
    probability_losses = probabilistic_waveform_loss(
        probabilities,
        target,
        valid,
        modalities,
        sample_rate,
        horizons,
        int(waveform["patch_samples"]),
    )
    amplitude = {}
    for modality_index, modality in enumerate(modalities):
        selected = valid[:, modality_index]
        generated_rms = prediction[selected, modality_index].square().mean(-1).sqrt().mean()
        target_rms = target[selected, modality_index].square().mean(-1).sqrt().mean()
        amplitude[modality] = {
            "generated_rms": float(generated_rms),
            "target_rms": float(target_rms),
            "generated_to_target_ratio": float(
                generated_rms / target_rms.clamp_min(1e-8)
            ),
        }
    return {
        "model_waveform": waveform_forecast_metrics(
            prediction, target, valid, modalities, sample_rate, horizons
        ),
        "repeat_last_window": waveform_forecast_metrics(
            recent, target, valid, modalities, sample_rate, horizons
        ),
        "probability": probabilistic_waveform_metrics(
            probabilities,
            target,
            recent,
            valid,
            modalities,
            sample_rate,
            horizons,
            int(waveform["patch_samples"]),
        ),
        "mean_probability_loss": float(probability_losses["loss"]),
        "amplitude": amplitude,
    }


def main() -> int:
    args = parse_args()
    config = with_normalization(
        with_manifest(load_config(args.config), args.manifest), args.normalization
    )
    config = copy.deepcopy(config)
    if args.checkpoint:
        config["waveform"]["initial_waveform_checkpoint"] = args.checkpoint
    if args.feature_manifest:
        config["physiology"]["feature_manifest_path"] = args.feature_manifest
    if args.feature_statistics:
        config["physiology"]["feature_statistics_path"] = args.feature_statistics
    if args.seed is not None:
        config["experiment"]["seed"] = int(args.seed)
    if args.epochs is not None:
        config["train"]["epochs"] = int(args.epochs)
    if args.output_dir:
        config["train"]["output_dir"] = str(Path(args.output_dir))
    if args.smoke:
        config["train"]["epochs"] = 1
        config["train"]["num_workers"] = 0
    if not bool(config["waveform"].get("probabilistic_event_heads", False)):
        raise ValueError("probabilistic waveform heads are not enabled")
    seed_everything(int(config["experiment"]["seed"]))
    device = resolve_device(str(config["train"].get("device", "cuda")))
    run_dir = prepare_run(config)
    model = build_model(config).to(device)
    initialization = load_probability_initialization(
        model, config["waveform"]["initial_waveform_checkpoint"]
    )
    train_dataset = physio_feature_sequence_dataset(config, "train")
    val_dataset = physio_feature_sequence_dataset(config, "val")
    print("extracting frozen F5 train contexts")
    train_cache = extract_cache(model, train_dataset, config, device)
    print("extracting frozen F5 validation contexts")
    val_cache = extract_cache(model, val_dataset, config, device)
    print(
        f"cached train={len(train_dataset)} validation={len(val_dataset)} "
        f"waveform_samples={model.waveform_decoder.max_samples}"
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["waveform"].get("learning_rate", 1e-4)),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    best_key = (2, float("inf"))
    best_epoch = 0
    best_metrics = None
    training_curve = []
    relative_improvement = float(
        config["waveform"].get("relative_mae_improvement", 0.02)
    )
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_losses = train_epoch(model, train_cache, config, optimizer, device)
        val_metrics = evaluate(model, val_cache, config, device)
        waveform_mae = float(
            val_metrics["model_waveform"]["all"]["mean_standardized_mae"]
        )
        baseline_mae = float(
            val_metrics["repeat_last_window"]["all"]["mean_standardized_mae"]
        )
        eligible = waveform_mae < baseline_mae * (1.0 - relative_improvement)
        selection_key = (
            0 if eligible else 1,
            float(val_metrics["mean_probability_loss"]),
        )
        training_curve.append(
            {
                "epoch": epoch,
                "train": train_losses,
                "val_waveform_mae": waveform_mae,
                "val_probability_loss": val_metrics["mean_probability_loss"],
                "waveform_gate_eligible": eligible,
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={train_losses['loss']:.6f} "
            f"val_waveform_mae={waveform_mae:.6f} "
            f"val_probability_loss={val_metrics['mean_probability_loss']:.6f}"
        )
        if selection_key < best_key:
            best_key = selection_key
            best_epoch = epoch
            best_metrics = val_metrics
            save_checkpoint(
                run_dir / "best.pt",
                {
                    "experiment_id": config["experiment"]["id"],
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "validation_metrics": val_metrics,
                    "config": config,
                    "f5_and_f5w2_frozen": True,
                    "initial_waveform_checkpoint": config["waveform"][
                        "initial_waveform_checkpoint"
                    ],
                    "initial_waveform_epoch": initialization.get("epoch"),
                },
            )
    if best_metrics is None:
        raise RuntimeError("no probabilistic waveform checkpoint was selected")
    gate_function = (
        refractory_ecg_gate_result
        if model.waveform_decoder.ecg_refractory_event_enabled
        else probabilistic_waveform_gate_result
    )
    gate = gate_function(
        best_metrics["model_waveform"],
        best_metrics["repeat_last_window"],
        best_metrics["probability"],
        relative_improvement,
    )
    payload = {
        "best_epoch": best_epoch,
        "best_validation": best_metrics,
        "probabilistic_gate": gate,
        "training_curve": training_curve,
        "f5_and_f5w2_frozen": True,
        "initial_waveform_checkpoint": config["waveform"][
            "initial_waveform_checkpoint"
        ],
        "train_sequences": len(train_dataset),
        "validation_sequences": len(val_dataset),
        "test_split_accessed": False,
    }
    write_json(run_dir / "metrics.json", payload)
    write_json(
        run_dir / "evaluation_val.json",
        {
            "checkpoint": str(run_dir / "best.pt"),
            "split": "val",
            "results": best_metrics,
            "probabilistic_gate": gate,
            "f5_and_f5w2_frozen": True,
        },
    )
    print(json.dumps({"best_epoch": best_epoch, "gate": gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
