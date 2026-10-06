from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.utils.data import Subset

from sleepwm.training.probabilistic_waveform_forecast import evaluate
from sleepwm.training.waveform_forecast import build_model, cache_loader, extract_cache
from sleepwm.config import load_config, with_manifest, with_normalization
from sleepwm.engine import load_checkpoint, physio_feature_sequence_dataset, prepare_run, resolve_device, save_checkpoint, seed_everything, write_json
from sleepwm.waveforms import probabilistic_waveform_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an event-conditioned waveform generator without a repeat skip."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--manifest")
    parser.add_argument("--normalization")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_initialization(model, checkpoint_path: str) -> dict:
    checkpoint = load_checkpoint(checkpoint_path)
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "independent waveform initialization mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.waveform_decoder.requires_grad_(True)
    if model.waveform_decoder.output_baseline != "none":
        raise ValueError("independent waveform training requires output_baseline=none")
    return checkpoint


def train_epoch(model, cache: dict, config: dict, optimizer, device) -> dict:
    waveform = config["waveform"]
    model.eval()
    model.waveform_decoder.train()
    totals = {
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
        prediction, structures, probabilities = model.waveform_decoder(
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
            structures,
            tuple(
                int(value)
                for value in waveform.get("multi_resolution_fft_sizes", ())
            ),
            float(waveform.get("auxiliary_structure_weight", 0.25)),
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
            model.waveform_decoder.parameters(),
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
            totals[name] += float(value.detach().cpu()) * batch_size
    return {name: value / samples_seen for name, value in totals.items()}


def independent_gate(metrics: dict, maximum_repeat_ratio: float) -> dict:
    model_mae = float(metrics["model_waveform"]["all"]["mean_standardized_mae"])
    repeat_mae = float(metrics["repeat_last_window"]["all"]["mean_standardized_mae"])
    probability = metrics["probability"]["by_horizon_seconds"]
    qrs_f1 = [float(values["ECG"]["qrs_event_f1"]) for values in probability.values()]
    model_horizons = metrics["model_waveform"]["by_horizon_seconds"]
    repeat_horizons = metrics["repeat_last_window"]["by_horizon_seconds"]
    primary_metrics = {
        "ECG": ("qrs_event_f1", "higher"),
        "EMG": ("envelope_correlation", "higher"),
    }
    improved_horizons = {
        "EEG": [
            horizon
            for horizon, values in probability.items()
            if float(values["EEG"]["spectral_mean_mae"])
            < float(values["EEG"]["baseline_spectral_mae"])
        ]
    }
    for modality, (metric, direction) in primary_metrics.items():
        improved = []
        for horizon, values in model_horizons.items():
            model_value = float(values["by_modality"][modality][metric])
            repeat_value = float(
                repeat_horizons[horizon]["by_modality"][modality][metric]
            )
            if (direction == "lower" and model_value < repeat_value) or (
                direction == "higher" and model_value > repeat_value
            ):
                improved.append(horizon)
        improved_horizons[modality] = improved
    amplitude_ratios = {
        modality: float(values["generated_to_target_ratio"])
        for modality, values in metrics["amplitude"].items()
    }
    result = {
        "no_repeat_skip_active": True,
        "model_waveform_mae": model_mae,
        "repeat_baseline_mae": repeat_mae,
        "waveform_ratio": model_mae / max(repeat_mae, 1e-12),
        "maximum_repeat_ratio": float(maximum_repeat_ratio),
        "waveform_retained": model_mae <= float(maximum_repeat_ratio) * repeat_mae,
        "ecg_event_path_active": max(qrs_f1) > 0.0,
        "physiological_improved_horizons": improved_horizons,
        "all_modalities_physiologically_improved": all(
            len(values) >= 2 for values in improved_horizons.values()
        ),
        "amplitude_ratios": amplitude_ratios,
        "noncollapsed_amplitude": all(
            0.25 <= value <= 2.5 for value in amplitude_ratios.values()
        ),
    }
    result["passed"] = bool(
        result["waveform_retained"]
        and result["ecg_event_path_active"]
        and result["all_modalities_physiologically_improved"]
        and result["noncollapsed_amplitude"]
    )
    return result


def main() -> int:
    args = parse_args()
    config = with_normalization(
        with_manifest(load_config(args.config), args.manifest), args.normalization
    )
    config = copy.deepcopy(config)
    if args.checkpoint:
        config["waveform"]["initial_waveform_checkpoint"] = args.checkpoint
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
    model = build_model(config).to(device)
    initialization = load_initialization(
        model, str(config["waveform"]["initial_waveform_checkpoint"])
    )
    train_data = physio_feature_sequence_dataset(config, "train")
    val_data = physio_feature_sequence_dataset(config, "val")
    if args.smoke:
        train_data = Subset(train_data, range(min(64, len(train_data))))
        val_data = Subset(val_data, range(min(64, len(val_data))))
    print("extracting frozen world-model train contexts", flush=True)
    train_cache = extract_cache(model, train_data, config, device)
    print("extracting frozen world-model validation contexts", flush=True)
    val_cache = extract_cache(model, val_data, config, device)
    optimizer = torch.optim.AdamW(
        model.waveform_decoder.parameters(),
        lr=float(config["waveform"].get("learning_rate", 1e-4)),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    best_score = float("inf")
    best_epoch = 0
    best_metrics = None
    training_curve = []
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_metrics = train_epoch(model, train_cache, config, optimizer, device)
        validation = evaluate(model, val_cache, config, device)
        score = float(
            validation["model_waveform"]["all"]["mean_standardized_mae"]
        )
        training_curve.append(
            {"epoch": epoch, "train": train_metrics, "validation_mae": score}
        )
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.6f} "
            f"waveform_mae={score:.6f}",
            flush=True,
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = validation
            save_checkpoint(
                run_dir / "best.pt",
                {
                    "experiment_id": config["experiment"]["id"],
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "validation_metrics": validation,
                    "config": config,
                    "repeat_skip_used": False,
                    "initial_checkpoint": config["waveform"][
                        "initial_waveform_checkpoint"
                    ],
                    "initial_epoch": initialization.get("epoch"),
                },
            )
    if best_metrics is None:
        raise RuntimeError("no independent waveform checkpoint was selected")
    gate = independent_gate(
        best_metrics,
        float(config["waveform"].get("maximum_repeat_ratio", 1.15)),
    )
    payload = {
        "best_epoch": best_epoch,
        "best_validation": best_metrics,
        "gate": gate,
        "training_curve": training_curve,
        "repeat_skip_used": False,
        "initial_checkpoint": config["waveform"]["initial_waveform_checkpoint"],
        "train_sequences": len(train_data),
        "validation_sequences": len(val_data),
        "test_split_accessed": False,
    }
    write_json(run_dir / "metrics.json", payload)
    write_json(run_dir / "independent_waveform_val.json", payload)
    print(json.dumps({"best_epoch": best_epoch, "gate": gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
