from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from sleepwm.config import load_config, with_manifest, with_normalization
from sleepwm.engine import data_loader, load_checkpoint, physio_feature_sequence_dataset, prepare_run, resolve_device, save_checkpoint, seed_everything, write_json
from sleepwm.models import PhysiologyFrontendEncoder, ShortHorizonWaveformWorldModel, observation_config
from sleepwm.metrics import structured_waveform_gate_result, waveform_forecast_metrics, waveform_gate_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train frozen-F5 short-horizon multimodal waveform decoders."
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


def build_model(config: dict) -> ShortHorizonWaveformWorldModel:
    physiology = config["physiology"]
    waveform = config["waveform"]
    group_sizes = {
        group: len(names) for group, names in physiology["feature_groups"].items()
    }
    return ShortHorizonWaveformWorldModel(
        PhysiologyFrontendEncoder(observation_config(config["data"], config["model"])),
        config["data"]["future_horizons"],
        num_classes=int(config["data"].get("num_classes", 5)),
        transition_layers=int(config["model"].get("transition_layers", 2)),
        transition_heads=int(config["model"].get("transition_heads", 4)),
        dropout=float(config["model"].get("dropout", 0.1)),
        freeze_observation_encoder=bool(
            config["forecast"].get("freeze_observation_encoder", False)
        ),
        stage_residual_from_current=bool(
            config["forecast"].get("stage_residual_from_current", False)
        ),
        use_frozen_target_encoder=bool(
            config["forecast"].get("use_frozen_target_encoder", False)
        ),
        factorized_transition_head=bool(
            config["forecast"].get("factorized_transition_head", False)
        ),
        change_prior_probabilities=config["forecast"].get(
            "change_prior_probabilities"
        ),
        physiology_group_sizes=group_sizes,
        physiology_hidden_dim=int(physiology.get("hidden_dim", 128)),
        physiology_dynamics_dim=int(physiology.get("physiology_dynamics_dim", 64)),
        physiology_dynamics_layers=int(
            physiology.get("physiology_dynamics_layers", 1)
        ),
        physiology_dynamics_heads=int(
            physiology.get("physiology_dynamics_heads", 4)
        ),
        waveform_seconds=int(waveform.get("max_seconds", 10)),
        waveform_patch_samples=int(waveform.get("patch_samples", 64)),
        waveform_decoder_dim=int(waveform.get("decoder_dim", 64)),
        waveform_decoder_layers=int(waveform.get("decoder_layers", 1)),
        waveform_decoder_heads=int(waveform.get("decoder_heads", 4)),
        waveform_structured_event_heads=bool(
            waveform.get("structured_event_heads", False)
        ),
        waveform_probabilistic_event_heads=bool(
            waveform.get("probabilistic_event_heads", False)
        ),
        waveform_ecg_refractory_event_head=bool(
            waveform.get("ecg_refractory_event_head", False)
        ),
        waveform_ecg_rr_bins=int(waveform.get("ecg_rr_bins", 48)),
        waveform_output_baseline=str(waveform.get("output_baseline", "repeat")),
        waveform_physiological_event_renderer=bool(
            waveform.get("physiological_event_renderer", False)
        ),
        waveform_ecg_time_aligned_renderer=bool(
            waveform.get("ecg_time_aligned_renderer", False)
        ),
        waveform_ecg_recursive_event_renderer=bool(
            waveform.get("ecg_recursive_event_renderer", False)
        ),
        waveform_ecg_recent_rr_residual=bool(
            waveform.get("ecg_recent_rr_residual", False)
        ),
        waveform_ecg_recent_amplitude_calibration=bool(
            waveform.get("ecg_recent_amplitude_calibration", False)
        ),
        waveform_safe_modality_residual_refinement=bool(
            waveform.get("safe_modality_residual_refinement", False)
        ),
        waveform_missing_modality_conditioning=bool(
            waveform.get("missing_modality_conditioning", False)
        ),
        waveform_missing_physiology_calibration=bool(
            waveform.get("missing_physiology_calibration", False)
        ),
        waveform_missing_emg_teacher_calibration=bool(
            waveform.get("missing_emg_teacher_calibration", False)
        ),
        waveform_ecg_event_sigma_seconds=float(
            waveform.get("ecg_event_sigma_seconds", 0.02)
        ),
    )


def load_f5_initialization(model, checkpoint_path: str) -> dict:
    checkpoint = load_checkpoint(checkpoint_path)
    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("F5 checkpoint does not contain model_state")
    incompatible = model.load_state_dict(state, strict=False)
    unexpected_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("waveform_decoder.")
    ]
    if incompatible.unexpected_keys or unexpected_missing:
        raise ValueError(
            f"F5 waveform initialization mismatch: missing={unexpected_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.waveform_decoder.parameters():
        parameter.requires_grad = True
    return checkpoint


def extract_cache(model, dataset, config, device: torch.device) -> dict:
    model.eval().to(device)
    maximum_samples = model.waveform_decoder.max_samples
    cached = {
        "shared_state": [],
        "dynamics_state": [],
        "recent_waveform": [],
        "target_waveform": [],
        "valid": [],
    }
    with torch.no_grad():
        for batch in data_loader(dataset, config, shuffle=False):
            history_signals = batch["history_signals"].to(
                device=device, dtype=torch.float32
            )
            history_present = batch["history_present"].to(
                device=device, dtype=torch.bool
            )
            output = model.rollout_context(history_signals, history_present)
            cached["shared_state"].append(output["predicted_states"][:, 0].cpu())
            cached["dynamics_state"].append(
                output["physiology_dynamics_states"][:, 0].cpu()
            )
            cached["recent_waveform"].append(
                history_signals[:, -1, :, -maximum_samples:].half().cpu()
            )
            cached["target_waveform"].append(
                batch["future_signals"][:, 0, :, :maximum_samples].half()
            )
            cached["valid"].append(batch["future_present"][:, 0].bool())
    return {key: torch.cat(values) for key, values in cached.items()}


def cache_loader(cache: dict, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        TensorDataset(
            cache["shared_state"],
            cache["dynamics_state"],
            cache["recent_waveform"],
            cache["target_waveform"],
            cache["valid"],
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def predict_cache(model, cache: dict, batch_size: int, device: torch.device) -> torch.Tensor:
    model.waveform_decoder.eval()
    predictions = []
    with torch.no_grad():
        for shared, dynamics, recent, _, _ in cache_loader(
            cache, batch_size, False, 0
        ):
            predictions.append(
                model.waveform_decoder(
                    recent.to(device=device, dtype=torch.float32),
                    shared.to(device=device, dtype=torch.float32),
                    dynamics.to(device=device, dtype=torch.float32),
                ).cpu()
            )
    return torch.cat(predictions)


def train_epoch(model, cache: dict, config: dict, optimizer, device: torch.device) -> dict:
    waveform = config["waveform"]
    model.eval()
    model.waveform_decoder.train()
    total = {
        "loss": 0.0,
        "time_loss": 0.0,
        "spectral_loss": 0.0,
        "structure_loss": 0.0,
        "auxiliary_loss": 0.0,
    }
    samples = 0
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
        structured = bool(waveform.get("structured_event_heads", False))
        decoded = model.waveform_decoder(
            recent, shared, dynamics, return_structure=structured
        )
        if structured:
            prediction, structure_predictions = decoded
        else:
            prediction = decoded
            structure_predictions = None
        losses = model.waveform_decoder.waveform_loss(
            prediction,
            target,
            valid,
            waveform["horizons_seconds"],
            float(waveform.get("time_weight", 1.0)),
            float(waveform.get("spectral_weight", 0.25)),
            float(waveform.get("structure_weight", 0.25)),
            structure_predictions,
            tuple(int(value) for value in waveform.get("multi_resolution_fft_sizes", ())),
            float(waveform.get("auxiliary_structure_weight", 0.0)),
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.waveform_decoder.parameters(), 1.0)
        optimizer.step()
        batch_size = len(shared)
        samples += batch_size
        for key in total:
            total[key] += float(losses[key].detach().cpu()) * batch_size
    return {key: value / samples for key, value in total.items()}


def evaluate(model, cache: dict, config: dict, device: torch.device) -> dict:
    prediction = predict_cache(
        model, cache, int(config["train"]["batch_size"]), device
    )
    modalities = tuple(config["data"]["modalities"])
    sample_rate = int(config["data"]["sample_rate"])
    horizons = tuple(int(value) for value in config["waveform"]["horizons_seconds"])
    target = cache["target_waveform"].float()
    valid = cache["valid"]
    return {
        "model": waveform_forecast_metrics(
            prediction, target, valid, modalities, sample_rate, horizons
        ),
        "repeat_last_window": waveform_forecast_metrics(
            cache["recent_waveform"].float(),
            target,
            valid,
            modalities,
            sample_rate,
            horizons,
        ),
    }


def main() -> int:
    args = parse_args()
    config = with_normalization(
        with_manifest(load_config(args.config), args.manifest), args.normalization
    )
    config = copy.deepcopy(config)
    if args.checkpoint:
        config["waveform"]["initial_world_model_checkpoint"] = args.checkpoint
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
    seed_everything(int(config["experiment"]["seed"]))
    device = resolve_device(str(config["train"].get("device", "cuda")))
    run_dir = prepare_run(config)
    model = build_model(config).to(device)
    initialization = load_f5_initialization(
        model, config["waveform"]["initial_world_model_checkpoint"]
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
        train_losses = train_epoch(model, train_cache, config, optimizer, device)
        val_metrics = evaluate(model, val_cache, config, device)
        score = float(val_metrics["model"]["all"]["mean_standardized_mae"])
        training_curve.append(
            {"epoch": epoch, "train": train_losses, "val_waveform_mae": score}
        )
        print(
            f"epoch={epoch:03d} train_loss={train_losses['loss']:.6f} "
            f"val_waveform_mae={score:.6f}"
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = val_metrics
            save_checkpoint(
                run_dir / "best.pt",
                {
                    "experiment_id": config["experiment"]["id"],
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "waveform_decoder_state": model.waveform_decoder.state_dict(),
                    "validation_metrics": val_metrics,
                    "config": config,
                    "f5_frozen": True,
                    "initial_world_model_checkpoint": config["waveform"][
                        "initial_world_model_checkpoint"
                    ],
                    "initial_world_model_epoch": initialization.get("epoch"),
                },
            )
    if best_metrics is None:
        raise RuntimeError("no waveform decoder checkpoint was selected")
    gate_function = (
        structured_waveform_gate_result
        if bool(config["waveform"].get("structured_event_heads", False))
        else waveform_gate_result
    )
    gate = gate_function(
        best_metrics["model"],
        best_metrics["repeat_last_window"],
        float(config["waveform"].get("relative_mae_improvement", 0.02)),
    )
    payload = {
        "best_epoch": best_epoch,
        "best_validation": best_metrics,
        "waveform_gate": gate,
        "training_curve": training_curve,
        "f5_frozen": True,
        "inherited_f5_validation": initialization.get("validation_metrics"),
        "initial_world_model_checkpoint": config["waveform"][
            "initial_world_model_checkpoint"
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
            "waveform_gate": gate,
            "f5_frozen": True,
        },
    )
    print(json.dumps({"best_epoch": best_epoch, "gate": gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
