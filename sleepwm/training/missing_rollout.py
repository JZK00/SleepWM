from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from sleepwm.protocols import evaluate_subset
from sleepwm.training.waveform_forecast import build_model
from sleepwm.config import load_config, with_manifest, with_normalization
from sleepwm.data import balanced_class_weights
from sleepwm.engine import data_loader, load_checkpoint, physio_feature_sequence_dataset, prepare_run, resolve_device, save_checkpoint, seed_everything, write_json
from sleepwm.masking import nonempty_modality_subsets
from sleepwm.masking import missing_rollout_gate_result, missing_rollout_summary, sampled_history_view
from sleepwm.waveforms import probabilistic_waveform_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune causal rollout with full-biased missing history modalities."
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
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if model.target_encoder is not None:
        for parameter in model.target_encoder.parameters():
            parameter.requires_grad = False
    return checkpoint


def build_optimizer(model, config: dict) -> torch.optim.Optimizer:
    waveform_ids = {
        id(parameter) for parameter in model.waveform_decoder.parameters()
    }
    waveform_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) in waveform_ids
    ]
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in waveform_ids
    ]
    return torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": float(config["stage7"].get("backbone_learning_rate", 1e-5)),
            },
            {
                "params": waveform_parameters,
                "lr": float(config["stage7"].get("waveform_learning_rate", 5e-5)),
            },
        ],
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )


def train_epoch(model, dataset, config: dict, optimizer, class_weights, device) -> dict:
    model.train()
    totals = {
        "loss": 0.0,
        "world_model_loss": 0.0,
        "probability_loss": 0.0,
        "latent_loss": 0.0,
        "stage_loss": 0.0,
        "history_physiology_loss": 0.0,
        "future_physiology_loss": 0.0,
        "future_waveform_loss": 0.0,
    }
    samples_seen = 0
    waveform = config["waveform"]
    forecast = config["forecast"]
    physiology = config["physiology"]
    for batch in data_loader(dataset, config, shuffle=True):
        history_signals = batch["history_signals"].to(
            device=device, dtype=torch.float32
        )
        history_present = batch["history_present"].to(
            device=device, dtype=torch.bool
        )
        history_signals, history_present, _ = sampled_history_view(
            history_signals,
            history_present,
            float(config["stage7"].get("full_modality_probability", 0.35)),
        )
        future_signals = batch["future_signals"].to(
            device=device, dtype=torch.float32
        )
        future_present = batch["future_present"].to(
            device=device, dtype=torch.bool
        )
        future_labels = batch["future_labels"].to(device=device, dtype=torch.long)
        history_labels = batch["history_labels"].to(device=device, dtype=torch.long)
        waveform_target = future_signals[
            :, 0, :, : model.waveform_decoder.max_samples
        ]
        waveform_valid = future_present[:, 0]
        output = model(
            history_signals,
            history_present,
            future_signals,
            future_present,
            future_labels,
            history_labels=history_labels,
            latent_weight=float(forecast.get("latent_weight", 1.0)),
            stage_weight=float(forecast.get("stage_weight", 1.0)),
            current_stage_weight=float(forecast.get("current_stage_weight", 1.0)),
            stage_class_weights=class_weights,
            history_physiology_targets=batch["history_physiology"].to(
                device=device, dtype=torch.float32
            ),
            history_physiology_valid=batch["history_physiology_valid"].to(
                device=device, dtype=torch.bool
            ),
            future_physiology_targets=batch["future_physiology"].to(
                device=device, dtype=torch.float32
            ),
            future_physiology_valid=batch["future_physiology_valid"].to(
                device=device, dtype=torch.bool
            ),
            history_physiology_weight=float(physiology.get("history_weight", 1.0)),
            future_physiology_weight=float(physiology.get("future_weight", 1.0)),
            future_waveform_targets=waveform_target,
            future_waveform_valid=waveform_valid,
            waveform_horizons_seconds=tuple(waveform["horizons_seconds"]),
            waveform_weight=float(waveform.get("time_weight", 1.0)),
            waveform_time_weight=float(waveform.get("time_weight", 1.0)),
            waveform_spectral_weight=float(waveform.get("spectral_weight", 0.25)),
            waveform_structure_weight=float(waveform.get("structure_weight", 0.25)),
            waveform_multi_resolution_fft_sizes=tuple(
                int(value) for value in waveform.get("multi_resolution_fft_sizes", ())
            ),
        )
        probability = probabilistic_waveform_loss(
            output["future_waveform_probabilities"],
            waveform_target,
            waveform_valid,
            tuple(config["data"]["modalities"]),
            int(config["data"]["sample_rate"]),
            tuple(int(value) for value in waveform["horizons_seconds"]),
            int(waveform["patch_samples"]),
        )
        world_model_loss = output["loss"]
        loss = world_model_loss + float(waveform.get("probability_weight", 0.25)) * probability["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            float(config["train"].get("grad_clip", 1.0)),
        )
        optimizer.step()
        batch_size = len(history_signals)
        samples_seen += batch_size
        values = {
            "loss": loss,
            "world_model_loss": world_model_loss,
            "probability_loss": probability["loss"],
            "latent_loss": output["latent_loss"],
            "stage_loss": output["stage_loss"],
            "history_physiology_loss": output["history_physiology_loss"],
            "future_physiology_loss": output["future_physiology_loss"],
            "future_waveform_loss": output["future_waveform_loss"],
        }
        for name, value in values.items():
            totals[name] += float(value.detach().cpu()) * batch_size
    return {name: value / samples_seen for name, value in totals.items()}


def evaluate_matrix(model, dataset, config: dict, device) -> tuple[dict, dict, dict]:
    modalities = tuple(config["data"]["modalities"])
    results = {
        "+".join(subset): evaluate_subset(model, dataset, config, device, subset)
        for subset in nonempty_modality_subsets(modalities)
    }
    summary = missing_rollout_summary(results, modalities)
    stage7 = config["stage7"]
    gate = missing_rollout_gate_result(
        summary,
        float(stage7.get("mean_stage_drop_limit", 0.10)),
        float(stage7.get("worst_stage_drop_limit", 0.20)),
        float(stage7.get("physiology_mae_ratio_limit", 1.25)),
        float(stage7.get("waveform_mae_ratio_limit", 1.25)),
        int(stage7.get("uncertainty_modalities_required", 2)),
    )
    return results, summary, gate


def main() -> int:
    args = parse_args()
    config = with_normalization(
        with_manifest(load_config(args.config), args.manifest), args.normalization
    )
    config = copy.deepcopy(config)
    if args.checkpoint:
        config["stage7"]["checkpoint_path"] = args.checkpoint
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
    initialization = load_initialization(model, config["stage7"]["checkpoint_path"])
    train_data = physio_feature_sequence_dataset(config, "train")
    val_data = physio_feature_sequence_dataset(config, "val")
    class_weights = balanced_class_weights(train_data.future_label_counts).to(device)
    optimizer = build_optimizer(model, config)
    best_key = (-1, float("-inf"))
    best_epoch = 0
    best_results = None
    best_summary = None
    best_gate = None
    training_curve = []
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_metrics = train_epoch(
            model, train_data, config, optimizer, class_weights, device
        )
        results, summary, gate = evaluate_matrix(model, val_data, config, device)
        full_stage = float(summary["full_modality"]["stage_macro_f1"])
        mean_nonfull_stage = full_stage - float(
            summary["nonfull_mean_stage_macro_f1_drop"]
        )
        eligible = full_stage >= float(
            config["stage7"].get("full_stage_retention_floor", 0.5941)
        )
        selection_key = (1 if eligible else 0, mean_nonfull_stage)
        training_curve.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "full_stage_macro_f1": full_stage,
                "mean_nonfull_stage_macro_f1": mean_nonfull_stage,
                "full_stage_eligible": eligible,
                "gate": gate,
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.6f} "
            f"full_stage={full_stage:.4f} missing_mean_stage={mean_nonfull_stage:.4f} "
            f"gate={gate['passed']}",
            flush=True,
        )
        if selection_key > best_key:
            best_key = selection_key
            best_epoch = epoch
            best_results = results
            best_summary = summary
            best_gate = gate
            save_checkpoint(
                run_dir / "best.pt",
                {
                    "experiment_id": config["experiment"]["id"],
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "validation_summary": summary,
                    "validation_gate": gate,
                    "config": config,
                    "initial_checkpoint": config["stage7"]["checkpoint_path"],
                    "initial_epoch": initialization.get("epoch"),
                },
            )
    if best_results is None or best_summary is None or best_gate is None:
        raise RuntimeError("no Stage 7-M1 checkpoint was selected")
    payload = {
        "best_epoch": best_epoch,
        "results": best_results,
        "summary": best_summary,
        "gate": best_gate,
        "training_curve": training_curve,
        "initial_checkpoint": config["stage7"]["checkpoint_path"],
        "full_modality_probability": float(
            config["stage7"].get("full_modality_probability", 0.35)
        ),
        "train_sequences": len(train_data),
        "validation_sequences": len(val_data),
        "test_split_accessed": False,
    }
    write_json(run_dir / "metrics.json", payload)
    write_json(run_dir / "missing_rollout_val.json", payload)
    print(json.dumps({"best_epoch": best_epoch, "gate": best_gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
