from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.nn.functional as F

from uniphysio_wm.config import load_config, with_manifest, with_normalization
from uniphysio_wm.data import balanced_class_weights
from uniphysio_wm.engine import (
    AverageMeter,
    data_loader,
    load_checkpoint,
    load_encoder_state,
    prepare_run,
    resolve_device,
    save_checkpoint,
    seed_everything,
    sequence_dataset,
    trainable_parameters,
    write_json,
)
from uniphysio_wm.metrics import (
    classification_metrics,
    factorized_transition_metrics,
    forecast_subgroup_metrics,
)
from uniphysio_wm.models import CausalPhysioWorldModel, MultiModalEncoder, observation_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a history-only latent physiological transition model.")
    parser.add_argument("--config", required=True, help="forecast YAML configuration")
    parser.add_argument("--manifest", help="override data.manifest_path")
    parser.add_argument("--normalization", help="train-split normalization JSON")
    parser.add_argument("--checkpoint", help="override forecast.pretrained_checkpoint")
    parser.add_argument("--random-init", action="store_true", help="do not load a pretrained encoder")
    parser.add_argument("--seed", type=int, help="override experiment seed")
    parser.add_argument("--epochs", type=int, help="override training epochs")
    parser.add_argument("--output-dir", help="override train.output_dir")
    parser.add_argument("--smoke", action="store_true", help="run one epoch with no worker processes")
    return parser.parse_args()


def build_optimizer(model, config):
    base_learning_rate = float(config["train"]["learning_rate"])
    encoder_learning_rate = float(config["forecast"].get("encoder_learning_rate", base_learning_rate))
    encoder_parameters = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    encoder_parameter_ids = {id(parameter) for parameter in encoder_parameters}
    other_parameters = [
        parameter
        for parameter in trainable_parameters(model)
        if id(parameter) not in encoder_parameter_ids
    ]
    parameter_groups = []
    if other_parameters:
        parameter_groups.append({"params": other_parameters, "lr": base_learning_rate})
    if encoder_parameters:
        parameter_groups.append({"params": encoder_parameters, "lr": encoder_learning_rate})
    return torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )


def mean_feature_std(feature_sum, feature_square_sum, count):
    variance = (feature_square_sum / count) - (feature_sum / count).pow(2)
    return float(variance.clamp_min(0.0).sqrt().mean())


def forecast_epoch(model, loader, device, config, optimizer=None, class_weights=None):
    training = optimizer is not None
    model.train(training)
    loss_meter = AverageMeter()
    latent_meter = AverageMeter()
    all_logits = []
    all_labels = []
    all_current_labels = []
    all_latent_errors = []
    all_change_logits = []
    all_destination_logits = []
    predicted_sum = None
    predicted_square_sum = None
    target_sum = None
    target_square_sum = None
    state_count = 0
    forecast_config = config["forecast"]
    for batch in loader:
        history_signals = batch["history_signals"].to(device=device, dtype=torch.float32)
        history_present = batch["history_present"].to(device=device, dtype=torch.bool)
        history_labels = batch["history_labels"].to(device=device, dtype=torch.long)
        future_signals = batch["future_signals"].to(device=device, dtype=torch.float32)
        future_present = batch["future_present"].to(device=device, dtype=torch.bool)
        future_labels = batch["future_labels"].to(device=device, dtype=torch.long)
        with torch.set_grad_enabled(training):
            output = model(
                history_signals,
                history_present,
                future_signals,
                future_present,
                future_labels,
                history_labels=history_labels,
                latent_weight=float(forecast_config.get("latent_weight", 1.0)),
                stage_weight=float(forecast_config.get("stage_weight", 1.0)),
                current_stage_weight=float(forecast_config.get("current_stage_weight", 0.0)),
                transition_stage_weight=float(forecast_config.get("transition_stage_weight", 1.0)),
                change_loss_weight=float(forecast_config.get("change_loss_weight", 0.0)),
                destination_loss_weight=float(forecast_config.get("destination_loss_weight", 0.0)),
                stage_class_weights=class_weights,
            )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                output["loss"].backward()
                grad_clip = float(config["train"].get("grad_clip", 0.0))
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        batch_size = history_signals.shape[0]
        loss_meter.update(float(output["loss"].detach().cpu()), batch_size)
        latent_errors = 1.0 - F.cosine_similarity(
            output["predicted_states"].detach(), output["target_states"].detach(), dim=-1
        )
        latent_meter.update(float(latent_errors.mean().cpu()), batch_size)
        all_latent_errors.append(latent_errors.cpu())
        predicted_states = output["predicted_states"].detach().float().reshape(-1, output["predicted_states"].shape[-1])
        target_states = output["target_states"].detach().float().reshape(-1, output["target_states"].shape[-1])
        batch_predicted_sum = predicted_states.sum(dim=0).cpu()
        batch_predicted_square_sum = predicted_states.square().sum(dim=0).cpu()
        batch_target_sum = target_states.sum(dim=0).cpu()
        batch_target_square_sum = target_states.square().sum(dim=0).cpu()
        predicted_sum = batch_predicted_sum if predicted_sum is None else predicted_sum + batch_predicted_sum
        predicted_square_sum = (
            batch_predicted_square_sum
            if predicted_square_sum is None
            else predicted_square_sum + batch_predicted_square_sum
        )
        target_sum = batch_target_sum if target_sum is None else target_sum + batch_target_sum
        target_square_sum = (
            batch_target_square_sum if target_square_sum is None else target_square_sum + batch_target_square_sum
        )
        state_count += predicted_states.shape[0]
        all_logits.append(output["stage_logits"].detach().cpu())
        all_labels.append(future_labels.detach().cpu())
        all_current_labels.append(history_labels[:, -1].detach().cpu())
        if "change_logits" in output:
            all_change_logits.append(output["change_logits"].detach().cpu())
            all_destination_logits.append(output["destination_logits"].detach().cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    current_labels = torch.cat(all_current_labels)
    num_classes = int(config["data"].get("num_classes", 5))
    metrics = classification_metrics(logits, labels, num_classes)
    metrics["loss"] = loss_meter.average
    metrics["latent_cosine_error"] = latent_meter.average
    metrics["predicted_state_feature_std"] = mean_feature_std(
        predicted_sum, predicted_square_sum, state_count
    )
    metrics["target_state_feature_std"] = mean_feature_std(target_sum, target_square_sum, state_count)
    latent_errors = torch.cat(all_latent_errors)
    metrics["latent_cosine_error_by_horizon"] = {
        str(horizon): float(latent_errors[:, index].mean())
        for index, horizon in enumerate(config["data"]["future_horizons"])
    }
    metrics["by_horizon"] = {
        str(horizon): classification_metrics(logits[:, index], labels[:, index], num_classes)
        for index, horizon in enumerate(config["data"]["future_horizons"])
    }
    metrics["subgroups"] = forecast_subgroup_metrics(
        logits,
        labels,
        current_labels,
        config["data"]["future_horizons"],
        num_classes,
    )
    if all_change_logits:
        metrics.update(
            factorized_transition_metrics(
                torch.cat(all_change_logits),
                torch.cat(all_destination_logits),
                labels,
                current_labels,
                config["data"]["future_horizons"],
                num_classes,
            )
        )
    return metrics


def main() -> int:
    args = parse_args()
    config = with_manifest(load_config(args.config), args.manifest)
    config = with_normalization(config, args.normalization)
    if config["experiment"]["kind"] != "forecast":
        raise ValueError("train_forecast.py requires experiment.kind=forecast")
    if args.checkpoint:
        config = copy.deepcopy(config)
        config["forecast"]["pretrained_checkpoint"] = args.checkpoint
    if args.seed is not None:
        config = copy.deepcopy(config)
        config["experiment"]["seed"] = int(args.seed)
    if args.epochs is not None:
        config = copy.deepcopy(config)
        config["train"]["epochs"] = int(args.epochs)
    if args.output_dir:
        config = copy.deepcopy(config)
        config["train"]["output_dir"] = str(Path(args.output_dir))
    if args.smoke:
        config = copy.deepcopy(config)
        config["train"]["epochs"] = 1
        config["train"]["num_workers"] = 0

    seed_everything(int(config["experiment"]["seed"]))
    device = resolve_device(str(config["train"].get("device", "cuda")))
    run_dir = prepare_run(config)
    train_data = sequence_dataset(config, "train")
    val_data = sequence_dataset(config, "val")
    train_loader = data_loader(train_data, config, shuffle=True)
    val_loader = data_loader(val_data, config, shuffle=False)
    transition_statistics = {
        "train_window_fraction": train_data.transition_window_fraction,
        "train_fraction_by_horizon": train_data.transition_fraction_by_horizon,
        "val_window_fraction": val_data.transition_window_fraction,
        "val_fraction_by_horizon": val_data.transition_fraction_by_horizon,
        "sampled_train_window_probability": config["train"].get("transition_window_probability"),
    }
    print(f"transition_statistics={transition_statistics}")

    encoder = MultiModalEncoder(observation_config(config["data"], config["model"]))
    if not args.random_init:
        load_encoder_state(encoder, config["forecast"]["pretrained_checkpoint"])
    model = CausalPhysioWorldModel(
        encoder,
        config["data"]["future_horizons"],
        num_classes=int(config["data"].get("num_classes", 5)),
        transition_layers=int(config["model"].get("transition_layers", 2)),
        transition_heads=int(config["model"].get("transition_heads", 4)),
        dropout=float(config["model"].get("dropout", 0.1)),
        freeze_observation_encoder=bool(config["forecast"].get("freeze_observation_encoder", False)),
        stage_residual_from_current=bool(config["forecast"].get("stage_residual_from_current", False)),
        use_frozen_target_encoder=bool(config["forecast"].get("use_frozen_target_encoder", False)),
        factorized_transition_head=bool(config["forecast"].get("factorized_transition_head", False)),
        change_prior_probabilities=config["forecast"].get("change_prior_probabilities"),
    ).to(device)
    current_head_initialized = False
    if not args.random_init and model.current_stage_head is not None:
        checkpoint = load_checkpoint(config["forecast"]["pretrained_checkpoint"])
        state = checkpoint.get("model_state", {})
        current_head_state = {
            key[len("head.") :]: value
            for key, value in state.items()
            if isinstance(key, str) and key.startswith("head.")
        }
        if current_head_state:
            model.current_stage_head.load_state_dict(current_head_state, strict=True)
            current_head_initialized = True
    optimizer = build_optimizer(model, config)
    class_weights = None
    if config["train"].get("class_weighting") == "balanced":
        class_weights = balanced_class_weights(train_data.future_label_counts).to(device)

    best_score = -1.0
    best_metrics = {}
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_metrics = forecast_epoch(model, train_loader, device, config, optimizer, class_weights)
        val_metrics = forecast_epoch(model, val_loader, device, config, class_weights=class_weights)
        change_text = ""
        if "change_detection" in val_metrics:
            change_text = (
                f" val_change_f1="
                f"{val_metrics['change_detection']['all_horizons']['per_class_f1'][1]:.4f}"
            )
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_latent_cosine_error={val_metrics['latent_cosine_error']:.4f} "
            f"val_target_feature_std={val_metrics['target_state_feature_std']:.4f}{change_text}"
        )
        if float(val_metrics["macro_f1"]) > best_score:
            best_score = float(val_metrics["macro_f1"])
            best_metrics = val_metrics
            save_checkpoint(
                run_dir / "best.pt",
                {
                    "experiment_id": config["experiment"]["id"],
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "encoder_state": model.encoder.state_dict(),
                    "validation_metrics": val_metrics,
                    "config": config,
                    "initialization": "random" if args.random_init else "pretrained",
                    "current_stage_head_initialized": current_head_initialized,
                    "transition_statistics": transition_statistics,
                },
            )
    write_json(
        run_dir / "metrics.json",
        {
            "best_validation": best_metrics,
            "initialization": "random" if args.random_init else "pretrained",
            "current_stage_head_initialized": current_head_initialized,
            "transition_statistics": transition_statistics,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
