from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from sleepwm.config import load_config, with_manifest, with_normalization
from sleepwm.data import balanced_class_weights
from sleepwm.engine import AverageMeter, data_loader, load_checkpoint, physio_feature_sequence_dataset, prepare_run, resolve_device, save_checkpoint, seed_everything, write_json
from sleepwm.metrics import classification_metrics, forecast_subgroup_metrics
from sleepwm.models import MultiModalEncoder, PrivateTemporalPhysiologyWorldModel, PhysiologyFrontendEncoder, PhysiologyAwareWorldModel, PhysiologyStateSpaceWorldModel, TrajectoryPhysiologyWorldModel, observation_config
from sleepwm.metrics import standardized_physiology_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train physiology-aware causal world model heads.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--normalization")
    parser.add_argument("--checkpoint", help="override physiology.initial_world_model_checkpoint")
    parser.add_argument("--feature-manifest")
    parser.add_argument("--feature-statistics")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def build_model(config: dict) -> PhysiologyAwareWorldModel:
    group_sizes = {
        group: len(names) for group, names in config["physiology"]["feature_groups"].items()
    }
    state_space_dynamics = bool(
        config["physiology"].get("physiology_state_space_dynamics", False)
    )
    private_temporal = bool(config["physiology"].get("private_temporal_dynamics", False))
    model_class = PhysiologyStateSpaceWorldModel if state_space_dynamics else (
        PrivateTemporalPhysiologyWorldModel if private_temporal else (
        TrajectoryPhysiologyWorldModel
        if bool(config["physiology"].get("history_trajectory_supervision", False))
        else PhysiologyAwareWorldModel
        )
    )
    model_arguments = {}
    if state_space_dynamics:
        model_arguments = {
            "physiology_dynamics_dim": int(
                config["physiology"].get("physiology_dynamics_dim", 64)
            ),
            "physiology_dynamics_layers": int(
                config["physiology"].get("physiology_dynamics_layers", 1)
            ),
            "physiology_dynamics_heads": int(
                config["physiology"].get("physiology_dynamics_heads", 4)
            ),
        }
    elif private_temporal:
        model_arguments = {
            "private_transition_layers": int(
                config["physiology"].get("private_transition_layers", 1)
            ),
            "private_transition_heads": int(
                config["physiology"].get("private_transition_heads", 4)
            ),
        }
    encoder_class = (
        PhysiologyFrontendEncoder
        if bool(config["physiology"].get("modality_physiology_frontend", False))
        else MultiModalEncoder
    )
    return model_class(
        encoder_class(observation_config(config["data"], config["model"])),
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
        physiology_group_sizes=group_sizes,
        physiology_hidden_dim=int(config["physiology"].get("hidden_dim", 128)),
        **model_arguments,
    )


def load_stage4_initialization(model: PhysiologyAwareWorldModel, checkpoint_path: str) -> dict:
    checkpoint = load_checkpoint(checkpoint_path)
    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("Stage 4 checkpoint does not contain model_state")
    incompatible = model.load_state_dict(state, strict=False)
    allowed_prefixes = (
        "current_physiology_heads.",
        "future_physiology_delta_heads.",
        "physiology_state_adapters.",
        "private_transitions.",
        "private_horizon_embeddings.",
        "private_state_predictors.",
        "private_future_state_adapters.",
        "encoder.physiology_frontends.",
        "target_encoder.physiology_frontends.",
        "physiology_dynamics_embeddings.",
        "physiology_dynamics_transitions.",
        "physiology_dynamics_horizon_embeddings.",
        "physiology_dynamics_predictors.",
        "physiology_dynamics_delta_heads.",
        "physiology_dynamics_state_adapters.",
    )
    unexpected_missing = [
        key for key in incompatible.missing_keys if not key.startswith(allowed_prefixes)
    ]
    if incompatible.unexpected_keys or unexpected_missing:
        raise ValueError(
            f"Stage 4 initialization mismatch: missing={unexpected_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return checkpoint


def build_optimizer(model: PhysiologyAwareWorldModel, config: dict) -> torch.optim.Optimizer:
    state_space_dynamics = bool(
        getattr(model, "uses_physiology_state_space_dynamics", False)
    )
    head_parameters = []
    if not state_space_dynamics:
        head_parameters += list(model.current_physiology_heads.parameters())
        head_parameters += list(model.future_physiology_delta_heads.parameters())
        if hasattr(model, "physiology_state_adapters"):
            head_parameters += list(model.physiology_state_adapters.parameters())
        if hasattr(model.encoder, "physiology_frontends"):
            head_parameters += list(model.encoder.physiology_frontends.parameters())
    module_names = (
        "physiology_dynamics_embeddings",
        "physiology_dynamics_transitions",
        "physiology_dynamics_horizon_embeddings",
        "physiology_dynamics_predictors",
        "physiology_dynamics_delta_heads",
        "physiology_dynamics_state_adapters",
    ) if state_space_dynamics else (
        "private_transitions",
        "private_horizon_embeddings",
        "private_state_predictors",
        "private_future_state_adapters",
    )
    for module_name in module_names:
        if hasattr(model, module_name):
            head_parameters += list(getattr(model, module_name).parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    return torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": float(config["physiology"].get("backbone_learning_rate", 1e-5)),
            },
            {
                "params": head_parameters,
                "lr": float(config["physiology"].get("head_learning_rate", 1e-4)),
            },
        ],
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )


def physio_forecast_epoch(model, loader, device, config, optimizer=None, class_weights=None):
    training = optimizer is not None
    model.train(training)
    loss_meter = AverageMeter()
    latent_meter = AverageMeter()
    observation_loss_meter = AverageMeter()
    future_loss_meter = AverageMeter()
    private_latent_meter = AverageMeter()
    all_logits = []
    all_labels = []
    all_current_labels = []
    current_prediction = []
    current_target = []
    current_valid = []
    future_prediction = []
    future_target = []
    future_valid = []
    forecast = config["forecast"]
    physiology = config["physiology"]
    trajectory_supervision = bool(getattr(model, "uses_history_trajectory", False))
    private_temporal = bool(getattr(model, "uses_private_temporal_dynamics", False))
    observation_loss_key = (
        "history_physiology_loss" if trajectory_supervision else "current_physiology_loss"
    )
    for batch in loader:
        history_signals = batch["history_signals"].to(device=device, dtype=torch.float32)
        history_present = batch["history_present"].to(device=device, dtype=torch.bool)
        history_labels = batch["history_labels"].to(device=device, dtype=torch.long)
        future_signals = batch["future_signals"].to(device=device, dtype=torch.float32)
        future_present = batch["future_present"].to(device=device, dtype=torch.bool)
        future_labels = batch["future_labels"].to(device=device, dtype=torch.long)
        batch_current_target = batch["current_physiology"].to(device=device, dtype=torch.float32)
        batch_current_valid = batch["current_physiology_valid"].to(device=device, dtype=torch.bool)
        batch_future_target = batch["future_physiology"].to(device=device, dtype=torch.float32)
        batch_future_valid = batch["future_physiology_valid"].to(device=device, dtype=torch.bool)
        physiology_arguments = {
            "future_physiology_targets": batch_future_target,
            "future_physiology_valid": batch_future_valid,
            "future_physiology_weight": float(physiology.get("future_weight", 1.0)),
        }
        if trajectory_supervision:
            physiology_arguments.update(
                {
                    "history_physiology_targets": batch["history_physiology"].to(
                        device=device, dtype=torch.float32
                    ),
                    "history_physiology_valid": batch["history_physiology_valid"].to(
                        device=device, dtype=torch.bool
                    ),
                    "history_physiology_weight": float(physiology.get("history_weight", 1.0)),
                }
            )
        else:
            physiology_arguments.update(
                {
                    "current_physiology_targets": batch_current_target,
                    "current_physiology_valid": batch_current_valid,
                    "current_physiology_weight": float(physiology.get("current_weight", 1.0)),
                }
            )
        if private_temporal:
            physiology_arguments["private_latent_weight"] = float(
                physiology.get("private_latent_weight", 1.0)
            )
        with torch.set_grad_enabled(training):
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
                **physiology_arguments,
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
        observation_loss_meter.update(float(output[observation_loss_key].detach().cpu()), batch_size)
        future_loss_meter.update(float(output["future_physiology_loss"].detach().cpu()), batch_size)
        if "private_latent_loss" in output:
            private_latent_meter.update(float(output["private_latent_loss"].detach().cpu()), batch_size)
        latent_error = 1.0 - F.cosine_similarity(
            output["predicted_states"].detach(), output["target_states"].detach(), dim=-1
        )
        latent_meter.update(float(latent_error.mean().cpu()), batch_size)
        all_logits.append(output["stage_logits"].detach().cpu())
        all_labels.append(future_labels.detach().cpu())
        all_current_labels.append(history_labels[:, -1].detach().cpu())
        current_prediction.append(output["current_physiology"].detach().cpu())
        current_target.append(batch_current_target.detach().cpu())
        current_valid.append(batch_current_valid.detach().cpu())
        future_prediction.append(output["future_physiology"].detach().cpu())
        future_target.append(batch_future_target.detach().cpu())
        future_valid.append(batch_future_valid.detach().cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    current_labels = torch.cat(all_current_labels)
    metrics = classification_metrics(logits, labels, int(config["data"].get("num_classes", 5)))
    metrics.update(
        {
            "loss": loss_meter.average,
            "latent_cosine_error": latent_meter.average,
            observation_loss_key: observation_loss_meter.average,
            "future_physiology_loss": future_loss_meter.average,
            **(
                {"private_latent_loss": private_latent_meter.average}
                if private_temporal
                else {}
            ),
            "by_horizon": {
                str(horizon): classification_metrics(
                    logits[:, index], labels[:, index], int(config["data"].get("num_classes", 5))
                )
                for index, horizon in enumerate(config["data"]["future_horizons"])
            },
            "subgroups": forecast_subgroup_metrics(
                logits,
                labels,
                current_labels,
                config["data"]["future_horizons"],
                int(config["data"].get("num_classes", 5)),
            ),
        }
    )
    feature_names = tuple(physiology["feature_names"])
    feature_groups = {group: tuple(names) for group, names in physiology["feature_groups"].items()}
    metrics["current_physiology"] = standardized_physiology_metrics(
        torch.cat(current_prediction),
        torch.cat(current_target),
        torch.cat(current_valid),
        feature_names,
        feature_groups,
    )
    metrics["future_physiology"] = standardized_physiology_metrics(
        torch.cat(future_prediction),
        torch.cat(future_target),
        torch.cat(future_valid),
        feature_names,
        feature_groups,
        config["data"]["future_horizons"],
    )
    return metrics


def gate_result(metrics: dict, config: dict) -> dict:
    physiology = config["physiology"]
    future = metrics["future_physiology"]
    improved_groups = [
        group
        for group, threshold in physiology["ar_group_normalized_mae"].items()
        if future["by_group"][group]["mean_normalized_mae"] < float(threshold)
    ]
    result = {
        "future_below_ar20": future["all_features"]["mean_normalized_mae"]
        < float(physiology["ar_normalized_mae"]),
        "groups_below_ar20": improved_groups,
        "stage_retained": float(metrics["macro_f1"])
        >= float(physiology["stage_retention_floor"]),
    }
    result["passed"] = bool(
        result["future_below_ar20"]
        and len(improved_groups) >= 2
        and result["stage_retained"]
    )
    return result


def trajectory_gate_result(metrics: dict, config: dict):
    physiology = config["physiology"]
    if "f1_current_normalized_mae" not in physiology:
        return None
    relative_improvement = float(physiology.get("current_relative_improvement", 0.10))
    current_threshold = float(physiology["f1_current_normalized_mae"]) * (
        1.0 - relative_improvement
    )
    result = {
        "current_threshold": current_threshold,
        "current_improved": metrics["current_physiology"]["all_features"][
            "mean_normalized_mae"
        ]
        < current_threshold,
        "future_below_f1": metrics["future_physiology"]["all_features"][
            "mean_normalized_mae"
        ]
        < float(physiology["f1_future_normalized_mae"]),
        "stage_retained": float(metrics["macro_f1"])
        >= float(physiology["stage_retention_floor"]),
    }
    result["passed"] = bool(
        result["current_improved"]
        and result["future_below_f1"]
        and result["stage_retained"]
    )
    return result


def private_temporal_gate_result(metrics: dict, config: dict):
    physiology = config["physiology"]
    if "f2_group_normalized_mae" not in physiology:
        return None
    future_threshold = float(physiology["f2_future_normalized_mae"]) * (
        1.0 - float(physiology.get("future_relative_improvement", 0.05))
    )
    current_ceiling = float(physiology["f2_current_normalized_mae"]) + float(
        physiology.get("current_regression_tolerance", 0.01)
    )
    improved_groups = [
        group
        for group, threshold in physiology["f2_group_normalized_mae"].items()
        if metrics["future_physiology"]["by_group"][group]["mean_normalized_mae"]
        < float(threshold)
    ]
    result = {
        "future_threshold": future_threshold,
        "future_improved": metrics["future_physiology"]["all_features"][
            "mean_normalized_mae"
        ]
        < future_threshold,
        "current_ceiling": current_ceiling,
        "current_retained": metrics["current_physiology"]["all_features"][
            "mean_normalized_mae"
        ]
        <= current_ceiling,
        "groups_below_f2": improved_groups,
        "stage_retained": float(metrics["macro_f1"])
        >= float(physiology["stage_retention_floor"]),
    }
    result["passed"] = bool(
        result["future_improved"]
        and result["current_retained"]
        and len(improved_groups) >= 2
        and result["stage_retained"]
    )
    return result


def observation_frontend_gate_result(metrics: dict, config: dict):
    physiology = config["physiology"]
    if "f2_current_ecg_normalized_mae" not in physiology:
        return None
    improvement = float(physiology.get("observation_relative_improvement", 0.10))
    current_threshold = float(physiology["f2_current_normalized_mae"]) * (1.0 - improvement)
    ecg_threshold = float(physiology["f2_current_ecg_normalized_mae"]) * (1.0 - improvement)
    result = {
        "current_threshold": current_threshold,
        "current_improved": metrics["current_physiology"]["all_features"][
            "mean_normalized_mae"
        ]
        < current_threshold,
        "ecg_current_threshold": ecg_threshold,
        "ecg_current_improved": metrics["current_physiology"]["by_group"]["ECG"][
            "mean_normalized_mae"
        ]
        < ecg_threshold,
        "future_below_f2": metrics["future_physiology"]["all_features"][
            "mean_normalized_mae"
        ]
        < float(physiology["f2_future_normalized_mae"]),
        "stage_retained": float(metrics["macro_f1"])
        >= float(physiology["stage_retention_floor"]),
    }
    result["passed"] = bool(
        result["current_improved"]
        and result["ecg_current_improved"]
        and result["future_below_f2"]
        and result["stage_retained"]
    )
    return result


def physiology_state_space_gate_result(metrics: dict, config: dict):
    physiology = config["physiology"]
    if "f4_group_normalized_mae" not in physiology:
        return None
    future_threshold = float(physiology["f4_future_normalized_mae"]) * (
        1.0 - float(physiology.get("future_relative_improvement", 0.10))
    )
    current_ceiling = float(physiology["f4_current_normalized_mae"]) + float(
        physiology.get("current_regression_tolerance", 0.01)
    )
    improved_groups = [
        group
        for group, threshold in physiology["f4_group_normalized_mae"].items()
        if metrics["future_physiology"]["by_group"][group]["mean_normalized_mae"]
        < float(threshold)
    ]
    result = {
        "future_threshold": future_threshold,
        "future_improved": metrics["future_physiology"]["all_features"][
            "mean_normalized_mae"
        ]
        < future_threshold,
        "current_ceiling": current_ceiling,
        "current_retained": metrics["current_physiology"]["all_features"][
            "mean_normalized_mae"
        ]
        <= current_ceiling,
        "groups_below_f4": improved_groups,
        "stage_retained": float(metrics["macro_f1"])
        >= float(physiology["stage_retention_floor"]),
    }
    result["passed"] = bool(
        result["future_improved"]
        and result["current_retained"]
        and len(improved_groups) >= 2
        and result["stage_retained"]
    )
    return result


def main() -> int:
    args = parse_args()
    config = with_normalization(with_manifest(load_config(args.config), args.manifest), args.normalization)
    config = copy.deepcopy(config)
    if args.checkpoint:
        config["physiology"]["initial_world_model_checkpoint"] = args.checkpoint
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

    statistics = json.loads(Path(config["physiology"]["feature_statistics_path"]).read_text())
    if tuple(statistics["feature_names"]) != tuple(config["physiology"]["feature_names"]):
        raise ValueError("configured physiology features do not match train statistics")
    seed_everything(int(config["experiment"]["seed"]))
    device = resolve_device(str(config["train"].get("device", "cuda")))
    run_dir = prepare_run(config)
    train_data = physio_feature_sequence_dataset(config, "train")
    val_data = physio_feature_sequence_dataset(config, "val")
    train_loader = data_loader(train_data, config, shuffle=True)
    val_loader = data_loader(val_data, config, shuffle=False)
    model = build_model(config).to(device)
    initialization = load_stage4_initialization(
        model, config["physiology"]["initial_world_model_checkpoint"]
    )
    optimizer = build_optimizer(model, config)
    class_weights = balanced_class_weights(train_data.future_label_counts).to(device)

    stage_floor = float(config["physiology"]["stage_retention_floor"])
    best_feasible_score = float("inf")
    best_any_score = float("inf")
    best_feasible = False
    best_metrics = None
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_metrics = physio_forecast_epoch(
            model, train_loader, device, config, optimizer, class_weights
        )
        val_metrics = physio_forecast_epoch(model, val_loader, device, config, class_weights=class_weights)
        future_score = float(
            val_metrics["future_physiology"]["all_features"]["mean_normalized_mae"]
        )
        feasible = float(val_metrics["macro_f1"]) >= stage_floor
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_current_physio_nmae="
            f"{val_metrics['current_physiology']['all_features']['mean_normalized_mae']:.4f} "
            f"val_future_physio_nmae={future_score:.4f} feasible={feasible}"
        )
        should_save = False
        if feasible and (not best_feasible or future_score < best_feasible_score):
            best_feasible = True
            best_feasible_score = future_score
            should_save = True
        elif not best_feasible and future_score < best_any_score:
            best_any_score = future_score
            should_save = True
        if should_save:
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
                    "initial_world_model_checkpoint": config["physiology"][
                        "initial_world_model_checkpoint"
                    ],
                    "initial_world_model_epoch": initialization.get("epoch"),
                    "stage_retention_feasible": feasible,
                },
            )
    if best_metrics is None:
        raise RuntimeError("no physiology-aware checkpoint was selected")
    gate = gate_result(best_metrics, config)
    trajectory_gate = trajectory_gate_result(best_metrics, config)
    private_temporal_gate = private_temporal_gate_result(best_metrics, config)
    observation_frontend_gate = observation_frontend_gate_result(best_metrics, config)
    physiology_state_space_gate = physiology_state_space_gate_result(best_metrics, config)
    payload = {
        "best_validation": best_metrics,
        "gate": gate,
        **({"trajectory_gate": trajectory_gate} if trajectory_gate is not None else {}),
        **(
            {"private_temporal_gate": private_temporal_gate}
            if private_temporal_gate is not None
            else {}
        ),
        **(
            {"observation_frontend_gate": observation_frontend_gate}
            if observation_frontend_gate is not None
            else {}
        ),
        **(
            {"physiology_state_space_gate": physiology_state_space_gate}
            if physiology_state_space_gate is not None
            else {}
        ),
        "selection_stage_floor": stage_floor,
        "initial_world_model_checkpoint": config["physiology"]["initial_world_model_checkpoint"],
    }
    write_json(run_dir / "metrics.json", payload)
    write_json(
        run_dir / "evaluation_val.json",
        {
            "checkpoint": str(run_dir / "best.pt"),
            "split": "val",
            "results": best_metrics,
            "gate": gate,
            **({"trajectory_gate": trajectory_gate} if trajectory_gate is not None else {}),
            **(
                {"private_temporal_gate": private_temporal_gate}
                if private_temporal_gate is not None
                else {}
            ),
            **(
                {"observation_frontend_gate": observation_frontend_gate}
                if observation_frontend_gate is not None
                else {}
            ),
            **(
                {"physiology_state_space_gate": physiology_state_space_gate}
                if physiology_state_space_gate is not None
                else {}
            ),
        },
    )
    print(f"gate_passed={gate['passed']}")
    if trajectory_gate is not None:
        print(f"trajectory_gate_passed={trajectory_gate['passed']}")
    if private_temporal_gate is not None:
        print(f"private_temporal_gate_passed={private_temporal_gate['passed']}")
    if observation_frontend_gate is not None:
        print(f"observation_frontend_gate_passed={observation_frontend_gate['passed']}")
    if physiology_state_space_gate is not None:
        print(f"physiology_state_space_gate_passed={physiology_state_space_gate['passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
