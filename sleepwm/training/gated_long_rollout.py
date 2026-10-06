from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from sleepwm.config import load_config, with_manifest, with_normalization
from sleepwm.data import balanced_class_weights
from sleepwm.engine import data_loader, load_checkpoint, physio_feature_sequence_dataset, prepare_run, resolve_device, save_checkpoint, seed_everything, write_json
from sleepwm.masking import nonempty_modality_subsets
from sleepwm.metrics import classification_metrics, forecast_subgroup_metrics
from sleepwm.masking import forced_history_view, sampled_history_view
from sleepwm.models import GatedRecursiveTaskAwareWaveformWorldModel, PhysiologyFrontendEncoder, observation_config
from sleepwm.metrics import standardized_physiology_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train quality-gated recursive latent rollout through 300/420 seconds."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--manifest")
    parser.add_argument("--normalization")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def build_model(config: dict) -> GatedRecursiveTaskAwareWaveformWorldModel:
    physiology = config["physiology"]
    waveform = config["waveform"]
    observation = config["observation_repair"]
    recursive = config["recursive"]
    group_sizes = {
        group: len(names) for group, names in physiology["feature_groups"].items()
    }
    return GatedRecursiveTaskAwareWaveformWorldModel(
        PhysiologyFrontendEncoder(observation_config(config["data"], config["model"])),
        recursive["direct_horizons"],
        rollout_horizons=config["data"]["future_horizons"],
        num_classes=int(config["data"].get("num_classes", 5)),
        transition_layers=int(config["model"].get("transition_layers", 2)),
        transition_heads=int(config["model"].get("transition_heads", 4)),
        dropout=float(config["model"].get("dropout", 0.1)),
        freeze_observation_encoder=False,
        stage_residual_from_current=True,
        use_frozen_target_encoder=True,
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
        waveform_structured_event_heads=True,
        waveform_probabilistic_event_heads=True,
        observation_adapter_hidden_dim=int(observation.get("hidden_dim", 128)),
        task_adapter_hidden_dim=int(observation.get("task_hidden_dim", 128)),
        recursive_hidden_dim=int(recursive.get("hidden_dim", 128)),
        recursive_anchor_direct=bool(recursive.get("anchor_direct", False)),
    )


def load_initialization(model, checkpoint_path: str) -> dict:
    checkpoint = load_checkpoint(checkpoint_path)
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    invalid_missing = [
        key for key in incompatible.missing_keys if not key.startswith("recursive_")
    ]
    if incompatible.unexpected_keys or invalid_missing:
        raise ValueError(
            "gated recursive initialization mismatch: "
            f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
        )
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.recursive_parameters():
        parameter.requires_grad = True
    return checkpoint


def _set_recursive_train_mode(model) -> None:
    model.eval()
    for name in (
        "recursive_state_norm",
        "recursive_hidden_initializer",
        "recursive_state_cell",
        "recursive_uncertainty_head",
        "recursive_update_gate",
        "recursive_delta_head",
    ):
        getattr(model, name).train()


def train_epoch(model, dataset, config: dict, optimizer, class_weights, device) -> dict:
    _set_recursive_train_mode(model)
    recursive = config["recursive"]
    horizons = tuple(int(value) for value in config["data"]["future_horizons"])
    direct_horizons = tuple(int(value) for value in recursive["direct_horizons"])
    direct_indices = [horizons.index(value) for value in direct_horizons]
    totals = {
        "loss": 0.0,
        "latent_loss": 0.0,
        "stage_loss": 0.0,
        "physiology_loss": 0.0,
        "short_teacher_loss": 0.0,
        "uncertainty_loss": 0.0,
    }
    samples_seen = 0
    for batch in data_loader(dataset, config, shuffle=True):
        history_signals = batch["history_signals"].to(device, torch.float32)
        history_present = batch["history_present"].to(device, torch.bool)
        history_signals, history_present, _ = sampled_history_view(
            history_signals,
            history_present,
            float(recursive.get("full_modality_probability", 0.35)),
        )
        future_signals = batch["future_signals"].to(device, torch.float32)
        future_present = batch["future_present"].to(device, torch.bool)
        with torch.no_grad():
            target_states = model._encode_epochs(
                future_signals,
                future_present,
                require_gradient=False,
                encoder=model.target_encoder,
            )
        output = model.rollout_context_horizons(
            history_signals, history_present, horizons
        )
        latent_loss = F.smooth_l1_loss(output["predicted_states"], target_states)
        stage_loss = F.cross_entropy(
            output["stage_logits"].reshape(-1, output["stage_logits"].shape[-1]),
            batch["future_labels"].to(device, torch.long).reshape(-1),
            weight=class_weights,
        )
        physiology_loss, _ = model._group_masked_loss(
            output["future_physiology"],
            batch["future_physiology"].to(device, torch.float32),
            batch["future_physiology_valid"].to(device, torch.bool),
        )
        short_teacher_loss = F.smooth_l1_loss(
            output["predicted_states"][:, direct_indices],
            output["observation_predicted_states"].detach(),
        )
        squared_error = (output["predicted_states"] - target_states).square().mean(-1)
        log_variance = output["recursive_log_variance"]
        uncertainty_loss = 0.5 * (
            torch.exp(-log_variance) * squared_error.detach() + log_variance
        ).mean()
        loss = (
            float(recursive.get("latent_weight", 1.0)) * latent_loss
            + float(recursive.get("stage_weight", 1.0)) * stage_loss
            + float(recursive.get("physiology_weight", 1.0)) * physiology_loss
            + float(recursive.get("short_teacher_weight", 0.25))
            * short_teacher_loss
            + float(recursive.get("uncertainty_weight", 0.1)) * uncertainty_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.recursive_parameters()),
            float(config["train"].get("grad_clip", 1.0)),
        )
        optimizer.step()
        batch_size = len(history_signals)
        samples_seen += batch_size
        values = {
            "loss": loss,
            "latent_loss": latent_loss,
            "stage_loss": stage_loss,
            "physiology_loss": physiology_loss,
            "short_teacher_loss": short_teacher_loss,
            "uncertainty_loss": uncertainty_loss,
        }
        for name, value in values.items():
            totals[name] += float(value.detach().cpu()) * batch_size
    return {name: value / samples_seen for name, value in totals.items()}


def evaluate_subset(model, dataset, config: dict, device, subset) -> dict:
    modalities = tuple(config["data"]["modalities"])
    horizons = tuple(int(value) for value in config["data"]["future_horizons"])
    stage_logits = []
    direct_stage_logits = []
    labels = []
    current_labels = []
    physiology = []
    physiology_target = []
    physiology_valid = []
    predicted_states = []
    target_states = []
    persistence_states = []
    corrections = []
    gates = []
    log_variances = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader(dataset, config, shuffle=False):
            history = batch["history_signals"].to(device, torch.float32)
            present = batch["history_present"].to(device, torch.bool)
            forced_history, forced_present = forced_history_view(
                history, present, subset, modalities
            )
            future = batch["future_signals"].to(device, torch.float32)
            future_present = batch["future_present"].to(device, torch.bool)
            output = model.rollout_context_horizons(
                forced_history, forced_present, horizons
            )
            target = model._encode_epochs(
                future,
                future_present,
                require_gradient=False,
                encoder=model.target_encoder,
            )
            stage_logits.append(output["stage_logits"].cpu())
            direct_stage_logits.append(output["observation_stage_logits"].cpu())
            labels.append(batch["future_labels"])
            current_labels.append(batch["history_labels"][:, -1])
            physiology.append(output["future_physiology"].cpu())
            physiology_target.append(batch["future_physiology"])
            physiology_valid.append(batch["future_physiology_valid"])
            predicted_states.append(output["predicted_states"].cpu())
            target_states.append(target.cpu())
            persistence_states.append(output["recursive_baseline_states"].cpu())
            corrections.append(output["recursive_state_correction"].cpu())
            gates.append(output["recursive_update_gate"].cpu())
            log_variances.append(output["recursive_log_variance"].cpu())
    logits = torch.cat(stage_logits)
    targets = torch.cat(labels)
    current = torch.cat(current_labels)
    states = torch.cat(predicted_states)
    latent_target = torch.cat(target_states)
    persistence = torch.cat(persistence_states)
    correction = torch.cat(corrections)
    gate_values = torch.cat(gates)
    uncertainty = torch.cat(log_variances).exp().sqrt()
    num_classes = int(config["data"].get("num_classes", 5))
    by_horizon = {}
    for index, horizon in enumerate(horizons):
        recurrent_loss = float(F.smooth_l1_loss(states[:, index], latent_target[:, index]))
        persistence_loss = float(
            F.smooth_l1_loss(persistence[:, index], latent_target[:, index])
        )
        by_horizon[str(horizon)] = {
            "recurrent_smooth_l1": recurrent_loss,
            "persistence_smooth_l1": persistence_loss,
            "relative_improvement": (persistence_loss - recurrent_loss)
            / max(persistence_loss, 1e-12),
            "cosine_similarity": float(
                F.cosine_similarity(states[:, index], latent_target[:, index], dim=-1).mean()
            ),
            "correction_rms": float(correction[:, index].square().mean().sqrt()),
            "mean_update_gate": float(gate_values[:, index].mean()),
            "mean_predicted_std": float(uncertainty[:, index].mean()),
        }
    direct_logits = torch.cat(direct_stage_logits)
    direct_count = direct_logits.shape[1]
    feature_names = tuple(config["physiology"]["feature_names"])
    feature_groups = {
        group: tuple(names)
        for group, names in config["physiology"]["feature_groups"].items()
    }
    return {
        "stage": {
            "all_horizons": classification_metrics(logits, targets, num_classes),
            "by_horizon": {
                str(horizon): classification_metrics(
                    logits[:, index], targets[:, index], num_classes
                )
                for index, horizon in enumerate(horizons)
            },
            "subgroups": forecast_subgroup_metrics(
                logits, targets, current, horizons, num_classes
            ),
            "direct_short": classification_metrics(
                direct_logits, targets[:, :direct_count], num_classes
            ),
            "recurrent_short": classification_metrics(
                logits[:, :direct_count], targets[:, :direct_count], num_classes
            ),
            "current_stage_persistence": classification_metrics(
                torch.nn.functional.one_hot(current, num_classes).float()
                .unsqueeze(1)
                .expand(-1, len(horizons), -1),
                targets,
                num_classes,
            ),
        },
        "future_physiology": standardized_physiology_metrics(
            torch.cat(physiology),
            torch.cat(physiology_target),
            torch.cat(physiology_valid),
            feature_names,
            feature_groups,
            horizons,
        ),
        "latent": {
            "by_horizon": by_horizon,
            "recurrent_smooth_l1": float(F.smooth_l1_loss(states, latent_target)),
            "persistence_smooth_l1": float(
                F.smooth_l1_loss(persistence, latent_target)
            ),
            "correction_rms": float(correction.square().mean().sqrt()),
        },
        "gate": {
            "mean": float(gate_values.mean()),
            "std": float(gate_values.std()),
            "predicted_std_mean": float(uncertainty.mean()),
            "predicted_std_std": float(uncertainty.std()),
        },
    }


def promotion_gate(results: dict, config: dict) -> dict:
    recursive = config["recursive"]
    modalities = tuple(config["data"]["modalities"])
    full_name = "+".join(modalities)
    full = results[full_name]
    long_horizons = tuple(str(value) for value in recursive["long_horizons"])
    long_improvements = {
        horizon: float(full["latent"]["by_horizon"][horizon]["relative_improvement"])
        for horizon in long_horizons
    }
    direct_short = float(full["stage"]["direct_short"]["macro_f1"])
    recurrent_short = float(full["stage"]["recurrent_short"]["macro_f1"])
    gate_means = [float(value["gate"]["mean"]) for value in results.values()]
    result = {
        "long_latent_improved": all(value > 0.0 for value in long_improvements.values()),
        "long_horizon_improvements": long_improvements,
        "short_stage_retained": direct_short - recurrent_short
        <= float(recursive.get("maximum_short_stage_drop", 0.03)),
        "direct_short_stage_macro_f1": direct_short,
        "recurrent_short_stage_macro_f1": recurrent_short,
        "recursive_path_active": float(full["latent"]["correction_rms"])
        >= float(recursive.get("minimum_correction_rms", 1e-4)),
        "uncertainty_nonconstant": float(full["gate"]["predicted_std_std"])
        >= float(recursive.get("minimum_uncertainty_std", 1e-4)),
        "modality_gate_responsive": max(gate_means) - min(gate_means)
        >= float(recursive.get("minimum_subset_gate_range", 1e-4)),
        "subset_gate_range": max(gate_means) - min(gate_means),
        "full_long_stage_macro_f1": sum(
            float(full["stage"]["by_horizon"][horizon]["macro_f1"])
            for horizon in long_horizons
        )
        / len(long_horizons),
    }
    result["passed"] = all(
        bool(result[name])
        for name in (
            "long_latent_improved",
            "short_stage_retained",
            "recursive_path_active",
            "uncertainty_nonconstant",
            "modality_gate_responsive",
        )
    )
    return result


def main() -> int:
    args = parse_args()
    config = with_normalization(
        with_manifest(load_config(args.config), args.manifest), args.normalization
    )
    config = copy.deepcopy(config)
    if args.checkpoint:
        config["recursive"]["checkpoint_path"] = args.checkpoint
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
        model, str(config["recursive"]["checkpoint_path"])
    )
    train_data = physio_feature_sequence_dataset(config, "train")
    val_data = physio_feature_sequence_dataset(config, "val")
    class_weights = balanced_class_weights(train_data.future_label_counts).to(device)
    if args.smoke:
        train_data = Subset(train_data, range(min(64, len(train_data))))
        val_data = Subset(val_data, range(min(64, len(val_data))))
    optimizer = torch.optim.AdamW(
        list(model.recursive_parameters()),
        lr=float(config["recursive"].get("learning_rate", 1e-4)),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    modalities = tuple(config["data"]["modalities"])
    best_key = (2.0, float("inf"))
    best_epoch = 0
    training_curve = []
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_metrics = train_epoch(
            model, train_data, config, optimizer, class_weights, device
        )
        validation = evaluate_subset(model, val_data, config, device, modalities)
        direct_short = float(validation["stage"]["direct_short"]["macro_f1"])
        recurrent_short = float(validation["stage"]["recurrent_short"]["macro_f1"])
        eligible = direct_short - recurrent_short <= float(
            config["recursive"].get("maximum_short_stage_drop", 0.03)
        )
        long_loss = sum(
            float(validation["latent"]["by_horizon"][str(h)]["recurrent_smooth_l1"])
            for h in config["recursive"]["long_horizons"]
        )
        key = (0.0 if eligible else 1.0, long_loss)
        training_curve.append(
            {"epoch": epoch, "train": train_metrics, "validation": validation}
        )
        print(
            f"epoch={epoch:03d} loss={train_metrics['loss']:.6f} "
            f"short_stage={recurrent_short:.4f} long_latent={long_loss:.6f} "
            f"eligible={eligible}",
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
                    "validation": validation,
                    "config": config,
                    "initial_checkpoint": config["recursive"]["checkpoint_path"],
                    "initial_epoch": initialization.get("epoch"),
                },
            )
    if best_epoch < 1:
        raise RuntimeError("no gated recursive checkpoint was selected")
    selected = load_checkpoint(run_dir / "best.pt")
    model.load_state_dict(selected["model_state"], strict=True)
    results = {
        "+".join(subset): evaluate_subset(model, val_data, config, device, subset)
        for subset in nonempty_modality_subsets(modalities)
    }
    gate = promotion_gate(results, config)
    payload = {
        "best_epoch": best_epoch,
        "results": results,
        "gate": gate,
        "training_curve": training_curve,
        "initial_checkpoint": config["recursive"]["checkpoint_path"],
        "train_sequences": len(train_data),
        "validation_sequences": len(val_data),
        "test_split_accessed": False,
    }
    write_json(run_dir / "metrics.json", payload)
    write_json(run_dir / "gated_long_rollout_val.json", payload)
    print(json.dumps({"best_epoch": best_epoch, "gate": gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
