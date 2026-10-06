from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from sleepwm.protocols import evaluate_subset
from sleepwm.training.observation_repair import selection_values
from sleepwm.config import load_config, with_manifest, with_normalization
from sleepwm.data import balanced_class_weights
from sleepwm.engine import data_loader, load_checkpoint, physio_feature_sequence_dataset, prepare_run, resolve_device, save_checkpoint, seed_everything, write_json
from sleepwm.masking import nonempty_modality_subsets
from sleepwm.masking import missing_rollout_summary, sampled_history_view
from sleepwm.models import PhysiologyFrontendEncoder, TaskAwareReliabilityObservationWaveformWorldModel, observation_config
from sleepwm.rollout import observation_repair_gate_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add modality-specific task distillation to repaired observation states."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--manifest")
    parser.add_argument("--normalization")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def build_model(config: dict) -> TaskAwareReliabilityObservationWaveformWorldModel:
    physiology = config["physiology"]
    waveform = config["waveform"]
    observation = config["observation_repair"]
    group_sizes = {
        group: len(names) for group, names in physiology["feature_groups"].items()
    }
    return TaskAwareReliabilityObservationWaveformWorldModel(
        PhysiologyFrontendEncoder(observation_config(config["data"], config["model"])),
        config["data"]["future_horizons"],
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
    )


def load_initialization(model, checkpoint_path: str) -> dict:
    checkpoint = load_checkpoint(checkpoint_path)
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    invalid_missing = [
        key for key in incompatible.missing_keys if not key.startswith("task_")
    ]
    if incompatible.unexpected_keys or invalid_missing:
        raise ValueError(
            "task-aware initialization mismatch: "
            f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
        )
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.task_parameters():
        parameter.requires_grad = True
    return checkpoint


def train_epoch(model, dataset, config: dict, optimizer, class_weights, device) -> dict:
    model.eval()
    model.task_stage_residual_heads.train()
    model.task_physiology_residual_heads.train()
    totals = {
        "loss": 0.0,
        "world_model_loss": 0.0,
        "stage_distillation_loss": 0.0,
        "physiology_distillation_loss": 0.0,
        "stage_loss": 0.0,
        "future_physiology_loss": 0.0,
    }
    samples_seen = 0
    observation = config["observation_repair"]
    for batch in data_loader(dataset, config, shuffle=True):
        natural_history = batch["history_signals"].to(
            device=device, dtype=torch.float32
        )
        natural_present = batch["history_present"].to(
            device=device, dtype=torch.bool
        )
        history_signals, history_present, _ = sampled_history_view(
            natural_history,
            natural_present,
            float(observation.get("full_modality_probability", 0.35)),
        )
        future_signals = batch["future_signals"].to(
            device=device, dtype=torch.float32
        )
        future_present = batch["future_present"].to(
            device=device, dtype=torch.bool
        )
        with torch.no_grad():
            teacher = model.direct_rollout_context(natural_history, natural_present)
        output = model(
            history_signals,
            history_present,
            future_signals,
            future_present,
            batch["future_labels"].to(device=device, dtype=torch.long),
            history_labels=batch["history_labels"].to(
                device=device, dtype=torch.long
            ),
            latent_weight=0.0,
            stage_weight=1.0,
            current_stage_weight=0.0,
            stage_class_weights=class_weights,
            future_physiology_targets=batch["future_physiology"].to(
                device=device, dtype=torch.float32
            ),
            future_physiology_valid=batch["future_physiology_valid"].to(
                device=device, dtype=torch.bool
            ),
            future_physiology_weight=1.0,
        )
        stage_distillation = F.kl_div(
            F.log_softmax(output["stage_logits"], dim=-1),
            F.softmax(teacher["stage_logits"], dim=-1),
            reduction="batchmean",
        ) / output["stage_logits"].shape[1]
        physiology_distillation = F.smooth_l1_loss(
            output["future_physiology"], teacher["future_physiology"]
        )
        world_model_loss = output["loss"]
        loss = (
            world_model_loss
            + float(observation.get("stage_distillation_weight", 1.0))
            * stage_distillation
            + float(observation.get("physiology_distillation_weight", 1.0))
            * physiology_distillation
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.task_parameters()),
            float(config["train"].get("grad_clip", 1.0)),
        )
        optimizer.step()
        batch_size = len(history_signals)
        samples_seen += batch_size
        values = {
            "loss": loss,
            "world_model_loss": world_model_loss,
            "stage_distillation_loss": stage_distillation,
            "physiology_distillation_loss": physiology_distillation,
            "stage_loss": output["stage_loss"],
            "future_physiology_loss": output["future_physiology_loss"],
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
    if args.checkpoint:
        config["observation_repair"]["checkpoint_path"] = args.checkpoint
    if args.epochs is not None:
        config["train"]["epochs"] = int(args.epochs)
    if args.output_dir:
        config["train"]["output_dir"] = str(Path(args.output_dir))
    seed_everything(int(config["experiment"]["seed"]))
    device = resolve_device(str(config["train"].get("device", "cuda")))
    run_dir = prepare_run(config)
    model = build_model(config).to(device)
    initialization = load_initialization(
        model, str(config["observation_repair"]["checkpoint_path"])
    )
    train_data = physio_feature_sequence_dataset(config, "train")
    val_data = physio_feature_sequence_dataset(config, "val")
    class_weights = balanced_class_weights(train_data.future_label_counts).to(device)
    optimizer = torch.optim.AdamW(
        list(model.task_parameters()),
        lr=float(config["observation_repair"].get("task_learning_rate", 1e-4)),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    modalities = tuple(config["data"]["modalities"])
    selection_subsets = (("ECG",), ("EMG",), modalities)
    best_key = (0.0, float("-inf"), float("-inf"))
    best_epoch = 0
    training_curve = []
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_metrics = train_epoch(
            model, train_data, config, optimizer, class_weights, device
        )
        selection_results = {
            "+".join(subset): evaluate_subset(
                model, val_data, config, device, subset
            )
            for subset in selection_subsets
        }
        key, values = selection_values(selection_results, modalities, config)
        training_curve.append(
            {"epoch": epoch, "train": train_metrics, "validation": values}
        )
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.6f} "
            f"full={values['full_stage_macro_f1']:.4f} "
            f"ecg={values['ecg_stage_macro_f1']:.4f} "
            f"emg={values['emg_stage_macro_f1']:.4f} "
            f"eligible={values['eligible']}",
            flush=True,
        )
        if key > best_key:
            best_key = key
            best_epoch = epoch
            save_checkpoint(
                run_dir / "best.pt",
                {
                    "experiment_id": config["experiment"]["id"],
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "validation": values,
                    "config": config,
                    "initial_checkpoint": config["observation_repair"][
                        "checkpoint_path"
                    ],
                    "initial_epoch": initialization.get("epoch"),
                },
            )
    selected = load_checkpoint(run_dir / "best.pt")
    model.load_state_dict(selected["model_state"], strict=True)
    results = {
        "+".join(subset): evaluate_subset(model, val_data, config, device, subset)
        for subset in nonempty_modality_subsets(modalities)
    }
    summary = missing_rollout_summary(results, modalities)
    observation = config["observation_repair"]
    gate = observation_repair_gate_result(
        results,
        modalities,
        minimum_alignment_improvement=float(
            observation.get("minimum_alignment_improvement", 0.05)
        ),
        maximum_full_stage_drop=float(
            observation.get("maximum_full_stage_drop", 0.005)
        ),
        minimum_nonfull_stage_delta=float(
            observation.get("minimum_nonfull_stage_delta", 0.0)
        ),
        minimum_single_modality_stage_delta=float(
            observation.get("minimum_single_modality_stage_delta", -0.005)
        ),
        maximum_waveform_ratio=float(
            observation.get("maximum_waveform_ratio", 1.02)
        ),
    )
    payload = {
        "best_epoch": best_epoch,
        "results": results,
        "summary": summary,
        "gate": gate,
        "training_curve": training_curve,
        "initial_checkpoint": config["observation_repair"]["checkpoint_path"],
        "train_sequences": len(train_data),
        "validation_sequences": len(val_data),
        "test_split_accessed": False,
    }
    write_json(run_dir / "metrics.json", payload)
    write_json(run_dir / "task_observation_repair_val.json", payload)
    print(json.dumps({"best_epoch": best_epoch, "gate": gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
