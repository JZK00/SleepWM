from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Subset


from sleepwm.readouts import LatentHazardSafeAdapter
from sleepwm.readouts import build_direct_branch
from sleepwm.protocols import dynamic_view
from sleepwm.protocols import average_precision, event_risk_scores, load_label_map, next_event_offset
from sleepwm.training.recursive_belief_filter import build_student
from sleepwm.training.trajectory_outcome_adapter import build_adapter, masked_smooth_l1
from sleepwm.engine import data_loader, load_checkpoint, physio_feature_sequence_dataset, resolve_device, save_checkpoint, seed_everything
from sleepwm.metrics import classification_metrics
from sleepwm.masking import DynamicObservationSpec
from sleepwm.metrics import standardized_physiology_metrics


PRIMARY_HORIZONS = (1, 2, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--po3-checkpoint")
    parser.add_argument("--outcome-checkpoint")
    parser.add_argument("--direct-checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-test", action="store_true", default=True)
    parser.add_argument("--evaluate-test", dest="skip_test", action="store_false", help="Explicitly enable held-out test evaluation")
    return parser.parse_args()


def evaluation_specs(
    modalities: Sequence[str],
) -> tuple[DynamicObservationSpec, ...]:
    all_for = lambda duration: {modality: duration for modality in modalities}
    return (
        DynamicObservationSpec("hard_eeg_1ep", {"EEG": 1}),
        DynamicObservationSpec("hard_eeg_4ep", {"EEG": 4}),
        DynamicObservationSpec("hard_eeg_10ep", {"EEG": 10}),
        DynamicObservationSpec("hard_all_1ep", all_for(1)),
        DynamicObservationSpec("hard_all_4ep", all_for(4)),
        DynamicObservationSpec("hard_all_10ep", all_for(10)),
        DynamicObservationSpec(
            "linear_decay_all_4ep", all_for(4), profile="linear_decay"
        ),
        DynamicObservationSpec(
            "asynchronous_eeg4_ecg2_emg1",
            {"EEG": 4, "ECG": 2, "EMG": 1},
        ),
    )


def training_specs(
    modalities: Sequence[str],
) -> tuple[Optional[DynamicObservationSpec], ...]:
    all_for = lambda duration: {modality: duration for modality in modalities}
    return (
        None,
        None,
        DynamicObservationSpec("hard_eeg_1ep", {"EEG": 1}),
        DynamicObservationSpec("hard_eeg_4ep", {"EEG": 4}),
        DynamicObservationSpec("hard_eeg_10ep", {"EEG": 10}),
        DynamicObservationSpec("hard_all_1ep", all_for(1)),
        DynamicObservationSpec("hard_all_4ep", all_for(4)),
        DynamicObservationSpec("hard_all_10ep", all_for(10)),
        DynamicObservationSpec(
            "linear_decay_all_4ep", all_for(4), profile="linear_decay"
        ),
        DynamicObservationSpec(
            "asynchronous_eeg4_ecg2_emg1",
            {"EEG": 4, "ECG": 2, "EMG": 1},
        ),
    )


def transition_targets(
    batch,
    label_map: dict[str, np.ndarray],
    horizons: Sequence[int],
    history_epochs: int,
    device: torch.device,
) -> torch.Tensor:
    origins = (batch["start_epoch"] + history_epochs - 1).tolist()
    maximum = max(int(value) for value in horizons)
    offsets = [
        next_event_offset(
            label_map[str(record)], int(origin), "transition", maximum
        )
        for record, origin in zip(batch["record_id"], origins)
    ]
    return torch.tensor(
        [[offset <= int(horizon) for horizon in horizons] for offset in offsets],
        device=device,
        dtype=torch.float32,
    )


def frozen_outputs(
    sleep_model,
    sleep_adapter,
    direct_model,
    signals: torch.Tensor,
    present: torch.Tensor,
    horizons: Sequence[int],
):
    sleep_output = sleep_model.rollout_context_horizons(signals, present, horizons)
    sleep_adapted = sleep_adapter(sleep_output)
    direct_output = direct_model(signals, present)
    sleep_current = sleep_output.get(
        "belief_current_stage_logits", sleep_output["current_stage_logits"]
    )
    sleep_risk = event_risk_scores(
        sleep_current.softmax(dim=-1),
        sleep_adapted["stage_logits"].softmax(dim=-1),
    )["transition"]
    direct_risk = event_risk_scores(
        direct_output["current_logits"].softmax(dim=-1),
        direct_output["future_logits"].softmax(dim=-1),
    )["transition"]
    return sleep_output, sleep_adapted, sleep_risk, direct_output, direct_risk


def event_ranking_loss(risk: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    losses = []
    for index in range(risk.shape[1]):
        positive = risk[target[:, index] > 0.5, index]
        negative = risk[target[:, index] <= 0.5, index]
        if positive.numel() and negative.numel():
            differences = positive[:, None] - negative[None, :]
            losses.append(F.softplus(-differences).mean())
    if not losses:
        return risk.sum() * 0.0
    return torch.stack(losses).mean()


def train_epoch(
    sleep_model,
    sleep_adapter,
    direct_model,
    adapter,
    dataset,
    label_map,
    config,
    optimizer,
    device,
    epoch,
):
    sleep_model.eval()
    sleep_adapter.eval()
    direct_model.eval()
    adapter.train()
    modalities = tuple(config["data"]["modalities"])
    horizons = tuple(int(value) for value in config["data"]["future_horizons"])
    specs = training_specs(modalities)
    section = config["fusion"]
    totals = {
        "loss": 0.0,
        "physiology": 0.0,
        "event": 0.0,
        "event_ranking": 0.0,
        "gate": 0.0,
    }
    samples = 0
    for batch_index, batch in enumerate(data_loader(dataset, config, shuffle=True)):
        natural_signals = batch["history_signals"].to(device=device, dtype=torch.float32)
        natural_present = batch["history_present"].to(device=device, dtype=torch.bool)
        spec = specs[(batch_index + epoch - 1) % len(specs)]
        signals, present, _ = dynamic_view(
            natural_signals, natural_present, modalities, spec
        )
        with torch.no_grad():
            values = frozen_outputs(
                sleep_model,
                sleep_adapter,
                direct_model,
                signals,
                present,
                horizons,
            )
        sleep_output, sleep_adapted, _, direct_output, _ = values
        output = adapter(
            sleep_output,
            sleep_adapted["stage_logits"],
            sleep_adapted["future_physiology"],
            direct_output["future_physiology"],
        )
        physiology = batch["future_physiology"].to(device=device, dtype=torch.float32)
        physiology_valid = batch["future_physiology_valid"].to(
            device=device, dtype=torch.bool
        )
        event_target = transition_targets(
            batch,
            label_map,
            horizons,
            int(config["data"]["history_epochs"]),
            device,
        )
        physiology_loss = masked_smooth_l1(
            output["future_physiology"], physiology, physiology_valid
        )
        event_loss = F.binary_cross_entropy(
            output["transition_risk"], event_target
        )
        ranking_loss = event_ranking_loss(output["transition_risk"], event_target)

        sleep_error = (sleep_adapted["future_physiology"] - physiology).abs()
        direct_error = (direct_output["future_physiology"] - physiology).abs()
        gate_temperature = float(section.get("gate_target_temperature", 0.15))
        gate_target = torch.sigmoid(
            (sleep_error - direct_error) / max(gate_temperature, 1e-4)
        ).detach()
        gate_errors = F.binary_cross_entropy(
            output["physiology_gate"], gate_target, reduction="none"
        )
        valid_weight = physiology_valid.to(dtype=gate_errors.dtype)
        gate_loss = (gate_errors * valid_weight).sum() / valid_weight.sum().clamp_min(1.0)

        regularizer = output["physiology_residual"].square().mean()
        loss = (
            float(section.get("physiology_weight", 1.0)) * physiology_loss
            + float(section.get("event_weight", 1.0)) * event_loss
            + float(section.get("event_ranking_weight", 0.2)) * ranking_loss
            + float(section.get("gate_supervision_weight", 0.2)) * gate_loss
            + float(section.get("regularizer_weight", 0.001)) * regularizer
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            adapter.parameters(), float(config["train"].get("grad_clip", 1.0))
        )
        optimizer.step()
        batch_size = len(signals)
        samples += batch_size
        for name, value in (
            ("loss", loss),
            ("physiology", physiology_loss),
            ("event", event_loss),
            ("event_ranking", ranking_loss),
            ("gate", gate_loss),
        ):
            totals[name] += float(value.detach().cpu()) * batch_size
    return {name: value / max(samples, 1) for name, value in totals.items()}


@torch.inference_mode()
def evaluate_condition(
    sleep_model,
    sleep_adapter,
    direct_model,
    adapter,
    dataset,
    label_map,
    config,
    device,
    spec: Optional[DynamicObservationSpec],
):
    for module in (sleep_model, sleep_adapter, direct_model, adapter):
        module.eval()
    modalities = tuple(config["data"]["modalities"])
    horizons = tuple(int(value) for value in config["data"]["future_horizons"])
    primary_indices = tuple(horizons.index(value) for value in PRIMARY_HORIZONS)
    routes = {
        name: {"stage": [], "physiology": [], "risk": []}
        for name in ("sleepwm", "direct_grud", "latent_safe")
    }
    labels, physiology, physiology_valid, event_targets = [], [], [], []
    gates = []
    for batch in data_loader(dataset, config, shuffle=False):
        natural_signals = batch["history_signals"].to(device=device, dtype=torch.float32)
        natural_present = batch["history_present"].to(device=device, dtype=torch.bool)
        signals, present, _ = dynamic_view(
            natural_signals, natural_present, modalities, spec
        )
        values = frozen_outputs(
            sleep_model,
            sleep_adapter,
            direct_model,
            signals,
            present,
            horizons,
        )
        sleep_output, sleep_adapted, sleep_risk, direct_output, direct_risk = values
        output = adapter(
            sleep_output,
            sleep_adapted["stage_logits"],
            sleep_adapted["future_physiology"],
            direct_output["future_physiology"],
        )
        routes["sleepwm"]["stage"].append(sleep_adapted["stage_logits"].cpu())
        routes["sleepwm"]["physiology"].append(
            sleep_adapted["future_physiology"].cpu()
        )
        routes["sleepwm"]["risk"].append(sleep_risk.cpu())
        routes["direct_grud"]["stage"].append(direct_output["future_logits"].cpu())
        routes["direct_grud"]["physiology"].append(
            direct_output["future_physiology"].cpu()
        )
        routes["direct_grud"]["risk"].append(direct_risk.cpu())
        routes["latent_safe"]["stage"].append(output["stage_logits"].cpu())
        routes["latent_safe"]["physiology"].append(output["future_physiology"].cpu())
        routes["latent_safe"]["risk"].append(output["transition_risk"].cpu())
        gates.append(output["physiology_gate"].cpu())
        labels.append(batch["future_labels"].cpu())
        physiology.append(batch["future_physiology"].cpu())
        physiology_valid.append(batch["future_physiology_valid"].cpu())
        event_targets.append(
            transition_targets(
                batch,
                label_map,
                horizons,
                int(config["data"]["history_epochs"]),
                torch.device("cpu"),
            )
        )

    label_tensor = torch.cat(labels)
    physiology_tensor = torch.cat(physiology)
    valid_tensor = torch.cat(physiology_valid)
    event_tensor = torch.cat(event_targets).numpy().astype(bool)
    feature_names = tuple(config["physiology"]["feature_names"])
    feature_groups = {
        key: tuple(value) for key, value in config["physiology"]["feature_groups"].items()
    }
    result = {}
    for name, values in routes.items():
        stage = torch.cat(values["stage"])
        predicted_physiology = torch.cat(values["physiology"])
        risk = torch.cat(values["risk"]).numpy()
        phys = standardized_physiology_metrics(
            predicted_physiology,
            physiology_tensor,
            valid_tensor,
            feature_names,
            feature_groups,
            horizons,
        )
        by_horizon = {
            str(horizon): {
                "stage_macro_f1": classification_metrics(
                    stage[:, index], label_tensor[:, index], int(config["data"]["num_classes"])
                )["macro_f1"],
                "physiology_mae": phys["by_horizon"][str(horizon)]["mean_normalized_mae"],
                "transition_auprc": average_precision(event_tensor[:, index], risk[:, index]),
            }
            for index, horizon in enumerate(horizons)
        }
        result[name] = {
            "stage_macro_f1": float(np.mean([by_horizon[str(horizons[i])]["stage_macro_f1"] for i in primary_indices])),
            "future_physiology_mae": float(np.mean([by_horizon[str(horizons[i])]["physiology_mae"] for i in primary_indices])),
            "transition_auprc": float(np.mean([by_horizon[str(horizons[i])]["transition_auprc"] for i in primary_indices])),
            "by_horizon": by_horizon,
        }
    result["latent_safe"]["mean_direct_gate"] = float(torch.cat(gates).mean())
    return result


def evaluate_protocol(
    sleep_model,
    sleep_adapter,
    direct_model,
    adapter,
    dataset,
    config,
    device,
):
    label_map = load_label_map(dataset.dataset if isinstance(dataset, Subset) else dataset)
    conditions = {}
    for spec in evaluation_specs(tuple(config["data"]["modalities"])):
        conditions[spec.name] = evaluate_condition(
            sleep_model,
            sleep_adapter,
            direct_model,
            adapter,
            dataset,
            label_map,
            config,
            device,
            spec,
        )
    aggregate = {}
    for route in ("sleepwm", "direct_grud", "latent_safe"):
        aggregate[route] = {
            metric: float(np.mean([values[route][metric] for values in conditions.values()]))
            for metric in ("stage_macro_f1", "future_physiology_mae", "transition_auprc")
        }
    aggregate["latent_safe"]["mean_direct_gate"] = float(
        np.mean([values["latent_safe"]["mean_direct_gate"] for values in conditions.values()])
    )
    return {"dynamic": aggregate, "conditions": conditions}


def main() -> int:
    args = parse_args()
    protocol = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(args.seed or protocol["experiment"]["seed"])
    po3_path = Path(args.po3_checkpoint or protocol["checkpoints"]["po3"])
    outcome_path = Path(args.outcome_checkpoint or protocol["checkpoints"]["outcome"])
    direct_path = Path(args.direct_checkpoint or protocol["checkpoints"]["direct"])
    po3_checkpoint = load_checkpoint(po3_path)
    outcome_checkpoint = load_checkpoint(outcome_path)
    direct_checkpoint = load_checkpoint(direct_path)
    config = copy.deepcopy(po3_checkpoint["config"])
    for section in ("experiment", "data", "physiology", "fusion", "train"):
        config.setdefault(section, {}).update(protocol.get(section, {}))
    config["experiment"]["seed"] = seed
    if args.output_dir:
        config["train"]["output_dir"] = args.output_dir
    if args.device:
        config["train"]["device"] = args.device
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.smoke:
        config["train"]["epochs"] = 1
        config["train"]["num_workers"] = 0

    seed_everything(seed)
    device = resolve_device(str(config["train"].get("device", "cuda:0")))
    sleep_model = build_student(config).to(device)
    sleep_model.load_state_dict(po3_checkpoint["model_state"], strict=True)
    sleep_model.requires_grad_(False).eval()
    sleep_adapter = build_adapter(sleep_model, outcome_checkpoint["config"]).to(device)
    sleep_adapter.load_state_dict(outcome_checkpoint["adapter_state"], strict=True)
    sleep_adapter.requires_grad_(False).eval()
    direct_model = build_direct_branch(direct_checkpoint["config"]).to(device)
    direct_model.load_state_dict(direct_checkpoint["model_state"], strict=True)
    direct_model.requires_grad_(False).eval()

    section = config["fusion"]
    adapter = LatentHazardSafeAdapter(
        state_dim=sleep_model.encoder.config.d_model,
        modality_count=len(config["data"]["modalities"]),
        num_classes=int(config["data"]["num_classes"]),
        physiology_features=len(config["physiology"]["feature_names"]),
        hidden_dim=int(section.get("hidden_dim", 128)),
        maximum_physiology_residual=float(section.get("maximum_physiology_residual", 0.35)),
        initial_direct_gate=float(section.get("initial_direct_gate", 0.65)),
        dropout=float(section.get("dropout", 0.1)),
    ).to(device)

    train_data = physio_feature_sequence_dataset(config, "train")
    validation_data = physio_feature_sequence_dataset(config, "val")
    if args.smoke:
        train_data = Subset(train_data, range(min(64, len(train_data))))
        validation_data = Subset(validation_data, range(min(64, len(validation_data))))
    train_label_map = load_label_map(
        train_data.dataset if isinstance(train_data, Subset) else train_data
    )
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(section.get("learning_rate", 3e-4)),
        weight_decay=float(config["train"].get("weight_decay", 0.01)),
    )
    output_dir = Path(config["train"]["output_dir"]) / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    initial = evaluate_protocol(
        sleep_model, sleep_adapter, direct_model, adapter, validation_data, config, device
    )
    save_checkpoint(
        output_dir / "best.pt",
        {"epoch": 0, "adapter_state": adapter.state_dict(), "validation": initial, "config": config},
    )
    best_key = (0.0, float("-inf"))
    best_epoch = 0
    curve = []
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        training = train_epoch(
            sleep_model,
            sleep_adapter,
            direct_model,
            adapter,
            train_data,
            train_label_map,
            config,
            optimizer,
            device,
            epoch,
        )
        validation = evaluate_protocol(
            sleep_model, sleep_adapter, direct_model, adapter, validation_data, config, device
        )
        values = validation["dynamic"]
        candidate = values["latent_safe"]
        sleep = values["sleepwm"]
        direct = values["direct_grud"]
        physiology_gain = sleep["future_physiology_mae"] - candidate["future_physiology_mae"]
        event_gain = candidate["transition_auprc"] - sleep["transition_auprc"]
        event_vs_direct = candidate["transition_auprc"] - direct["transition_auprc"]
        decay = validation["conditions"]["linear_decay_all_4ep"]
        decay_margin = (
            decay["sleepwm"]["future_physiology_mae"]
            - decay["latent_safe"]["future_physiology_mae"]
        )
        eligible = (
            physiology_gain > 0.0
            and event_gain > 0.0
            and event_vs_direct >= 0.0
            and decay_margin >= 0.0
        )
        key = (
            1.0 if eligible else 0.0,
            physiology_gain + event_gain + 0.5 * event_vs_direct + 0.25 * decay_margin,
        )
        curve.append(
            {
                "epoch": epoch,
                "training": training,
                "validation": validation,
                "eligible": eligible,
                "decay_margin": decay_margin,
            }
        )
        print(
            f"epoch={epoch:03d} loss={training['loss']:.5f} "
            f"phys={candidate['future_physiology_mae']:.4f} "
            f"event={candidate['transition_auprc']:.4f} "
            f"phys_gain={physiology_gain:+.4f} event_gain={event_gain:+.4f} "
            f"event_vs_direct={event_vs_direct:+.4f} decay_margin={decay_margin:+.4f} "
            f"eligible={eligible}",
            flush=True,
        )
        if key > best_key:
            best_key = key
            best_epoch = epoch
            save_checkpoint(
                output_dir / "best.pt",
                {"epoch": epoch, "adapter_state": adapter.state_dict(), "validation": validation, "config": config},
            )
    selected = load_checkpoint(output_dir / "best.pt")
    adapter.load_state_dict(selected["adapter_state"], strict=True)
    validation = evaluate_protocol(
        sleep_model, sleep_adapter, direct_model, adapter, validation_data, config, device
    )
    result = {
        "seed": seed,
        "best_epoch": int(selected["epoch"]),
        "validation": validation,
        "training_curve": curve,
        "frozen": {"po3": str(po3_path), "outcome": str(outcome_path), "direct": str(direct_path)},
        "test_split_accessed": False,
    }
    if not args.skip_test and not args.smoke:
        test_data = physio_feature_sequence_dataset(config, "test")
        result["test"] = evaluate_protocol(
            sleep_model, sleep_adapter, direct_model, adapter, test_data, config, device
        )
        result["test_split_accessed"] = True
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps({"best_epoch": best_epoch, "validation": validation["dynamic"], "test": result.get("test", {}).get("dynamic")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
