from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset



from sleepwm.readouts import DynamicEventHazardAdapter
from sleepwm.readouts import build_direct_branch, observation_age
from sleepwm.protocols import dynamic_view
from sleepwm.protocols import average_precision, load_label_map
from sleepwm.readouts import LatentHazardSafeAdapter
from sleepwm.readouts import ReliabilityGatedEventCorrection
from sleepwm.training.latent_hazard_safe_adapter import evaluation_specs, event_ranking_loss, frozen_outputs, training_specs, transition_targets
from sleepwm.training.recursive_belief_filter import build_student
from sleepwm.training.trajectory_outcome_adapter import build_adapter
from sleepwm.engine import data_loader, load_checkpoint, physio_feature_sequence_dataset, resolve_device, save_checkpoint, seed_everything


PRIMARY_HORIZONS = (1, 2, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--po17-checkpoint", required=True)
    parser.add_argument("--po3-checkpoint", required=True)
    parser.add_argument("--outcome-checkpoint", required=True)
    parser.add_argument("--direct-checkpoint", required=True)
    parser.add_argument("--direct-event-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-test", action="store_true", default=True)
    parser.add_argument("--evaluate-test", dest="skip_test", action="store_false", help="Explicitly enable held-out test evaluation")
    return parser.parse_args()


def build_po17(config: dict, sleep_model) -> LatentHazardSafeAdapter:
    section = config["fusion"]
    return LatentHazardSafeAdapter(
        state_dim=sleep_model.encoder.config.d_model,
        modality_count=len(config["data"]["modalities"]),
        num_classes=int(config["data"]["num_classes"]),
        physiology_features=len(config["physiology"]["feature_names"]),
        hidden_dim=int(section.get("hidden_dim", 128)),
        maximum_physiology_residual=float(
            section.get("maximum_physiology_residual", 0.35)
        ),
        initial_direct_gate=float(section.get("initial_direct_gate", 0.65)),
        dropout=float(section.get("dropout", 0.1)),
    )


def frozen_event_outputs(
    components: dict,
    signals: torch.Tensor,
    present: torch.Tensor,
    horizons: tuple[int, ...],
) -> tuple[dict, dict, dict]:
    sleep_output, sleep_adapted, _, direct_output, _ = frozen_outputs(
        components["sleep_model"],
        components["sleep_adapter"],
        components["direct_model"],
        signals,
        present,
        horizons,
    )
    po17_output = components["po17_adapter"](
        sleep_output,
        sleep_adapted["stage_logits"],
        sleep_adapted["future_physiology"],
        direct_output["future_physiology"],
    )
    direct_event = components["direct_event_adapter"](
        direct_output,
        present,
        observation_age(present).to(signals.device),
        torch.tensor(horizons, device=signals.device),
    )
    return sleep_output, po17_output, direct_event


def train_epoch(
    components: dict,
    correction,
    dataset,
    label_map,
    config,
    optimizer,
    device,
    epoch: int,
) -> dict[str, float]:
    correction.train()
    modalities = tuple(config["data"]["modalities"])
    horizons = tuple(int(value) for value in config["data"]["future_horizons"])
    specs = training_specs(modalities)
    totals = {"loss": 0.0, "bce": 0.0, "ranking": 0.0, "gate": 0.0}
    samples = 0
    for batch_index, batch in enumerate(data_loader(dataset, config, shuffle=True)):
        signals = batch["history_signals"].to(device=device, dtype=torch.float32)
        present = batch["history_present"].to(device=device, dtype=torch.bool)
        spec = specs[(batch_index + epoch - 1) % len(specs)]
        signals, present, _ = dynamic_view(signals, present, modalities, spec)
        with torch.no_grad():
            sleep_output, po17_output, direct_event = frozen_event_outputs(
                components, signals, present, horizons
            )
        output = correction(
            sleep_output,
            po17_output["stage_logits"],
            po17_output["interval_hazard"],
            direct_event["interval_hazard"],
        )
        target = transition_targets(
            batch,
            label_map,
            horizons,
            int(config["data"]["history_epochs"]),
            device,
        )
        bce = F.binary_cross_entropy(output["transition_risk"], target)
        ranking = event_ranking_loss(output["transition_risk"], target)
        residual_penalty = output["logit_residual"].square().mean()
        loss = bce + 0.2 * ranking + 0.001 * residual_penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(correction.parameters(), 1.0)
        optimizer.step()
        batch_size = len(signals)
        samples += batch_size
        for key, value in (
            ("loss", loss),
            ("bce", bce),
            ("ranking", ranking),
            ("gate", output["direct_gate"].mean()),
        ):
            totals[key] += float(value.detach().cpu()) * batch_size
    return {key: value / max(samples, 1) for key, value in totals.items()}


@torch.inference_mode()
def evaluate_condition(
    components: dict,
    correction,
    dataset,
    label_map,
    config,
    device,
    spec: Optional[object],
) -> dict:
    correction.eval()
    modalities = tuple(config["data"]["modalities"])
    horizons = tuple(int(value) for value in config["data"]["future_horizons"])
    risks, targets, gates = [], [], []
    for batch in data_loader(dataset, config, shuffle=False):
        signals = batch["history_signals"].to(device=device, dtype=torch.float32)
        present = batch["history_present"].to(device=device, dtype=torch.bool)
        signals, present, _ = dynamic_view(signals, present, modalities, spec)
        sleep_output, po17_output, direct_event = frozen_event_outputs(
            components, signals, present, horizons
        )
        output = correction(
            sleep_output,
            po17_output["stage_logits"],
            po17_output["interval_hazard"],
            direct_event["interval_hazard"],
        )
        risks.append(output["transition_risk"].cpu())
        gates.append(output["direct_gate"].cpu())
        targets.append(
            transition_targets(
                batch,
                label_map,
                horizons,
                int(config["data"]["history_epochs"]),
                torch.device("cpu"),
            )
        )
    risk = torch.cat(risks).numpy()
    target = torch.cat(targets).numpy().astype(bool)
    by_horizon = {
        str(horizon): average_precision(
            target[:, index], risk[:, index]
        )
        for index, horizon in enumerate(horizons)
        if horizon in PRIMARY_HORIZONS
    }
    return {
        "transition_auprc": float(np.mean(list(by_horizon.values()))),
        "by_horizon": by_horizon,
        "mean_direct_gate": float(torch.cat(gates).mean()),
    }


def evaluate_protocol(components, correction, dataset, config, device) -> dict:
    source = dataset.dataset if isinstance(dataset, Subset) else dataset
    label_map = load_label_map(source)
    conditions = {
        spec.name: evaluate_condition(
            components, correction, dataset, label_map, config, device, spec
        )
        for spec in evaluation_specs(tuple(config["data"]["modalities"]))
    }
    return {
        "dynamic": {
            "transition_auprc": float(
                np.mean([value["transition_auprc"] for value in conditions.values()])
            ),
            "mean_direct_gate": float(
                np.mean([value["mean_direct_gate"] for value in conditions.values()])
            ),
        },
        "conditions": conditions,
    }


def main() -> int:
    args = parse_args()
    po17_checkpoint = load_checkpoint(Path(args.po17_checkpoint))
    config = copy.deepcopy(po17_checkpoint["config"])
    config["experiment"]["seed"] = args.seed
    config["train"]["device"] = args.device
    config["train"]["num_workers"] = min(
        int(config["train"].get("num_workers", 2)), 2
    )
    if args.smoke:
        config["train"]["num_workers"] = 0
        args.epochs = 1
    seed_everything(args.seed)
    device = resolve_device(args.device)

    po3_checkpoint = load_checkpoint(Path(args.po3_checkpoint))
    outcome_checkpoint = load_checkpoint(Path(args.outcome_checkpoint))
    direct_checkpoint = load_checkpoint(Path(args.direct_checkpoint))
    direct_event_checkpoint = load_checkpoint(Path(args.direct_event_checkpoint))

    sleep_model = build_student(config).to(device)
    sleep_model.load_state_dict(po3_checkpoint["model_state"], strict=True)
    sleep_adapter = build_adapter(sleep_model, outcome_checkpoint["config"]).to(device)
    sleep_adapter.load_state_dict(outcome_checkpoint["adapter_state"], strict=True)
    direct_model = build_direct_branch(direct_checkpoint["config"]).to(device)
    direct_model.load_state_dict(direct_checkpoint["model_state"], strict=True)
    po17_adapter = build_po17(config, sleep_model).to(device)
    po17_adapter.load_state_dict(po17_checkpoint["adapter_state"], strict=True)
    direct_event_adapter = DynamicEventHazardAdapter(
        state_dim=int(direct_checkpoint["config"]["model"]["hidden_dim"]),
        modality_count=len(config["data"]["modalities"]),
        num_classes=int(config["data"]["num_classes"]),
        physiology_features=len(config["physiology"]["feature_names"]),
        hidden_dim=int(direct_checkpoint["config"]["model"]["hidden_dim"]),
        dropout=float(direct_checkpoint["config"]["model"].get("dropout", 0.1)),
    ).to(device)
    direct_event_adapter.load_state_dict(
        direct_event_checkpoint["adapter_state"], strict=True
    )
    components = {
        "sleep_model": sleep_model,
        "sleep_adapter": sleep_adapter,
        "direct_model": direct_model,
        "po17_adapter": po17_adapter,
        "direct_event_adapter": direct_event_adapter,
    }
    for module in components.values():
        module.requires_grad_(False).eval()

    correction = ReliabilityGatedEventCorrection(
        modality_count=len(config["data"]["modalities"]),
        num_classes=int(config["data"]["num_classes"]),
    ).to(device)
    train_data = physio_feature_sequence_dataset(config, "train")
    validation_data = physio_feature_sequence_dataset(config, "val")
    if args.smoke:
        train_data = Subset(train_data, range(min(96, len(train_data))))
        validation_data = Subset(validation_data, range(min(96, len(validation_data))))
    train_source = train_data.dataset if isinstance(train_data, Subset) else train_data
    train_labels = load_label_map(train_source)
    optimizer = torch.optim.AdamW(
        correction.parameters(), lr=3e-4, weight_decay=0.01
    )
    output_dir = Path(args.output_dir) / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score = float("-inf")
    curve = []
    for epoch in range(1, args.epochs + 1):
        training = train_epoch(
            components,
            correction,
            train_data,
            train_labels,
            config,
            optimizer,
            device,
            epoch,
        )
        validation = evaluate_protocol(
            components, correction, validation_data, config, device
        )
        score = validation["dynamic"]["transition_auprc"]
        curve.append({"epoch": epoch, "training": training, "validation": validation})
        print(
            f"epoch={epoch:03d} loss={training['loss']:.5f} "
            f"event={score:.4f} gate={validation['dynamic']['mean_direct_gate']:.3f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            save_checkpoint(
                output_dir / "best.pt",
                {
                    "epoch": epoch,
                    "adapter_state": correction.state_dict(),
                    "config": config,
                    "validation": validation,
                },
            )
    selected = load_checkpoint(output_dir / "best.pt")
    correction.load_state_dict(selected["adapter_state"], strict=True)
    result = {
        "seed": args.seed,
        "best_epoch": int(selected["epoch"]),
        "validation": evaluate_protocol(
            components, correction, validation_data, config, device
        ),
        "training_curve": curve,
        "test_split_accessed": False,
    }
    if not args.skip_test and not args.smoke:
        test_data = physio_feature_sequence_dataset(config, "test")
        result["test"] = evaluate_protocol(
            components, correction, test_data, config, device
        )
        result["test_split_accessed"] = True
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "best_epoch": result["best_epoch"],
                "validation": result["validation"]["dynamic"],
                "test": result.get("test", {}).get("dynamic"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
