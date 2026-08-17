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


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from dynamic_event_hazard_adapter import DynamicEventHazardAdapter
from dynamic_missing_baselines import build_dynamic_baseline, observation_age
from evaluate_sleep_events import average_precision, load_label_map, next_event_offset
from train_dynamic_missing_baseline import apply_spec, dynamic_specs, training_specs
from uniphysio_wm.engine import (
    data_loader,
    load_checkpoint,
    physio_feature_sequence_dataset,
    resolve_device,
    save_checkpoint,
    seed_everything,
)
from uniphysio_wm.partial_observation import DynamicObservationSpec


PRIMARY_HORIZONS = (1, 2, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    return parser.parse_args()


def transition_targets(
    batch,
    label_map: dict[str, np.ndarray],
    horizons: tuple[int, ...],
    history_epochs: int,
    device: torch.device,
) -> torch.Tensor:
    origins = (batch["start_epoch"] + history_epochs - 1).tolist()
    maximum = max(horizons)
    offsets = [
        next_event_offset(label_map[str(record)], int(origin), "transition", maximum)
        for record, origin in zip(batch["record_id"], origins)
    ]
    return torch.tensor(
        [[offset <= horizon for horizon in horizons] for offset in offsets],
        device=device,
        dtype=torch.float32,
    )


def ranking_loss(risk: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    losses = []
    for index in range(risk.shape[1]):
        positive = risk[target[:, index] > 0.5, index]
        negative = risk[target[:, index] <= 0.5, index]
        if positive.numel() and negative.numel():
            losses.append(F.softplus(-(positive[:, None] - negative[None, :])).mean())
    return risk.sum() * 0.0 if not losses else torch.stack(losses).mean()


def train_epoch(
    model,
    adapter,
    dataset,
    label_map,
    config,
    optimizer,
    device,
    epoch,
):
    model.eval()
    adapter.train()
    modalities = tuple(config["data"]["modalities"])
    horizons = tuple(int(value) for value in config["data"]["future_horizons"])
    horizon_tensor = torch.tensor(horizons, device=device)
    specs = training_specs(modalities)
    total_loss = 0.0
    samples = 0
    for batch_index, batch in enumerate(data_loader(dataset, config, shuffle=True)):
        signals = batch["history_signals"].to(device=device, dtype=torch.float32)
        present = batch["history_present"].to(device=device, dtype=torch.bool)
        spec = specs[(batch_index + epoch) % len(specs)]
        signals, present = apply_spec(signals, present, modalities, spec)
        with torch.no_grad():
            baseline_output = model(signals, present)
        output = adapter(
            baseline_output,
            present,
            observation_age(present).to(device),
            horizon_tensor,
        )
        target = transition_targets(
            batch,
            label_map,
            horizons,
            int(config["data"]["history_epochs"]),
            device,
        )
        loss = F.binary_cross_entropy(output["transition_risk"], target)
        loss = loss + 0.2 * ranking_loss(output["transition_risk"], target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        batch_size = len(signals)
        total_loss += float(loss.detach().cpu()) * batch_size
        samples += batch_size
    return total_loss / max(samples, 1)


@torch.inference_mode()
def evaluate_condition(
    model,
    adapter,
    dataset,
    label_map,
    config,
    device,
    spec: Optional[DynamicObservationSpec],
):
    model.eval()
    adapter.eval()
    modalities = tuple(config["data"]["modalities"])
    horizons = tuple(int(value) for value in config["data"]["future_horizons"])
    horizon_tensor = torch.tensor(horizons, device=device)
    risks = []
    targets = []
    for batch in data_loader(dataset, config, shuffle=False):
        signals = batch["history_signals"].to(device=device, dtype=torch.float32)
        present = batch["history_present"].to(device=device, dtype=torch.bool)
        signals, present = apply_spec(signals, present, modalities, spec)
        baseline_output = model(signals, present)
        output = adapter(
            baseline_output,
            present,
            observation_age(present).to(device),
            horizon_tensor,
        )
        risks.append(output["transition_risk"].cpu())
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
            target[:, horizons.index(horizon)], risk[:, horizons.index(horizon)]
        )
        for horizon in PRIMARY_HORIZONS
    }
    return {
        "transition_auprc": float(np.mean(list(by_horizon.values()))),
        "by_horizon": by_horizon,
    }


def evaluate_protocol(model, adapter, dataset, config, device):
    source = dataset.dataset if isinstance(dataset, Subset) else dataset
    label_map = load_label_map(source)
    conditions = {
        spec.name: evaluate_condition(
            model, adapter, dataset, label_map, config, device, spec
        )
        for spec in dynamic_specs(tuple(config["data"]["modalities"]))
    }
    return {
        "dynamic": {
            "transition_auprc": float(
                np.mean([value["transition_auprc"] for value in conditions.values()])
            )
        },
        "conditions": conditions,
    }


def main() -> int:
    args = parse_args()
    checkpoint = load_checkpoint(Path(args.baseline_checkpoint))
    config = copy.deepcopy(checkpoint["config"])
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
    model = build_dynamic_baseline(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.requires_grad_(False).eval()
    adapter = DynamicEventHazardAdapter(
        state_dim=int(config["model"]["hidden_dim"]),
        modality_count=len(config["data"]["modalities"]),
        num_classes=int(config["data"]["num_classes"]),
        physiology_features=len(config["physiology"]["feature_names"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
        dropout=float(config["model"].get("dropout", 0.1)),
    ).to(device)
    train_data = physio_feature_sequence_dataset(config, "train")
    validation_data = physio_feature_sequence_dataset(config, "val")
    if args.smoke:
        train_data = Subset(train_data, range(min(96, len(train_data))))
        validation_data = Subset(validation_data, range(min(96, len(validation_data))))
    train_source = train_data.dataset if isinstance(train_data, Subset) else train_data
    train_labels = load_label_map(train_source)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=3e-4, weight_decay=0.01)
    output_dir = Path(args.output_dir) / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score = float("-inf")
    curve = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            model,
            adapter,
            train_data,
            train_labels,
            config,
            optimizer,
            device,
            epoch,
        )
        validation = evaluate_protocol(model, adapter, validation_data, config, device)
        score = validation["dynamic"]["transition_auprc"]
        curve.append({"epoch": epoch, "loss": loss, "validation": validation})
        print(f"epoch={epoch:03d} loss={loss:.5f} event_auprc={score:.4f}", flush=True)
        if score > best_score:
            best_score = score
            save_checkpoint(
                output_dir / "best.pt",
                {
                    "epoch": epoch,
                    "adapter_state": adapter.state_dict(),
                    "config": config,
                    "validation": validation,
                },
            )
    selected = load_checkpoint(output_dir / "best.pt")
    adapter.load_state_dict(selected["adapter_state"], strict=True)
    result = {
        "architecture": config["model"]["architecture"],
        "seed": args.seed,
        "best_epoch": int(selected["epoch"]),
        "validation": evaluate_protocol(model, adapter, validation_data, config, device),
        "training_curve": curve,
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "test_split_accessed": False,
    }
    if not args.skip_test and not args.smoke:
        test_data = physio_feature_sequence_dataset(config, "test")
        result["test"] = evaluate_protocol(model, adapter, test_data, config, device)
        result["test_split_accessed"] = True
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"best_epoch": result["best_epoch"], "validation": result["validation"]["dynamic"], "test": result.get("test", {}).get("dynamic")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
