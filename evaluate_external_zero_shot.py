from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from uniphysio_wm.engine import data_loader, load_checkpoint, resolve_device, sequence_dataset, write_json  # noqa: E402
from uniphysio_wm.mainline import build_mainline_model  # noqa: E402
from uniphysio_wm.metrics import classification_metrics, forecast_subgroup_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the frozen Stage 7 mainline on an external sleep dataset.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--dataset-id", choices=("isruc", "sleep_edfx"), required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--horizons", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sequence-stride", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-batches", type=int, default=0)
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calibration_metrics(logits: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> dict[str, float]:
    probabilities = logits.float().softmax(dim=-1)
    labels = labels.long()
    confidence, prediction = probabilities.max(dim=-1)
    correct = prediction.eq(labels).float()
    edges = torch.linspace(0.0, 1.0, bins + 1, device=logits.device)
    ece = logits.new_tensor(0.0, dtype=torch.float32)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            ece = ece + selected.float().mean() * (
                correct[selected].mean() - confidence[selected].mean()
            ).abs()
    one_hot = F.one_hot(labels, num_classes=probabilities.shape[-1]).float()
    return {
        "ece_15": float(ece.cpu()),
        "brier": float((probabilities - one_hot).pow(2).sum(dim=-1).mean().cpu()),
        "nll": float(F.cross_entropy(logits.float(), labels).cpu()),
    }


def predicted_persistence_metrics(
    current_logits: torch.Tensor,
    future_labels: torch.Tensor,
    horizons: Sequence[int],
    num_classes: int,
) -> dict[str, Any]:
    repeated = current_logits.unsqueeze(1).expand(-1, len(horizons), -1)
    return {
        "all_horizons": classification_metrics(repeated, future_labels, num_classes),
        "by_horizon": {
            str(horizon): classification_metrics(repeated[:, index], future_labels[:, index], num_classes)
            for index, horizon in enumerate(horizons)
        },
    }


@torch.inference_mode()
def evaluate(model, loader, device: torch.device, horizons: Sequence[int], num_classes: int, max_batches: int) -> dict[str, Any]:
    current_logits: list[torch.Tensor] = []
    current_labels: list[torch.Tensor] = []
    future_logits: list[torch.Tensor] = []
    future_labels: list[torch.Tensor] = []
    modality_patterns: Counter[str] = Counter()
    record_ids: set[str] = set()

    model.eval()
    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        signals = batch["history_signals"].to(device, non_blocking=True)
        present = batch["history_present"].to(device, non_blocking=True)
        output = model.rollout_context_horizons(signals, present, horizons)
        current_logits.append(output["current_stage_logits"].detach().cpu())
        future_logits.append(output["stage_logits"].detach().cpu())
        current_labels.append(batch["history_labels"][:, -1].cpu())
        future_labels.append(batch["future_labels"].cpu())
        record_ids.update(str(value) for value in batch["record_id"])
        availability = present.any(dim=1).detach().cpu().numpy()
        for row in availability:
            modality_patterns["+".join(name for name, value in zip(("EEG", "ECG", "EMG"), row) if value)] += 1

    if not current_logits:
        raise ValueError("evaluation produced zero batches")
    current_logits_tensor = torch.cat(current_logits)
    current_labels_tensor = torch.cat(current_labels)
    future_logits_tensor = torch.cat(future_logits)
    future_labels_tensor = torch.cat(future_labels)
    current = classification_metrics(current_logits_tensor, current_labels_tensor, num_classes)
    forecast = {
        "all_horizons": classification_metrics(future_logits_tensor, future_labels_tensor, num_classes),
        "by_horizon": {
            str(horizon): classification_metrics(
                future_logits_tensor[:, index], future_labels_tensor[:, index], num_classes
            )
            for index, horizon in enumerate(horizons)
        },
        "subgroups": forecast_subgroup_metrics(
            future_logits_tensor,
            future_labels_tensor,
            current_labels_tensor,
            horizons,
            num_classes,
        ),
    }
    calibration = {
        "current": calibration_metrics(current_logits_tensor, current_labels_tensor),
        "by_horizon": {
            str(horizon): calibration_metrics(future_logits_tensor[:, index], future_labels_tensor[:, index])
            for index, horizon in enumerate(horizons)
        },
    }
    return {
        "records_evaluated": len(record_ids),
        "sequences": int(current_labels_tensor.shape[0]),
        "modality_pattern_sequences": dict(modality_patterns),
        "current_stage": current,
        "future_stage": forecast,
        "predicted_current_persistence": predicted_persistence_metrics(
            current_logits_tensor, future_labels_tensor, horizons, num_classes
        ),
        "calibration": calibration,
    }


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    manifest_path = Path(args.manifest).resolve()
    normalization_path = Path(args.normalization).resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError("checkpoint does not contain its resolved config")

    config = copy.deepcopy(checkpoint_config)
    config["data"]["manifest_path"] = str(manifest_path)
    config["data"]["normalization_path"] = str(normalization_path)
    config["data"]["future_horizons"] = [int(value) for value in args.horizons]
    config["data"]["sequence_stride"] = int(args.sequence_stride)
    config["train"]["batch_size"] = int(args.batch_size)
    config["train"]["num_workers"] = int(args.workers)
    config["train"]["device"] = str(args.device)

    model = build_mainline_model(config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = resolve_device(str(args.device))
    model.to(device)
    dataset = sequence_dataset(config, args.split)
    loader = data_loader(dataset, config, shuffle=False)
    metrics = evaluate(
        model,
        loader,
        device,
        tuple(int(value) for value in args.horizons),
        int(config["data"].get("num_classes", 5)),
        int(args.max_batches),
    )
    payload = {
        "protocol": "stage8_sleep_transfer_v1",
        "evaluation": "frozen_zero_shot_external",
        "dataset_id": args.dataset_id,
        "split": args.split,
        "seed": int(config["experiment"]["seed"]),
        "no_target_parameter_updates": True,
        "horizons_epochs": [int(value) for value in args.horizons],
        "horizons_seconds": [int(value) * int(config["data"]["epoch_seconds"]) for value in args.horizons],
        "history_epochs": int(config["data"]["history_epochs"]),
        "sequence_stride": int(args.sequence_stride),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "normalization": str(normalization_path),
        "normalization_sha256": sha256_file(normalization_path),
        "hmc_cap_test_accessed": False,
        "metrics": metrics,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
