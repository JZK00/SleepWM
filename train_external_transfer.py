from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from evaluate_external_zero_shot import evaluate  # noqa: E402
from uniphysio_wm.data import balanced_class_weights  # noqa: E402
from uniphysio_wm.engine import (  # noqa: E402
    data_loader,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    seed_everything,
    sequence_dataset,
    write_json,
)
from uniphysio_wm.mainline import build_mainline_model  # noqa: E402


STAGE_TRANSFER_PREFIXES = (
    "encoder.",
    "transition.",
    "horizon_embedding",
    "state_predictor.",
    "stage_head.",
    "current_stage_head.",
    "observation_state_adapters.",
    "observation_reliability_heads.",
    "task_stage_residual_heads.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Target-subject transfer for external sleep staging and forecasting.")
    parser.add_argument("--architecture-checkpoint", required=True, help="Stage 7 C1 checkpoint supplying the fixed architecture")
    parser.add_argument("--initialization", choices=("random", "supervised", "uniphysio_wm"), required=True)
    parser.add_argument("--initialization-checkpoint")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--dataset-id", choices=("isruc", "sleep_edfx"), required=True)
    parser.add_argument("--label-fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--horizons", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--sequence-stride", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--current-loss-weight", type=float, default=1.0)
    parser.add_argument("--future-loss-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    return parser.parse_args()


def _subject_rank(subject: str, seed: int) -> str:
    return hashlib.sha1(f"{seed}:{subject}".encode("utf-8")).hexdigest()


def select_subject_fraction(dataset, fraction: float, seed: int):
    if not 0.0 < fraction <= 1.0:
        raise ValueError("label fraction must be in (0, 1]")
    subjects = sorted({str(record["subject"]) for record in dataset.records})
    ordered = sorted(subjects, key=lambda subject: (_subject_rank(subject, seed), subject))
    count = len(subjects) if fraction >= 1.0 else max(1, int(math.ceil(len(subjects) * fraction)))
    selected_subjects = set(ordered[:count])
    selected_record_indices = {
        index
        for index, record in enumerate(dataset.records)
        if str(record["subject"]) in selected_subjects
    }
    indices = [
        index
        for index, (record_index, _) in enumerate(dataset.index)
        if record_index in selected_record_indices
    ]
    if not indices:
        raise ValueError("subject budget selected zero training sequences")
    view = copy.copy(dataset)
    view.index = [dataset.index[index] for index in indices]
    view.transition_mask = [dataset.transition_mask[index] for index in indices]
    view.transition_window = [dataset.transition_window[index] for index in indices]
    return view, sorted(selected_subjects)


def compatible_stage_initialization(model: torch.nn.Module, checkpoint: dict[str, Any]) -> dict[str, Any]:
    source = checkpoint.get("model_state")
    if not isinstance(source, dict):
        raise ValueError("initialization checkpoint lacks model_state")
    target = model.state_dict()
    selected = {
        key: value
        for key, value in source.items()
        if key in target
        and target[key].shape == value.shape
        and key.startswith(STAGE_TRANSFER_PREFIXES)
    }
    if not selected:
        raise ValueError("supervised checkpoint has no compatible stage-path parameters")
    model.load_state_dict(selected, strict=False)
    return {
        "source_parameters": len(source),
        "loaded_parameters": len(selected),
        "loaded_prefix_counts": {
            prefix: sum(key.startswith(prefix) for key in selected)
            for prefix in STAGE_TRANSFER_PREFIXES
        },
    }


def initialize_model(model, mode: str, initialization_checkpoint: str | None) -> dict[str, Any]:
    if mode == "random":
        return {"mode": mode, "checkpoint": None, "loaded_parameters": 0}
    if not initialization_checkpoint:
        raise ValueError(f"{mode} initialization requires --initialization-checkpoint")
    checkpoint = load_checkpoint(initialization_checkpoint)
    if mode == "uniphysio_wm":
        model.load_state_dict(checkpoint["model_state"], strict=True)
        return {
            "mode": mode,
            "checkpoint": str(Path(initialization_checkpoint).resolve()),
            "loaded_parameters": len(checkpoint["model_state"]),
            "strict": True,
        }
    details = compatible_stage_initialization(model, checkpoint)
    return {
        "mode": mode,
        "checkpoint": str(Path(initialization_checkpoint).resolve()),
        "strict": False,
        **details,
    }


def set_stage_path_trainable(model: torch.nn.Module) -> dict[str, int]:
    trainable = 0
    frozen = 0
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(STAGE_TRANSFER_PREFIXES)
        count = parameter.numel()
        if parameter.requires_grad:
            trainable += count
        else:
            frozen += count
    if trainable == 0:
        raise ValueError("stage transfer policy selected zero trainable parameters")
    return {"trainable": trainable, "frozen": frozen}


def selected_class_weights(dataset, selected_subjects: Iterable[str], num_classes: int) -> torch.Tensor:
    selected = set(selected_subjects)
    counts = np.zeros(num_classes, dtype=np.int64)
    for record in dataset.records:
        if str(record["subject"]) not in selected:
            continue
        with np.load(record["path"], allow_pickle=False) as archive:
            labels = archive["labels"].astype(np.int64, copy=False)
        valid = labels[(labels >= 0) & (labels < num_classes)]
        counts += np.bincount(valid, minlength=num_classes)
    return balanced_class_weights(counts)


def stage_loss(output: dict[str, torch.Tensor], batch: dict[str, Any], weights: torch.Tensor, current_weight: float, future_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    current_labels = batch["history_labels"][:, -1].to(output["current_stage_logits"].device)
    future_labels = batch["future_labels"].to(output["stage_logits"].device)
    current = F.cross_entropy(output["current_stage_logits"], current_labels, weight=weights)
    future = F.cross_entropy(
        output["stage_logits"].reshape(-1, output["stage_logits"].shape[-1]),
        future_labels.reshape(-1),
        weight=weights,
    )
    total = current_weight * current + future_weight * future
    return total, {"current": float(current.detach().cpu()), "future": float(future.detach().cpu())}


def train_epoch(model, loader, optimizer, device, horizons, weights, args) -> dict[str, float]:
    model.train()
    total = current = future = 0.0
    batches = 0
    for batch_index, batch in enumerate(loader):
        if args.max_train_batches > 0 and batch_index >= args.max_train_batches:
            break
        signals = batch["history_signals"].to(device, non_blocking=True)
        present = batch["history_present"].to(device, non_blocking=True)
        output = model.rollout_context_horizons(signals, present, horizons)
        loss, parts = stage_loss(
            output,
            batch,
            weights,
            args.current_loss_weight,
            args.future_loss_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            1.0,
        )
        optimizer.step()
        total += float(loss.detach().cpu())
        current += parts["current"]
        future += parts["future"]
        batches += 1
    if batches == 0:
        raise ValueError("training produced zero batches")
    return {"loss": total / batches, "current_loss": current / batches, "future_loss": future / batches}


def selection_score(metrics: dict[str, Any]) -> float:
    current = float(metrics["current_stage"]["macro_f1"])
    future = [float(value["macro_f1"]) for value in metrics["future_stage"]["by_horizon"].values()]
    return float((current + sum(future)) / (1 + len(future)))


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive")
    seed_everything(args.seed)
    architecture = load_checkpoint(args.architecture_checkpoint)
    architecture_config = architecture.get("config")
    if not isinstance(architecture_config, dict):
        raise ValueError("architecture checkpoint lacks resolved config")
    config = copy.deepcopy(architecture_config)
    config["experiment"]["seed"] = int(args.seed)
    config["data"]["manifest_path"] = str(Path(args.manifest).resolve())
    config["data"]["normalization_path"] = str(Path(args.normalization).resolve())
    config["data"]["future_horizons"] = [int(value) for value in args.horizons]
    config["data"]["sequence_stride"] = int(args.sequence_stride)
    config["train"]["batch_size"] = int(args.batch_size)
    config["train"]["num_workers"] = int(args.workers)
    config["train"]["device"] = str(args.device)

    model = build_mainline_model(config)
    initialization = initialize_model(model, args.initialization, args.initialization_checkpoint)
    parameter_policy = set_stage_path_trainable(model)
    device = resolve_device(args.device)
    model.to(device)

    train_data_full = sequence_dataset(config, "train")
    train_data, selected_subjects = select_subject_fraction(train_data_full, args.label_fraction, args.seed)
    validation = sequence_dataset(config, "val")
    train_loader = data_loader(train_data, config, shuffle=True)
    validation_loader = data_loader(validation, config, shuffle=False)
    num_classes = int(config["data"].get("num_classes", 5))
    weights = selected_class_weights(train_data_full, selected_subjects, num_classes).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    stale = 0
    horizons = tuple(int(value) for value in args.horizons)
    for epoch in range(1, args.epochs + 1):
        training = train_epoch(model, train_loader, optimizer, device, horizons, weights, args)
        validation_metrics = evaluate(
            model,
            validation_loader,
            device,
            horizons,
            num_classes,
            args.max_eval_batches,
        )
        score = selection_score(validation_metrics)
        row = {"epoch": epoch, "train": training, "validation_score": score, "validation": validation_metrics}
        history.append(row)
        print(json.dumps({"epoch": epoch, "train": training, "validation_score": score}))
        if score > best_score:
            best_score = score
            stale = 0
            save_checkpoint(
                output_dir / "best.pt",
                {
                    "experiment_id": "S8_external_transfer",
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "config": config,
                    "initialization": initialization,
                    "label_fraction": args.label_fraction,
                    "selected_subjects": selected_subjects,
                    "validation_score": score,
                    "validation_metrics": validation_metrics,
                },
            )
        else:
            stale += 1
            if stale >= args.patience:
                break

    best = load_checkpoint(output_dir / "best.pt")
    model.load_state_dict(best["model_state"], strict=True)
    payload: dict[str, Any] = {
        "protocol": "stage8_sleep_transfer_v1",
        "dataset_id": args.dataset_id,
        "initialization": initialization,
        "label_fraction": args.label_fraction,
        "selected_subject_count": len(selected_subjects),
        "selected_subjects": selected_subjects,
        "train_sequences": len(train_data),
        "validation_sequences": len(validation),
        "parameter_policy": parameter_policy,
        "best_epoch": int(best["epoch"]),
        "best_validation_score": float(best["validation_score"]),
        "best_validation_metrics": best["validation_metrics"],
        "history": history,
        "hmc_cap_test_accessed": False,
    }
    if args.evaluate_test:
        test_data = sequence_dataset(config, "test")
        payload["test_metrics"] = evaluate(
            model,
            data_loader(test_data, config, shuffle=False),
            device,
            horizons,
            num_classes,
            args.max_eval_batches,
        )
    write_json(output_dir / "metrics.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
