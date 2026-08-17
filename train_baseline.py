from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

from uniphysio_wm.config import load_config, with_manifest, with_normalization
from uniphysio_wm.data import balanced_class_weights
from uniphysio_wm.engine import (
    classification_epoch,
    data_loader,
    epoch_dataset,
    model_size,
    prepare_run,
    profile_inference,
    resolve_device,
    save_checkpoint,
    seed_everything,
    write_json,
)
from uniphysio_wm.models import build_supervised_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a supervised sleep-staging baseline.")
    parser.add_argument("--config", required=True, help="baseline YAML configuration")
    parser.add_argument("--manifest", help="override data.manifest_path")
    parser.add_argument("--normalization", help="train-split normalization JSON")
    parser.add_argument("--smoke", action="store_true", help="run one epoch with no worker processes")
    parser.add_argument("--seed", type=int, help="override experiment seed and write to a seed-specific directory")
    parser.add_argument("--epochs", type=int, help="override training epochs")
    parser.add_argument("--output-dir", help="override train.output_dir")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = with_manifest(load_config(args.config), args.manifest)
    config = with_normalization(config, args.normalization)
    if config["experiment"]["kind"] != "baseline":
        raise ValueError("train_baseline.py requires experiment.kind=baseline")
    if args.smoke:
        config = copy.deepcopy(config)
        config["train"]["epochs"] = 1
        config["train"]["num_workers"] = 0
    if args.seed is not None:
        config = copy.deepcopy(config)
        config["experiment"]["seed"] = int(args.seed)
        config["train"]["output_dir"] = str(Path(config["train"]["output_dir"]) / f"seed_{args.seed}")
    if args.epochs is not None:
        config = copy.deepcopy(config)
        config["train"]["epochs"] = int(args.epochs)
    if args.output_dir:
        config = copy.deepcopy(config)
        config["train"]["output_dir"] = str(Path(args.output_dir))

    seed_everything(int(config["experiment"]["seed"]))
    device = resolve_device(str(config["train"].get("device", "cuda")))
    run_dir = prepare_run(config)
    train_data = epoch_dataset(config, "train")
    val_data = epoch_dataset(config, "val")
    train_loader = data_loader(train_data, config, shuffle=True)
    val_loader = data_loader(val_data, config, shuffle=False)

    model = build_supervised_model(config).to(device)
    size_metrics = model_size(model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    class_weights = None
    if config["train"].get("class_weighting") == "balanced":
        class_weights = balanced_class_weights(train_data.label_counts).to(device)

    best_score = -1.0
    best_metrics = {}
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_metrics = classification_epoch(
            model,
            train_loader,
            device,
            optimizer,
            int(config["data"].get("num_classes", 5)),
            class_weights=class_weights,
            grad_clip=float(config["train"].get("grad_clip", 0.0)),
        )
        val_metrics = classification_epoch(
            model,
            val_loader,
            device,
            None,
            int(config["data"].get("num_classes", 5)),
        )
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_macro_f1={val_metrics['macro_f1']:.4f}"
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
                    "model_name": config["model"]["name"],
                    "validation_metrics": val_metrics,
                    "config": config,
                },
            )
    resource_metrics = {
        **size_metrics,
        **profile_inference(model, val_loader, device),
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
        ),
    }
    write_json(run_dir / "metrics.json", {"best_validation": best_metrics, "resources": resource_metrics})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
