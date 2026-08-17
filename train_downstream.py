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
    load_encoder_state,
    model_size,
    prepare_run,
    profile_inference,
    resolve_device,
    save_checkpoint,
    seed_everything,
    trainable_parameters,
    write_json,
)
from uniphysio_wm.models import MultiModalEncoder, SleepStageClassifier, observation_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a sleep-staging probe or fine-tuned model.")
    parser.add_argument("--config", required=True, help="downstream YAML configuration")
    parser.add_argument("--manifest", help="override data.manifest_path")
    parser.add_argument("--normalization", help="train-split normalization JSON")
    parser.add_argument("--checkpoint", help="override downstream.pretrained_checkpoint")
    parser.add_argument("--smoke", action="store_true", help="run one epoch with no worker processes")
    parser.add_argument("--random-init", action="store_true", help="train without loading a pretrained encoder")
    parser.add_argument("--seed", type=int, help="override experiment seed")
    parser.add_argument("--epochs", type=int, help="override training epochs")
    parser.add_argument("--label-fraction", type=float, help="fraction of stratified train labels to use")
    parser.add_argument(
        "--modality-subset-sampling",
        choices=("none", "uniform_nonempty", "full_biased_nonempty"),
        help="override downstream modality subset sampling",
    )
    parser.add_argument(
        "--full-modality-probability",
        type=float,
        help="probability of retaining a full-observation batch with full-biased sampling",
    )
    parser.add_argument("--output-dir", help="override train.output_dir")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = with_manifest(load_config(args.config), args.manifest)
    config = with_normalization(config, args.normalization)
    if config["experiment"]["kind"] != "downstream":
        raise ValueError("train_downstream.py requires experiment.kind=downstream")
    if args.checkpoint:
        config = copy.deepcopy(config)
        config["downstream"]["pretrained_checkpoint"] = args.checkpoint
    if args.seed is not None:
        config = copy.deepcopy(config)
        config["experiment"]["seed"] = int(args.seed)
    if args.epochs is not None:
        config = copy.deepcopy(config)
        config["train"]["epochs"] = int(args.epochs)
    if args.label_fraction is not None:
        if not 0.0 < args.label_fraction <= 1.0:
            raise ValueError("--label-fraction must be in (0, 1]")
        config = copy.deepcopy(config)
        config["data"]["label_fraction"] = float(args.label_fraction)
        config["data"]["label_subset_seed"] = int(config["experiment"]["seed"])
    if args.output_dir:
        config = copy.deepcopy(config)
        config["train"]["output_dir"] = str(Path(args.output_dir))
    if args.modality_subset_sampling:
        config = copy.deepcopy(config)
        config["downstream"]["modality_subset_sampling"] = args.modality_subset_sampling
    if args.full_modality_probability is not None:
        if not 0.0 <= args.full_modality_probability <= 1.0:
            raise ValueError("--full-modality-probability must be in [0, 1]")
        config = copy.deepcopy(config)
        config["downstream"]["full_modality_probability"] = float(args.full_modality_probability)
    if args.smoke:
        config = copy.deepcopy(config)
        config["train"]["epochs"] = 1
        config["train"]["num_workers"] = 0

    seed_everything(int(config["experiment"]["seed"]))
    device = resolve_device(str(config["train"].get("device", "cuda")))
    run_dir = prepare_run(config)
    train_data = epoch_dataset(config, "train")
    val_data = epoch_dataset(config, "val")
    train_loader = data_loader(train_data, config, shuffle=True)
    val_loader = data_loader(val_data, config, shuffle=False)

    encoder = MultiModalEncoder(observation_config(config["data"], config["model"]))
    if not args.random_init:
        load_encoder_state(encoder, config["downstream"]["pretrained_checkpoint"])
    model = SleepStageClassifier(
        encoder,
        num_classes=int(config["data"].get("num_classes", 5)),
        hidden_dim=int(config["model"].get("hidden_dim", 0)),
        freeze_encoder=bool(config["downstream"].get("freeze_encoder", False)),
    ).to(device)
    size_metrics = model_size(model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    parameters = list(trainable_parameters(model))
    optimizer = torch.optim.AdamW(
        parameters,
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
            modality_subset_sampling=str(config["downstream"].get("modality_subset_sampling", "none")),
            full_modality_probability=float(
                config["downstream"].get("full_modality_probability", 0.5)
            ),
        )
        val_metrics = classification_epoch(
            model, val_loader, device, None, int(config["data"].get("num_classes", 5))
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
                    "encoder_state": model.encoder.state_dict(),
                    "validation_metrics": val_metrics,
                    "config": config,
                    "initialization": "random" if args.random_init else "pretrained",
                },
            )
    resource_metrics = {
        **size_metrics,
        **profile_inference(model, val_loader, device),
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
        ),
    }
    write_json(
        run_dir / "metrics.json",
        {
            "best_validation": best_metrics,
            "resources": resource_metrics,
            "initialization": "random" if args.random_init else "pretrained",
            "label_fraction": float(config["data"].get("label_fraction", 1.0)),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
