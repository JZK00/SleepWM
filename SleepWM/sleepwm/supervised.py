"""Train SleepWM from random initialization on your own prepared dataset."""
import argparse
import json
from pathlib import Path

import torch

from sleepwm import SleepWM, default_config
from sleepwm.runner import (CLAIM_SCOPE, CONDITIONS, evaluate_model, make_loader,
                            make_training_datasets, read_statistics, save_checkpoint,
                            set_seed, train_epoch, write_json)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--statistics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compact", action="store_true", help="Small architecture for workflow checks")
    parser.add_argument("--history-epochs", type=int, default=20)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--validation-conditions", choices=CONDITIONS, nargs="+", default=list(CONDITIONS))
    parser.add_argument("--gap-epochs", type=int, default=4)
    parser.add_argument("--max-batches", type=int, help="SMOKE TEST ONLY: limit each train/validation pass")
    args = parser.parse_args(argv)
    for name in ("epochs", "batch_size", "history_epochs", "stride", "gap_epochs"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.history_epochs < 2:
        parser.error("--history-epochs must be at least 2")
    if args.learning_rate <= 0 or args.horizons != sorted(set(args.horizons)) or min(args.horizons) < 1:
        parser.error("Learning rate must be positive; horizons must be unique positive increasing integers")
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive")
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {output}")
    set_seed(args.seed)
    statistics, statistics_sha = read_statistics(args.statistics)
    config = default_config(compact=args.compact)
    config["data"]["history_epochs"] = args.history_epochs
    config["data"]["future_horizons"] = args.horizons
    config["recursive"]["direct_horizons"] = list(args.horizons)
    train_data, val_data = make_training_datasets(args.manifest, args.statistics, config, args.stride)
    if not len(train_data) or not len(val_data):
        raise ValueError("Both train and val need eligible continuous windows; test is never used for training")
    train_loader = make_loader(train_data, args.batch_size, seed=args.seed)
    val_loader = make_loader(val_data, args.batch_size)
    device = torch.device(args.device)
    model = SleepWM(config).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    initial = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
               if parameter.requires_grad}
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "claim_scope": CLAIM_SCOPE, "training_mode": "supervised",
        "smoke_test": args.max_batches is not None,
        "compact_architecture": args.compact, "seed": args.seed,
        "history_epochs": args.history_epochs, "horizons_epochs": args.horizons,
        "stride": args.stride, "validation_conditions": args.validation_conditions,
        "gap_epochs": args.gap_epochs, "max_batches": args.max_batches,
        "train_eligible_windows": len(train_data), "val_eligible_windows": len(val_data),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "selection": "Minimum mean validation CE + masked physiology Huber + event BCE across selected conditions",
        "normalization_sha256": statistics_sha,
        "arguments": vars(args),
    }
    best_loss, history = float("inf"), []
    print(CLAIM_SCOPE, flush=True)
    if args.max_batches is not None:
        print("SMOKE TEST: limited batches; scores are not a performance benchmark.", flush=True)
    for epoch in range(1, args.epochs + 1):
        training = train_epoch(model, train_loader, optimizer, device, args.max_batches)
        validation = evaluate_model(model, val_loader, device, args.horizons,
                                    args.validation_conditions, args.gap_epochs, args.max_batches)
        val_loss = sum(item["loss"] for item in validation.values()) / len(validation)
        row = {"epoch": epoch, "training": training, "validation_loss": val_loss,
               "validation": validation}
        history.append(row)
        changed = any(not torch.equal(initial[name], parameter.detach().cpu())
                      for name, parameter in model.named_parameters() if parameter.requires_grad)
        if not changed:
            raise RuntimeError("Training did not update any model parameters")
        if val_loss < best_loss:
            best_loss = val_loss
            metadata.update(best_epoch=epoch, best_validation_loss=val_loss,
                            parameters_updated=changed, train_windows_last_epoch=training["windows"])
            save_checkpoint(output / "best.pt", model, config, statistics, statistics_sha, metadata.copy())
        write_json(output / "history.json", {"metadata": metadata, "epochs": history})
        print(json.dumps({"epoch": epoch, "train_loss": training["loss"], "val_loss": val_loss,
                          "windows": training["windows"], "parameters_updated": changed}), flush=True)
    print(f"Saved best validation checkpoint: {output / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
