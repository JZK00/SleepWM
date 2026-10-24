"""Evaluate a custom-data checkpoint with its frozen training normalization."""
import argparse
from pathlib import Path

import torch

from sleepwm.data import SleepDataset
from sleepwm.runner import (CLAIM_SCOPE, CONDITIONS, evaluate_model, load_checkpoint,
                            make_loader, write_json)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--statistics", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val",
                        help="test must be explicitly requested; default is val")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--conditions", choices=CONDITIONS, nargs="+", default=list(CONDITIONS))
    parser.add_argument("--gap-epochs", type=int, default=4)
    parser.add_argument("--max-batches", type=int, help="SMOKE TEST ONLY: limit each condition")
    args = parser.parse_args(argv)
    if min(args.batch_size, args.stride, args.gap_epochs) < 1:
        parser.error("batch-size, stride, and gap-epochs must be positive")
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {output}")
    device = torch.device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, args.statistics, device)
    config = checkpoint["model_config"]
    horizons = config["data"]["future_horizons"]
    dataset = SleepDataset(args.manifest, args.statistics, args.split,
                           history_epochs=config["data"]["history_epochs"],
                           horizons=horizons, stride=args.stride)
    report = evaluate_model(model, make_loader(dataset, args.batch_size), device, horizons,
                            args.conditions, args.gap_epochs, args.max_batches)
    payload = {
        "claim_scope": checkpoint.get("claim_scope", CLAIM_SCOPE),
        "training_mode": checkpoint["metadata"].get("training_mode", "supervised"),
        "split": args.split, "eligible_windows": len(dataset),
        "smoke_test": args.max_batches is not None or checkpoint["metadata"].get("smoke_test", False),
        "checkpoint_training": checkpoint["metadata"],
        "conditions": report, "gap_epochs": args.gap_epochs, "stride": args.stride,
        "max_batches": args.max_batches,
        "metric_note": "Window-weighted metrics. Complete preserves native missingness; other conditions add tail masking. AP is null with no positive targets. Physiology MAE uses valid standardized targets only.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, payload)
    print(f"Saved {args.split} evaluation: {output}", flush=True)


if __name__ == "__main__":
    main()
