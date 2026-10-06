from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np



from sleepwm.data import read_manifest, resolve_record_path, validate_subject_splits


def parse_normalize() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute modality statistics from train subjects only.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--modalities", nargs="+", default=["EEG", "ECG", "EMG"])
    parser.add_argument("--chunk-epochs", type=int, default=32)
    return parser.parse_args()


def run_normalize() -> int:
    args = parse_normalize()
    manifest = Path(args.manifest).resolve()
    rows = read_manifest(manifest)
    validate_subject_splits(rows)
    train_rows = [row for row in rows if row["split"] == "train"]
    if not train_rows:
        raise ValueError("manifest contains no train recordings")

    modality_count = len(args.modalities)
    total = np.zeros(modality_count, dtype=np.float64)
    total_square = np.zeros(modality_count, dtype=np.float64)
    count = np.zeros(modality_count, dtype=np.int64)

    for record_index, row in enumerate(train_rows, start=1):
        path = resolve_record_path(manifest, row["npz_path"])
        with np.load(path, allow_pickle=False) as archive:
            signals = archive["signals"]
            if signals.ndim != 3 or signals.shape[1] != modality_count:
                raise ValueError(f"unexpected signals shape in {path}: {signals.shape}")
            if "modality_present" in archive:
                present = archive["modality_present"].astype(bool, copy=False)
            else:
                present = np.ones(signals.shape[:2], dtype=bool)
            for start in range(0, signals.shape[0], args.chunk_epochs):
                chunk = signals[start : start + args.chunk_epochs]
                chunk_present = present[start : start + args.chunk_epochs]
                for modality in range(modality_count):
                    values = chunk[chunk_present[:, modality], modality]
                    if values.size == 0:
                        continue
                    total[modality] += values.sum(dtype=np.float64)
                    total_square[modality] += np.square(values).sum(dtype=np.float64)
                    count[modality] += values.size
        print(f"[{record_index}/{len(train_rows)}] {row['record_id']}")

    if np.any(count == 0):
        raise ValueError(f"no train samples for modalities at indices {np.where(count == 0)[0].tolist()}")
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 1e-20)
    std = np.sqrt(variance)
    payload = {
        "modalities": list(args.modalities),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": count.tolist(),
        "source_split": "train",
        "train_record_count": len(train_rows),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {output}")
    return 0


import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from sleepwm.data import read_manifest, resolve_record_path
from sleepwm.engine import write_json
from sleepwm.features import FEATURE_GROUPS, FEATURE_NAMES, FEATURE_UNITS, extract_record_features


def parse_features() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract sealed Stage 5 train/validation targets.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-rate", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=("train", "val"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_filename(record_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", record_id)


def run_features() -> int:
    args = parse_features()
    if args.sample_rate < 64 or args.chunk_size < 1:
        raise ValueError("sample-rate and chunk-size are invalid")
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir)
    record_dir = output_dir / "records"
    record_dir.mkdir(parents=True, exist_ok=True)
    selected_splits = set(args.splits)
    output_rows = []
    train_features = []
    train_valid = []
    split_valid = {split: [] for split in selected_splits}

    for row_index, row in enumerate(read_manifest(manifest_path)):
        split = row["split"]
        if split not in selected_splits:
            continue
        source_path = resolve_record_path(manifest_path, row["npz_path"])
        filename = f"{row_index:04d}_{safe_filename(row['record_id'])}.npz"
        feature_path = record_dir / filename
        if feature_path.exists() and not args.overwrite:
            with np.load(feature_path, allow_pickle=False) as archive:
                features = archive["features"].astype(np.float32, copy=False)
                valid = archive["valid"].astype(bool, copy=False)
        else:
            with np.load(source_path, allow_pickle=False) as archive:
                signals = archive["signals"].astype(np.float32, copy=False)
            features, valid = extract_record_features(
                signals,
                sample_rate=args.sample_rate,
                chunk_size=args.chunk_size,
            )
            np.savez_compressed(
                feature_path,
                features=features,
                valid=valid,
                feature_names=np.asarray(FEATURE_NAMES),
            )
        if features.shape != valid.shape or features.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"invalid cached feature shape: {feature_path}")
        if split == "train":
            train_features.append(features)
            train_valid.append(valid)
        split_valid[split].append(valid)
        output_rows.append(
            {
                "record_id": row["record_id"],
                "subject": row["subject"],
                "split": split,
                "source_npz": str(source_path),
                "feature_npz": str(feature_path.resolve()),
                "epochs": str(features.shape[0]),
            }
        )

    if not train_features:
        raise ValueError("train split is required to compute feature statistics")
    train_values = np.concatenate(train_features)
    train_mask = np.concatenate(train_valid)
    median = []
    mean = []
    std = []
    for feature_index in range(len(FEATURE_NAMES)):
        values = train_values[train_mask[:, feature_index], feature_index].astype(np.float64)
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError(f"insufficient valid train values for {FEATURE_NAMES[feature_index]}")
        median.append(float(np.median(values)))
        mean.append(float(np.mean(values)))
        scale = float(np.std(values))
        std.append(scale if scale >= 1e-8 else 1.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_output = output_dir / "feature_manifest.csv"
    with manifest_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    valid_fraction = {}
    for split, masks in split_valid.items():
        combined = np.concatenate(masks)
        valid_fraction[split] = {
            name: float(combined[:, index].mean()) for index, name in enumerate(FEATURE_NAMES)
        }
    write_json(
        output_dir / "feature_statistics.json",
        {
            "source_manifest": str(manifest_path),
            "allowed_splits": sorted(selected_splits),
            "sample_rate": args.sample_rate,
            "feature_names": list(FEATURE_NAMES),
            "feature_groups": {key: list(value) for key, value in FEATURE_GROUPS.items()},
            "feature_units": FEATURE_UNITS,
            "train_median": median,
            "train_mean": mean,
            "train_std": std,
            "valid_fraction": valid_fraction,
            "records": len(output_rows),
        },
    )
    print(f"wrote {manifest_output}")
    print(f"wrote {output_dir / 'feature_statistics.json'}")
    return 0


import argparse
import csv
from pathlib import Path

import numpy as np


def parse_toy() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny synthetic dataset for software smoke tests.")
    parser.add_argument("--output-dir", default=".toy_data")
    parser.add_argument("--sample-rate", type=int, default=128)
    parser.add_argument("--epoch-seconds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--epochs", type=int, default=48, help="epochs per record; final SleepWM needs at least 34")
    return parser.parse_args()


def synthesize_record(epochs: int, samples: int, sample_rate: int, rng: np.random.Generator):
    labels = np.arange(epochs, dtype=np.int64) % 5
    time = np.arange(samples, dtype=np.float32) / sample_rate
    signals = np.zeros((epochs, 3, samples), dtype=np.float32)
    for epoch, label in enumerate(labels.tolist()):
        eeg_frequency = (10.0, 6.0, 3.0, 1.5, 7.0)[label]
        signals[epoch, 0] = np.sin(2 * np.pi * eeg_frequency * time)
        signals[epoch, 0] += 0.15 * rng.standard_normal(samples)

        heart_rate_hz = 1.0 + 0.05 * label
        phase = np.mod(time * heart_rate_hz, 1.0)
        signals[epoch, 1] = np.exp(-((phase - 0.08) ** 2) / 0.0008)
        signals[epoch, 1] += 0.03 * rng.standard_normal(samples)

        emg_scale = (0.30, 0.20, 0.12, 0.08, 0.15)[label]
        signals[epoch, 2] = emg_scale * rng.standard_normal(samples)
    present = np.ones((epochs, 3), dtype=bool)
    return signals, labels, present


def run_toy() -> int:
    args = parse_toy()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    samples = args.sample_rate * args.epoch_seconds
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    specifications = (("toy_train", "subject_train", "train", args.epochs), ("toy_val", "subject_val", "val", args.epochs), ("toy_test", "subject_test", "test", args.epochs))
    rows = []
    for record_id, subject, split, epochs in specifications:
        signals, labels, present = synthesize_record(epochs, samples, args.sample_rate, rng)
        path = output_dir / f"{record_id}.npz"
        np.savez_compressed(path, signals=signals, labels=labels, modality_present=present)
        rows.append({"record_id": record_id, "subject": subject, "split": split, "npz_path": path.name})

    manifest = output_dir / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["record_id", "subject", "split", "npz_path"])
        writer.writeheader()
        writer.writerows(rows)
    print(manifest)
    return 0

def main():
    import sys
    commands = {"normalize": run_normalize, "features": run_features, "toy": run_toy}
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python prepare_data.py {normalize,features,toy} [options]\n"
              "Use a command followed by --help for its options.")
        return 0
    command = sys.argv.pop(1)
    if command not in commands:
        raise SystemExit(f"Unknown command: {command}")
    return commands[command]()

if __name__ == "__main__":
    raise SystemExit(main())
