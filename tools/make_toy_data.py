from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny synthetic dataset for software smoke tests.")
    parser.add_argument("--output-dir", default=".toy_data")
    parser.add_argument("--sample-rate", type=int, default=128)
    parser.add_argument("--epoch-seconds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260804)
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


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    samples = args.sample_rate * args.epoch_seconds
    specifications = (("toy_train", "subject_train", "train", 24), ("toy_val", "subject_val", "val", 18), ("toy_test", "subject_test", "test", 18))
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


if __name__ == "__main__":
    raise SystemExit(main())

