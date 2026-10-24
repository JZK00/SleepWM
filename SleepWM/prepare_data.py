"""Prepare feature caches and training-only normalization for your own data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from sleepwm.data import EPOCH_SECONDS, FEATURE_MODALITY, SAMPLE_RATE, load_record, load_statistics, read_manifest
from sleepwm.features import FEATURE_NAMES, extract_record_features


class Moments:
    """Merge batches with the parallel variance formula, avoiding cancellation."""
    def __init__(self, size):
        self.count = np.zeros(size, dtype=np.int64)
        self.mean = np.zeros(size, dtype=np.float64)
        self.m2 = np.zeros(size, dtype=np.float64)

    def add(self, column, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        n = len(values)
        if not n:
            return
        mean = values.mean()
        delta = mean - self.mean[column]
        total = self.count[column] + n
        self.m2[column] += np.square(values - mean).sum() + delta ** 2 * self.count[column] * n / total
        self.mean[column] += delta * n / total
        self.count[column] = total

    def result(self, prefix):
        std = np.sqrt(self.m2 / np.maximum(self.count, 1))
        std = np.where(std > 1e-12, std, 1.0)
        return {f"{prefix}_mean": self.mean.tolist(), f"{prefix}_std": std.tolist(),
                f"{prefix}_count": self.count.tolist()}


def prepare(manifest, output_dir, splits=("train", "val"), statistics=None):
    rows = read_manifest(manifest)
    splits = tuple(splits)
    if not splits or set(splits) - {"train", "val", "test"}:
        raise ValueError("splits must be selected from train, val, test")
    selected = [row for row in rows if row["split"] in splits]
    if any(not any(row["split"] == split for row in selected) for split in splits):
        raise ValueError("one or more requested splits have no records")
    if statistics is None and ("test" in splits or "train" not in splits):
        raise ValueError("test/validation-only preparation requires --statistics from training")
    stats = load_statistics(statistics) if statistics else None
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        (staging / "features").mkdir()
        signals_moments, features_moments = Moments(3), Moments(len(FEATURE_NAMES))
        prepared = []
        for row in selected:
            record = load_record(row["npz_path"])
            features, valid = extract_record_features(record["signals"], SAMPLE_RATE)
            valid &= record["present"][:, FEATURE_MODALITY] & np.isfinite(features)
            features = np.where(valid, features, 0).astype(np.float32)
            if stats is None and row["split"] == "train":
                for channel in range(3):
                    signals_moments.add(channel, record["signals"][record["present"][:, channel], channel])
                for feature in range(len(FEATURE_NAMES)):
                    features_moments.add(feature, features[valid[:, feature], feature])
            filename = hashlib.sha256(row["record_id"].encode("utf-8")).hexdigest()[:20] + ".npz"
            np.savez_compressed(staging / "features" / filename, features=features, valid=valid,
                                epoch_start_seconds=record["onsets"])
            prepared.append({key: row[key] for key in ("record_id", "subject", "split", "npz_path")})
            prepared[-1]["features_path"] = "features/" + filename
        if stats is None:
            stats = {"schema_version": 1, "sample_rate": SAMPLE_RATE, "epoch_seconds": EPOCH_SECONDS,
                     "feature_names": list(FEATURE_NAMES), "fit_split": "train",
                     **signals_moments.result("signal"), **features_moments.result("feature")}
        with (staging / "statistics.json").open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2, allow_nan=False)
            handle.write("\n")
        with (staging / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["record_id", "subject", "split", "npz_path", "features_path"])
            writer.writeheader()
            writer.writerows(prepared)
        staging.rename(output_dir)
    except Exception:
        shutil.rmtree(staging)
        raise
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Raw CSV manifest")
    parser.add_argument("--output-dir", required=True, help="New directory; existing directories are rejected")
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val"])
    parser.add_argument("--statistics", help="Frozen training statistics; mandatory when preparing test")
    args = parser.parse_args()
    path = prepare(args.manifest, args.output_dir, args.splits, args.statistics)
    print(f"Prepared {path / 'manifest.csv'}")
    print(f"Statistics: {path / 'statistics.json'}")


if __name__ == "__main__":
    main()
