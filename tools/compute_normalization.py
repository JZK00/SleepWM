from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from uniphysio_wm.data import read_manifest, resolve_record_path, validate_subject_splits  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute modality statistics from train subjects only.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--modalities", nargs="+", default=["EEG", "ECG", "EMG"])
    parser.add_argument("--chunk-epochs", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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


if __name__ == "__main__":
    raise SystemExit(main())

