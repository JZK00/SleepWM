from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import mne
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from uniphysio_wm.preprocessing import (  # noqa: E402
    ChannelSpec,
    cap_unit_scale,
    materialize_channel,
    parse_cap_stage_events,
    resolve_channel_spec,
)


MODALITY_RULES = {
    "EEG": {
        "direct": ("C4-A1", "C4A1", "C3-A2", "C3A2", "F4-C4", "C4-P4"),
        "pairs": (("C4", "A1"), ("C3", "A2")),
    },
    "ECG": {
        "direct": ("ECG1-ECG2", "ECG", "EKG"),
        "pairs": (("ECG1", "ECG2"),),
    },
    "EMG": {
        "direct": ("EMG1-EMG2", "EMG-EMG", "EMG", "MILO"),
        "pairs": (("EMG1", "EMG2"), ("CHIN1", "CHIN2"), ("CHIN-0", "CHIN-1")),
    },
}
MODALITIES = tuple(MODALITY_RULES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare trimodal CAP sleep recordings as 30-second NPZ epochs.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--raw-manifest", required=True)
    parser.add_argument("--processed-manifest", required=True)
    parser.add_argument("--sample-rate", type=int, default=128)
    parser.add_argument("--epoch-seconds", type=int, default=30)
    parser.add_argument("--split-seed", type=int, default=20260804)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--records", nargs="*", default=(), help="optional EDF stems for targeted smoke runs")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1, help="parallel preprocessing workers")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def subject_split(subject: str, seed: int, train_fraction: float, val_fraction: float) -> str:
    digest = hashlib.sha1(f"{seed}:{subject}".encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(16**12)
    if value < train_fraction:
        return "train"
    if value < train_fraction + val_fraction:
        return "val"
    return "test"


def resolve_modalities(ch_names: Sequence[str]) -> dict[str, ChannelSpec | None]:
    return {
        modality: resolve_channel_spec(ch_names, rules["direct"], rules["pairs"])
        for modality, rules in MODALITY_RULES.items()
    }


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def discover_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(args.raw_dir).resolve()
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    edf_paths = sorted(root.glob("*.edf"))
    if args.records:
        requested = set(args.records)
        edf_paths = [path for path in edf_paths if path.stem in requested]
        found = {path.stem for path in edf_paths}
        missing = sorted(requested - found)
        if missing:
            raise FileNotFoundError(f"requested CAP records not found: {missing}")
    if args.limit > 0:
        edf_paths = edf_paths[: args.limit]

    for index, edf_path in enumerate(edf_paths, start=1):
        annotation_path = edf_path.with_suffix(".txt")
        if not annotation_path.exists():
            excluded.append({"record": edf_path.stem, "reason": "missing annotation text"})
            continue
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
        selected = resolve_modalities(raw.ch_names)
        missing = [name for name, spec in selected.items() if spec is None]
        if missing:
            excluded.append({"record": edf_path.stem, "reason": f"missing {','.join(missing)}"})
            print(f"[{index}/{len(edf_paths)}] exclude {edf_path.stem}: {','.join(missing)}")
            continue
        unit_scales = {
            name: cap_unit_scale(spec, raw._orig_units)
            for name, spec in selected.items()
            if spec is not None
        }
        subject = f"cap:{edf_path.stem}"
        rows.append(
            {
                "dataset_id": "cap",
                "record_id": f"cap_{edf_path.stem}",
                "subject": subject,
                "split": subject_split(
                    subject,
                    args.split_seed,
                    args.train_fraction,
                    args.val_fraction,
                ),
                "psg_path": str(edf_path),
                "annotation_path": str(annotation_path),
                "selected_channels": json.dumps(
                    {name: spec.display_name for name, spec in selected.items() if spec is not None},
                    sort_keys=True,
                ),
                "unit_scales": json.dumps(unit_scales, sort_keys=True),
            }
        )
    return rows, excluded


def load_existing_record(path: Path) -> tuple[int, dict[str, int], int]:
    with np.load(path, allow_pickle=False) as archive:
        labels = archive["labels"].astype(np.int64, copy=False)
        skipped = int(archive["skipped_annotations"]) if "skipped_annotations" in archive else 0
    counts = {str(label): int(count) for label, count in zip(*np.unique(labels, return_counts=True))}
    return int(labels.shape[0]), counts, skipped


def preprocess_record(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    processed_dir = Path(args.processed_dir).resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / f"{row['record_id']}.npz"
    if output_path.exists() and not args.overwrite:
        epoch_count, stage_counts, skipped = load_existing_record(output_path)
        return {
            **row,
            "npz_path": str(output_path),
            "n_epochs": epoch_count,
            "stage_counts": json.dumps(stage_counts, sort_keys=True),
            "skipped_annotations": skipped,
        }

    raw = mne.io.read_raw_edf(row["psg_path"], preload=True, verbose="ERROR")
    selected = resolve_modalities(raw.ch_names)
    if any(spec is None for spec in selected.values()):
        raise ValueError("required channel disappeared between manifest and preprocessing")
    if raw.info["meas_date"] is None:
        raise ValueError("EDF recording start time is missing")

    raw.resample(args.sample_rate, verbose="ERROR")
    specs = {name: spec for name, spec in selected.items() if spec is not None}
    unit_scales = {name: cap_unit_scale(spec, raw._orig_units) for name, spec in specs.items()}
    source_names = list(dict.fromkeys(source for spec in specs.values() for source in spec.sources))
    source_matrix = raw.get_data(picks=source_names).astype(np.float32, copy=False)
    signal_by_name = {name: source_matrix[index] for index, name in enumerate(source_names)}
    signals = np.stack(
        [
            materialize_channel(signal_by_name, specs[modality]) * unit_scales[modality]
            for modality in MODALITIES
        ],
        axis=0,
    )

    events = parse_cap_stage_events(row["annotation_path"], raw.info["meas_date"])
    samples_per_epoch = args.sample_rate * args.epoch_seconds
    epochs: list[np.ndarray] = []
    labels: list[int] = []
    skipped = 0
    for event in events:
        if event.duration_seconds < args.epoch_seconds * 0.99:
            skipped += 1
            continue
        start = int(round(event.onset_seconds * args.sample_rate))
        end = start + samples_per_epoch
        if start < 0 or end > signals.shape[1]:
            skipped += 1
            continue
        epoch = signals[:, start:end]
        if epoch.shape != (len(MODALITIES), samples_per_epoch) or not np.isfinite(epoch).all():
            skipped += 1
            continue
        epochs.append(epoch)
        labels.append(event.label)
    if not epochs:
        raise ValueError("record produced zero in-bounds finite epochs")

    epoch_array = np.stack(epochs).astype(np.float32, copy=False)
    label_array = np.asarray(labels, dtype=np.int64)
    present = np.ones(epoch_array.shape[:2], dtype=bool)
    stage_counts = {
        str(label): int(count) for label, count in zip(*np.unique(label_array, return_counts=True))
    }
    temporary_path = output_path.with_suffix(".npz.partial")
    with temporary_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            signals=epoch_array,
            labels=label_array,
            modality_present=present,
            modalities=np.asarray(MODALITIES),
            selected_channels=np.asarray([specs[name].display_name for name in MODALITIES]),
            unit_scales=np.asarray([unit_scales[name] for name in MODALITIES], dtype=np.float64),
            sample_rate=np.asarray(args.sample_rate),
            epoch_seconds=np.asarray(args.epoch_seconds),
            record_id=np.asarray(row["record_id"]),
            dataset_id=np.asarray("cap"),
            skipped_annotations=np.asarray(skipped),
        )
    temporary_path.replace(output_path)
    return {
        **row,
        "npz_path": str(output_path),
        "n_epochs": int(label_array.shape[0]),
        "stage_counts": json.dumps(stage_counts, sort_keys=True),
        "skipped_annotations": skipped,
    }


def main() -> int:
    args = parse_args()
    if not (0 < args.train_fraction < 1 and 0 <= args.val_fraction < 1):
        raise ValueError("invalid split fractions")
    if args.train_fraction + args.val_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test split")

    rows, excluded = discover_records(args)
    raw_fields = (
        "dataset_id",
        "record_id",
        "subject",
        "split",
        "psg_path",
        "annotation_path",
        "selected_channels",
        "unit_scales",
    )
    write_rows(Path(args.raw_manifest), rows, raw_fields)
    print("eligible_split_counts", dict(Counter(row["split"] for row in rows)))
    print(f"wrote {len(rows)} eligible records to {args.raw_manifest}; excluded={len(excluded)}")
    if excluded:
        exclusion_path = Path(args.raw_manifest).with_suffix(".excluded.json")
        exclusion_path.write_text(json.dumps(excluded, indent=2), encoding="utf-8")
        print(f"wrote exclusions to {exclusion_path}")
    if args.manifest_only:
        return 0

    processed_by_index: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []

    def record_result(index: int, row: dict[str, Any], result: dict[str, Any] | None, error: Exception | None) -> None:
        if error is None and result is not None:
            processed_by_index[index] = result
            print(f"[{index + 1}/{len(rows)}] {row['record_id']}: {result['n_epochs']} epochs")
        else:
            failures.append({"record_id": row["record_id"], "error": repr(error)})
            print(f"[{index + 1}/{len(rows)}] {row['record_id']}: FAILED {error}")

    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.workers == 1:
        for index, row in enumerate(rows):
            try:
                record_result(index, row, preprocess_record(row, args), None)
            except Exception as exc:
                record_result(index, row, None, exc)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(preprocess_record, row, args): (index, row)
                for index, row in enumerate(rows)
            }
            for future in as_completed(futures):
                index, row = futures[future]
                try:
                    record_result(index, row, future.result(), None)
                except Exception as exc:
                    record_result(index, row, None, exc)

    processed = [processed_by_index[index] for index in sorted(processed_by_index)]

    processed_fields = (*raw_fields, "npz_path", "n_epochs", "stage_counts", "skipped_annotations")
    write_rows(Path(args.processed_manifest), processed, processed_fields)
    print(f"wrote {len(processed)} processed records to {args.processed_manifest}")
    if failures:
        failure_path = Path(args.processed_manifest).with_suffix(".failures.json")
        failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print(f"wrote {len(failures)} failures to {failure_path}")
    return 0 if processed else 1


if __name__ == "__main__":
    raise SystemExit(main())
