from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import mne
import numpy as np
from scipy.signal import resample_poly


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from uniphysio_wm.external_sleep import (  # noqa: E402
    MODALITIES,
    edf_compatible_path,
    expand_sleep_edfx_annotations,
    map_isruc_labels,
    resolve_external_channels,
    sleep_edfx_pair_key,
    sleep_edfx_subject,
    subject_split,
    trimmed_sleep_interval,
)
from uniphysio_wm.preprocessing import ChannelSpec, materialize_channel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ISRUC or Sleep-EDF for sleep-only transfer experiments.")
    parser.add_argument("--dataset", choices=("isruc", "sleep_edfx"), required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--raw-manifest", required=True)
    parser.add_argument("--processed-manifest", required=True)
    parser.add_argument("--sample-rate", type=int, default=128)
    parser.add_argument("--epoch-seconds", type=int, default=30)
    parser.add_argument("--split-seed", type=int, default=20260804)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--wake-margin-minutes", type=int, default=30)
    parser.add_argument("--records", nargs="*", default=())
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_edf_header(path: Path) -> tuple[list[str], float]:
    with edf_compatible_path(path) as compatible:
        raw = mne.io.read_raw_edf(compatible, preload=False, verbose="ERROR")
    return list(raw.ch_names), float(raw.info["sfreq"])


def _selected_channels(ch_names: Sequence[str], dataset: str) -> dict[str, ChannelSpec | None]:
    return resolve_external_channels(ch_names, dataset)


def _serialized_channels(selected: dict[str, ChannelSpec | None]) -> str:
    return json.dumps(
        {name: spec.display_name if spec is not None else None for name, spec in selected.items()},
        sort_keys=True,
    )


def discover_isruc(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    root = Path(args.raw_dir).resolve()
    paths = sorted(
        root.rglob("*.rec"),
        key=lambda value: (0, int(value.stem)) if value.stem.isdigit() else (1, value.stem),
    )
    requested = set(args.records)
    if requested:
        paths = [path for path in paths if path.stem in requested]
        missing = requested - {path.stem for path in paths}
        if missing:
            raise FileNotFoundError(f"requested ISRUC records not found: {sorted(missing)}")
    if args.limit > 0:
        paths = paths[: args.limit]

    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for path in paths:
        label_path = path.with_name(f"{path.stem}_1.txt")
        if not label_path.exists():
            excluded.append({"record_id": path.stem, "reason": "missing scorer-1 labels"})
            continue
        ch_names, native_rate = _read_edf_header(path)
        selected = _selected_channels(ch_names, "isruc")
        missing_modalities = [name for name, value in selected.items() if value is None]
        if missing_modalities:
            excluded.append({"record_id": path.stem, "reason": f"missing {','.join(missing_modalities)}"})
            continue
        subject = f"isruc:{path.stem}"
        rows.append(
            {
                "dataset_id": "isruc",
                "record_id": f"isruc_{path.stem}",
                "subject": subject,
                "split": subject_split(subject, args.split_seed, args.train_fraction, args.val_fraction),
                "psg_path": str(path),
                "annotation_path": str(label_path),
                "selected_channels": _serialized_channels(selected),
                "available_modalities": "EEG,ECG,EMG",
                "native_sample_rate": native_rate,
            }
        )
    return rows, excluded


def discover_sleep_edfx(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    root = Path(args.raw_dir).resolve()
    psg_paths = sorted(root.rglob("*-PSG.edf"))
    annotations = {sleep_edfx_pair_key(path): path for path in root.rglob("*-Hypnogram.edf")}
    requested = set(args.records)
    if requested:
        psg_paths = [path for path in psg_paths if path.stem in requested]
        missing = requested - {path.stem for path in psg_paths}
        if missing:
            raise FileNotFoundError(f"requested Sleep-EDF records not found: {sorted(missing)}")
    if args.limit > 0:
        psg_paths = psg_paths[: args.limit]

    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for path in psg_paths:
        annotation_path = annotations.get(sleep_edfx_pair_key(path))
        if annotation_path is None:
            excluded.append({"record_id": path.stem, "reason": "missing paired hypnogram"})
            continue
        ch_names, native_rate = _read_edf_header(path)
        selected = _selected_channels(ch_names, "sleep_edfx")
        if selected["EEG"] is None or selected["EMG"] is None:
            excluded.append({"record_id": path.stem, "reason": "missing EEG or EMG"})
            continue
        subject = sleep_edfx_subject(path)
        rows.append(
            {
                "dataset_id": "sleep_edfx",
                "record_id": f"sleep_edfx_{path.stem}",
                "subject": subject,
                "split": subject_split(subject, args.split_seed, args.train_fraction, args.val_fraction),
                "psg_path": str(path),
                "annotation_path": str(annotation_path),
                "selected_channels": _serialized_channels(selected),
                "available_modalities": "EEG,EMG",
                "native_sample_rate": native_rate,
            }
        )
    return rows, excluded


def discover_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    return discover_isruc(args) if args.dataset == "isruc" else discover_sleep_edfx(args)


def _resample(signals: np.ndarray, native_rate: float, target_rate: int) -> np.ndarray:
    rounded_native = int(round(native_rate))
    if not np.isclose(native_rate, rounded_native):
        raise ValueError(f"non-integer source sample rate is unsupported: {native_rate}")
    if rounded_native == target_rate:
        return signals.astype(np.float32, copy=False)
    return resample_poly(signals, target_rate, rounded_native, axis=-1).astype(np.float32, copy=False)


def _load_selected_signals(row: dict[str, Any], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    with edf_compatible_path(row["psg_path"]) as compatible:
        raw = mne.io.read_raw_edf(compatible, preload=False, verbose="ERROR")
        selected = _selected_channels(raw.ch_names, args.dataset)
        source_names = list(
            dict.fromkeys(
                source
                for spec in selected.values()
                if spec is not None
                for source in spec.sources
            )
        )
        source = raw.get_data(picks=source_names).astype(np.float32, copy=False)
        source_rate = float(raw.info["sfreq"])
    source = _resample(source, source_rate, args.sample_rate)
    by_name = {name: source[index] for index, name in enumerate(source_names)}
    signals = np.zeros((len(MODALITIES), source.shape[-1]), dtype=np.float32)
    present = np.zeros(len(MODALITIES), dtype=bool)
    for index, modality in enumerate(MODALITIES):
        spec = selected[modality]
        if spec is not None:
            signals[index] = materialize_channel(by_name, spec)
            present[index] = True
    return signals, present


def _prepare_isruc(row: dict[str, Any], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    signals, present = _load_selected_signals(row, args)
    values = [int(line.strip()) for line in Path(row["annotation_path"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = map_isruc_labels(values)
    samples_per_epoch = args.sample_rate * args.epoch_seconds
    n_epochs = min(int(labels.shape[0]), int(signals.shape[-1] // samples_per_epoch))
    signals = signals[:, : n_epochs * samples_per_epoch].reshape(len(MODALITIES), n_epochs, samples_per_epoch)
    epochs = np.transpose(signals, (1, 0, 2)).copy()
    return epochs, labels[:n_epochs], np.broadcast_to(present, (n_epochs, len(MODALITIES))).copy(), 0


def _prepare_sleep_edfx(row: dict[str, Any], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    signals, present = _load_selected_signals(row, args)
    samples_per_epoch = args.sample_rate * args.epoch_seconds
    n_epochs = int(signals.shape[-1] // samples_per_epoch)
    annotations = mne.read_annotations(row["annotation_path"])
    labels = expand_sleep_edfx_annotations(
        annotations.onset,
        annotations.duration,
        annotations.description,
        n_epochs=n_epochs,
        epoch_seconds=args.epoch_seconds,
    )
    margin = int(round(args.wake_margin_minutes * 60 / args.epoch_seconds))
    start, end = trimmed_sleep_interval(labels, margin)
    signals = signals[:, start * samples_per_epoch : end * samples_per_epoch]
    epochs = np.transpose(
        signals.reshape(len(MODALITIES), end - start, samples_per_epoch),
        (1, 0, 2),
    ).copy()
    labels = labels[start:end]
    invalid = int(np.count_nonzero(labels < 0))
    return epochs, labels, np.broadcast_to(present, (end - start, len(MODALITIES))).copy(), invalid


def preprocess_record(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.processed_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{row['record_id']}.npz"
    if output_path.exists() and not args.overwrite:
        with np.load(output_path, allow_pickle=False) as archive:
            labels = archive["labels"]
        stage_counts = {str(label): int(count) for label, count in zip(*np.unique(labels[labels >= 0], return_counts=True))}
        return {**row, "npz_path": str(output_path), "n_epochs": int(labels.shape[0]), "stage_counts": json.dumps(stage_counts, sort_keys=True), "invalid_epochs": int(np.count_nonzero(labels < 0))}

    prepared = _prepare_isruc(row, args) if args.dataset == "isruc" else _prepare_sleep_edfx(row, args)
    epochs, labels, present, invalid = prepared
    if not epochs.size or not np.isfinite(epochs).all():
        raise ValueError("record produced empty or non-finite epochs")
    stage_counts = {str(label): int(count) for label, count in zip(*np.unique(labels[labels >= 0], return_counts=True))}
    temporary = output_path.with_suffix(".npz.partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            signals=epochs.astype(np.float32, copy=False),
            labels=labels.astype(np.int64, copy=False),
            modality_present=present,
            modalities=np.asarray(MODALITIES),
            selected_channels=np.asarray([json.loads(row["selected_channels"])[name] or "" for name in MODALITIES]),
            sample_rate=np.asarray(args.sample_rate),
            epoch_seconds=np.asarray(args.epoch_seconds),
            record_id=np.asarray(row["record_id"]),
            dataset_id=np.asarray(args.dataset),
        )
    temporary.replace(output_path)
    return {**row, "npz_path": str(output_path), "n_epochs": int(labels.shape[0]), "stage_counts": json.dumps(stage_counts, sort_keys=True), "invalid_epochs": invalid}


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.train_fraction <= 0 or args.train_fraction + args.val_fraction >= 1:
        raise ValueError("split fractions must leave non-empty train and test ranges")
    rows, excluded = discover_records(args)
    raw_fields = ("dataset_id", "record_id", "subject", "split", "psg_path", "annotation_path", "selected_channels", "available_modalities", "native_sample_rate")
    write_rows(Path(args.raw_manifest), rows, raw_fields)
    Path(args.raw_manifest).with_suffix(".excluded.json").write_text(json.dumps(excluded, indent=2), encoding="utf-8")
    print("eligible_split_counts", dict(Counter(row["split"] for row in rows)))
    print(f"wrote {len(rows)} records to {args.raw_manifest}; excluded={len(excluded)}")
    if args.manifest_only:
        return 0

    results: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    if args.workers == 1:
        for index, row in enumerate(rows):
            try:
                results[index] = preprocess_record(row, args)
                print(f"[{index + 1}/{len(rows)}] {row['record_id']}: {results[index]['n_epochs']} epochs")
            except Exception as error:
                failures.append({"record_id": row["record_id"], "error": repr(error)})
                print(f"[{index + 1}/{len(rows)}] {row['record_id']}: FAILED {error}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(preprocess_record, row, args): (index, row) for index, row in enumerate(rows)}
            for future in as_completed(futures):
                index, row = futures[future]
                try:
                    results[index] = future.result()
                    print(f"[{index + 1}/{len(rows)}] {row['record_id']}: {results[index]['n_epochs']} epochs")
                except Exception as error:
                    failures.append({"record_id": row["record_id"], "error": repr(error)})
                    print(f"[{index + 1}/{len(rows)}] {row['record_id']}: FAILED {error}")

    processed = [results[index] for index in sorted(results)]
    processed_fields = (*raw_fields, "npz_path", "n_epochs", "stage_counts", "invalid_epochs")
    write_rows(Path(args.processed_manifest), processed, processed_fields)
    Path(args.processed_manifest).with_suffix(".failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(f"wrote {len(processed)} records to {args.processed_manifest}; failures={len(failures)}")
    return 0 if processed else 1


if __name__ == "__main__":
    raise SystemExit(main())
