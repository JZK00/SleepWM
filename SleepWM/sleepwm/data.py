"""Validated 30-second EEG/ECG/EMG records and causal forecasting windows."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import FEATURE_NAMES

SAMPLE_RATE = 128
EPOCH_SECONDS = 30
SAMPLES_PER_EPOCH = SAMPLE_RATE * EPOCH_SECONDS
FEATURE_MODALITY = np.array([0] * 5 + [1] * 3 + [2] * 3)


def read_manifest(path):
    """Validate the entire split assignment without opening other splits' data."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"record_id", "subject", "split", "npz_path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"manifest requires columns: {sorted(required)}")
        rows = []
        subjects, record_ids, raw_paths = {}, set(), set()
        for line, raw in enumerate(reader, 2):
            row = {key: (value or "").strip() for key, value in raw.items() if key}
            if any(not row.get(key) for key in required):
                raise ValueError(f"manifest line {line}: required field is empty")
            subject, split = row["subject"], row["split"]
            if ":" not in subject or not all(subject.split(":", 1)):
                raise ValueError("subject must include dataset prefix, e.g. hospitalA:patient001")
            if split not in {"train", "val", "test"}:
                raise ValueError(f"unknown split {split!r}; use train, val or test")
            if subject in subjects and subjects[subject] != split:
                raise ValueError(f"subject {subject!r} appears in multiple splits")
            subjects[subject] = split
            if row["record_id"] in record_ids:
                raise ValueError(f"duplicate record_id: {row['record_id']}")
            record_ids.add(row["record_id"])
            for key in ("npz_path", "features_path"):
                if row.get(key):
                    row[key] = str((path.parent / row[key]).resolve())
            if row["npz_path"] in raw_paths:
                raise ValueError(f"raw NPZ reused by multiple records: {row['npz_path']}")
            raw_paths.add(row["npz_path"])
            rows.append(row)
    if not rows:
        raise ValueError("manifest is empty")
    return rows


def load_record(path):
    """Require real onsets: never remove unknown labels or close temporal gaps."""
    with np.load(path, allow_pickle=False) as record:
        required = {"signals", "labels", "modality_present", "epoch_start_seconds"}
        if not required.issubset(record.files):
            raise ValueError(f"{path}: required NPZ arrays: {sorted(required)}")
        signals = np.asarray(record["signals"])
        labels = np.asarray(record["labels"])
        present = np.asarray(record["modality_present"])
        onsets = np.asarray(record["epoch_start_seconds"], dtype=np.float64)
    if signals.ndim != 3 or signals.shape[1:] != (3, SAMPLES_PER_EPOCH):
        raise ValueError(f"{path}: signals must have shape [N, 3, {SAMPLES_PER_EPOCH}]")
    n = len(signals)
    if n == 0 or labels.shape != (n,) or present.shape != (n, 3) or onsets.shape != (n,):
        raise ValueError(f"{path}: empty record or inconsistent epoch dimensions")
    if not np.issubdtype(signals.dtype, np.number) or np.iscomplexobj(signals):
        raise ValueError(f"{path}: signals must be real numeric values in volts")
    if not np.issubdtype(labels.dtype, np.integer) or not np.isin(labels, [-1, 0, 1, 2, 3, 4]).all():
        raise ValueError(f"{path}: labels must be integers -1 or 0..4")
    if not np.isin(present, [0, 1]).all():
        raise ValueError(f"{path}: modality_present must contain only boolean/0/1 values")
    if not np.isfinite(onsets).all() or not (np.diff(onsets) > 0).all():
        raise ValueError(f"{path}: epoch_start_seconds must be finite and strictly increasing")
    present = present.astype(bool)
    if not np.isfinite(signals[present]).all():
        raise ValueError(f"{path}: available signals contain NaN/Inf")
    with np.errstate(over="ignore", invalid="ignore"):
        signals = np.where(present[..., None], signals, 0).astype(np.float32)
    if not np.isfinite(signals).all():
        raise ValueError(f"{path}: signal values cannot be represented as float32")
    return {
        "signals": signals, "labels": labels.astype(np.int64),
        "present": present, "onsets": onsets,
    }


def load_statistics(path):
    with Path(path).open(encoding="utf-8") as handle:
        stats = json.load(handle)
    if (stats.get("schema_version") != 1 or stats.get("sample_rate") != SAMPLE_RATE
            or stats.get("epoch_seconds") != EPOCH_SECONDS
            or stats.get("feature_names") != list(FEATURE_NAMES)):
        raise ValueError("statistics schema, sample rate or feature order mismatch")
    for prefix, size in (("signal", 3), ("feature", len(FEATURE_NAMES))):
        for suffix in ("mean", "std", "count"):
            array = np.asarray(stats.get(f"{prefix}_{suffix}", []), dtype=np.float64)
            if array.shape != (size,) or not np.isfinite(array).all():
                raise ValueError(f"invalid {prefix}_{suffix} statistics")
            if suffix == "std" and not (array > 0).all():
                raise ValueError("standard deviations must be positive")
            if suffix == "count" and not ((array >= 0) & (array == np.floor(array))).all():
                raise ValueError("statistics counts must be nonnegative integers")
    return stats


def window_starts(labels, onsets, history_epochs, horizons, stride):
    """A +1 target is the epoch immediately following the last history epoch."""
    span = history_epochs + max(horizons)
    # All intermediate future labels are required for cumulative transition targets.
    for start in range(0, len(labels) - span + 1, stride):
        stop = start + span
        if (labels[start:stop] >= 0).all() and np.allclose(
            np.diff(onsets[start:stop]), EPOCH_SECONDS, rtol=0, atol=1e-4
        ):
            yield start


class SleepDataset(Dataset):
    """One split only; retain at most one raw recording in the lazy cache."""

    def __init__(self, manifest_path, statistics_path, split, history_epochs=20,
                 horizons=(1, 2, 4), stride=8):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val or test")
        self.history_epochs = int(history_epochs)
        self.horizons = tuple(int(h) for h in horizons)
        self.stride = int(stride)
        if (self.history_epochs < 1 or self.stride < 1 or not self.horizons
                or min(self.horizons) < 1 or tuple(sorted(set(self.horizons))) != self.horizons):
            raise ValueError("history/stride must be positive; horizons must be sorted unique positive integers")
        self.stats = load_statistics(statistics_path)
        self.records = [row for row in read_manifest(manifest_path) if row["split"] == split]
        if not self.records:
            raise ValueError(f"manifest has no {split} records")
        self.index = []
        self._cache_record_index, self._cache = None, None
        for record_index, row in enumerate(self.records):
            record = self._load(row)
            self.index.extend((record_index, start) for start in window_starts(
                record["labels"], record["onsets"], self.history_epochs, self.horizons, self.stride
            ))
        if not self.index:
            raise ValueError(f"no valid {split} windows: need {self.history_epochs + max(self.horizons)} "
                             "consecutive 30-second epochs with known labels")

    @staticmethod
    def _load(row):
        record = load_record(row["npz_path"])
        if not row.get("features_path"):
            raise ValueError("prepared manifest requires features_path; run prepare_data.py")
        with np.load(row["features_path"], allow_pickle=False) as cached:
            features = np.asarray(cached["features"], dtype=np.float32)
            valid_raw = np.asarray(cached["valid"])
            if not np.isin(valid_raw, [0, 1]).all():
                raise ValueError("feature valid mask must contain boolean/0/1 values")
            valid = valid_raw.astype(bool)
            if ("epoch_start_seconds" not in cached.files
                    or not np.array_equal(cached["epoch_start_seconds"], record["onsets"])):
                raise ValueError("feature cache timestamps mismatch; rerun prepare_data.py")
        expected = (len(record["labels"]), len(FEATURE_NAMES))
        if features.shape != expected or valid.shape != expected:
            raise ValueError(f"feature cache must have shape {expected}")
        if not np.isfinite(features[valid]).all():
            raise ValueError("valid physiology features contain NaN/Inf")
        valid &= record["present"][:, FEATURE_MODALITY]
        record.update(features=np.where(valid, features, 0), feature_valid=valid)
        return record

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        record_index, start = self.index[item]
        row = self.records[record_index]
        if self._cache_record_index != record_index:
            self._cache = self._load(row)
            self._cache_record_index = record_index
        record = self._cache
        end = start + self.history_epochs
        targets = end - 1 + np.asarray(self.horizons)
        present = record["present"][start:end]
        signals = (record["signals"][start:end] - np.asarray(self.stats["signal_mean"], dtype=np.float32)[None, :, None])
        signals /= np.asarray(self.stats["signal_std"], dtype=np.float32)[None, :, None]
        signals = np.where(present[..., None], signals, 0).astype(np.float32)
        valid = record["feature_valid"][targets]
        physiology = (record["features"][targets] - np.asarray(self.stats["feature_mean"], dtype=np.float32))
        physiology /= np.asarray(self.stats["feature_std"], dtype=np.float32)
        physiology = np.where(valid, physiology, 0).astype(np.float32)
        labels = record["labels"]
        transition = [float(np.any(np.diff(labels[end - 1:end + h]) != 0)) for h in self.horizons]
        return {
            "signals": torch.from_numpy(signals), "present": torch.from_numpy(present.copy()),
            "stage": torch.from_numpy(labels[targets].copy()),
            "physiology": torch.from_numpy(physiology), "physiology_valid": torch.from_numpy(valid.copy()),
            "transition": torch.tensor(transition, dtype=torch.float32),
            "subject": row["subject"], "record_id": row["record_id"], "start_epoch": start,
        }
