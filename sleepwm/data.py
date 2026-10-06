from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def read_manifest(path: str | Path) -> List[Dict[str, str]]:
    manifest = Path(path)
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"record_id", "subject", "split", "npz_path"}
    if not rows:
        raise ValueError(f"manifest is empty: {manifest}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    return rows


def validate_subject_splits(rows: Sequence[Dict[str, str]]) -> None:
    subject_splits: Dict[str, set[str]] = {}
    for row in rows:
        subject_splits.setdefault(row["subject"], set()).add(row["split"])
    leaking = {subject: splits for subject, splits in subject_splits.items() if len(splits) > 1}
    if leaking:
        preview = list(leaking.items())[:5]
        raise ValueError(f"subjects occur in multiple splits: {preview}")


def resolve_record_path(manifest_path: str | Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path(manifest_path).resolve().parent / path).resolve()


def load_normalization(
    path: str | Path | None,
    modalities: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray] | None:
    if path is None:
        return None
    stats_path = Path(path).expanduser()
    if not stats_path.exists():
        raise FileNotFoundError(f"normalization statistics not found: {stats_path}")
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    stats_modalities = list(payload["modalities"])
    indices = [stats_modalities.index(name) for name in modalities]
    mean = np.asarray(payload["mean"], dtype=np.float32)[indices]
    std = np.asarray(payload["std"], dtype=np.float32)[indices]
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError(f"invalid normalization statistics: {stats_path}")
    return mean, std


class _RecordCache:
    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self.items: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()

    def load(self, path: Path) -> Dict[str, np.ndarray]:
        key = str(path)
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key]
        with np.load(path, allow_pickle=False) as archive:
            signals = archive["signals"].astype(np.float32, copy=False)
            labels = archive["labels"].astype(np.int64, copy=False)
            if "modality_present" in archive:
                present = archive["modality_present"].astype(bool, copy=False)
            else:
                present = np.ones(signals.shape[:2], dtype=bool)
        if signals.ndim != 3:
            raise ValueError(f"signals must be [epochs, modalities, samples]: {path}")
        if labels.shape != (signals.shape[0],):
            raise ValueError(f"labels do not match signals: {path}")
        if present.shape != signals.shape[:2]:
            raise ValueError(f"modality_present does not match signals: {path}")
        item = {"signals": signals, "labels": labels, "modality_present": present}
        self.items[key] = item
        while len(self.items) > self.capacity:
            self.items.popitem(last=False)
        return item


class EpochDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        stored_modalities: Sequence[str] = ("EEG", "ECG", "EMG"),
        modalities: Sequence[str] = ("EEG", "ECG", "EMG"),
        num_classes: int = 5,
        cache_records: int = 4,
        normalization_path: str | Path | None = None,
        label_fraction: float = 1.0,
        subset_seed: int = 0,
    ) -> None:
        if not 0.0 < label_fraction <= 1.0:
            raise ValueError("label_fraction must be in (0, 1]")
        self.manifest_path = str(manifest_path)
        self.split = split
        self.stored_modalities = tuple(stored_modalities)
        self.modalities = tuple(modalities)
        self.modality_indices = [self.stored_modalities.index(name) for name in self.modalities]
        self.normalization = load_normalization(normalization_path, self.modalities)
        self.rows = read_manifest(manifest_path)
        validate_subject_splits(self.rows)
        self.records: List[Dict[str, object]] = []
        self.index: List[Tuple[int, int]] = []
        self.labels: List[int] = []
        self.label_counts = np.zeros(num_classes, dtype=np.int64)
        self.cache = _RecordCache(cache_records)

        for row in self.rows:
            if row["split"] != split:
                continue
            path = resolve_record_path(manifest_path, row["npz_path"])
            if not path.exists():
                raise FileNotFoundError(f"processed recording not found: {path}")
            with np.load(path, allow_pickle=False) as archive:
                labels = archive["labels"].astype(np.int64, copy=False)
            record_index = len(self.records)
            self.records.append({**row, "path": path, "epochs": int(labels.shape[0])})
            for epoch_index, label in enumerate(labels.tolist()):
                if 0 <= int(label) < num_classes:
                    self.index.append((record_index, epoch_index))
                    self.labels.append(int(label))
                    self.label_counts[int(label)] += 1
        if not self.index:
            raise ValueError(f"no valid epochs for split={split}")
        if label_fraction < 1.0:
            rng = np.random.default_rng(subset_seed)
            labels_array = np.asarray(self.labels, dtype=np.int64)
            selected = []
            for class_index in range(num_classes):
                candidates = np.flatnonzero(labels_array == class_index)
                if not len(candidates):
                    continue
                count = max(1, int(round(len(candidates) * label_fraction)))
                selected.extend(rng.choice(candidates, size=count, replace=False).tolist())
            selected.sort()
            self.index = [self.index[index] for index in selected]
            self.labels = [self.labels[index] for index in selected]
            self.label_counts = np.bincount(self.labels, minlength=num_classes)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record_index, epoch_index = self.index[index]
        meta = self.records[record_index]
        record = self.cache.load(meta["path"])  # type: ignore[arg-type]
        signals = record["signals"][epoch_index, self.modality_indices].copy()
        if self.normalization is not None:
            mean, std = self.normalization
            signals = (signals - mean[:, None]) / std[:, None]
        present = record["modality_present"][epoch_index, self.modality_indices].copy()
        return {
            "signals": torch.from_numpy(signals),
            "label": torch.tensor(int(record["labels"][epoch_index]), dtype=torch.long),
            "modality_present": torch.from_numpy(present),
            "record_id": str(meta["record_id"]),
            "subject": str(meta["subject"]),
            "epoch_index": torch.tensor(epoch_index, dtype=torch.long),
        }


class RecordBatchSampler(Sampler[List[int]]):
    """Shuffle records and within-record samples without repeated NPZ decompression."""

    def __init__(
        self,
        dataset: EpochDataset | "SequenceDataset",
        batch_size: int,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = drop_last
        self.epoch = 0
        self.indices_by_record: Dict[int, List[int]] = {}
        for dataset_index, (record_index, _) in enumerate(dataset.index):
            self.indices_by_record.setdefault(record_index, []).append(dataset_index)

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        record_indices = np.asarray(list(self.indices_by_record), dtype=np.int64)
        rng.shuffle(record_indices)
        for record_index in record_indices.tolist():
            indices = np.asarray(self.indices_by_record[record_index], dtype=np.int64)
            rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size].tolist()
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return sum(len(indices) // self.batch_size for indices in self.indices_by_record.values())
        return sum((len(indices) + self.batch_size - 1) // self.batch_size for indices in self.indices_by_record.values())


class TransitionBalancedRecordBatchSampler(Sampler[List[int]]):
    """Build record-local batches with a fixed transition-window fraction."""

    def __init__(
        self,
        dataset: "SequenceDataset",
        batch_size: int,
        seed: int,
        transition_probability: float = 0.5,
        drop_last: bool = False,
    ) -> None:
        if not 0.0 <= transition_probability <= 1.0:
            raise ValueError("transition_probability must be in [0, 1]")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.transition_probability = float(transition_probability)
        self.drop_last = drop_last
        self.epoch = 0
        self.indices_by_record: Dict[int, List[int]] = {}
        for dataset_index, (record_index, _) in enumerate(dataset.index):
            self.indices_by_record.setdefault(record_index, []).append(dataset_index)

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        record_indices = np.asarray(list(self.indices_by_record), dtype=np.int64)
        rng.shuffle(record_indices)
        for record_index in record_indices.tolist():
            indices = self.indices_by_record[record_index]
            transition = np.asarray(
                [index for index in indices if self.dataset.transition_window[index]],
                dtype=np.int64,
            )
            stable = np.asarray(
                [index for index in indices if not self.dataset.transition_window[index]],
                dtype=np.int64,
            )
            rng.shuffle(transition)
            rng.shuffle(stable)
            cursors = {"transition": 0, "stable": 0}

            def draw(pool: np.ndarray, count: int, name: str) -> List[int]:
                selected: List[int] = []
                while len(selected) < count:
                    if cursors[name] >= len(pool):
                        rng.shuffle(pool)
                        cursors[name] = 0
                    take = min(count - len(selected), len(pool) - cursors[name])
                    selected.extend(pool[cursors[name] : cursors[name] + take].tolist())
                    cursors[name] += take
                return selected

            for start in range(0, len(indices), self.batch_size):
                current_batch_size = min(self.batch_size, len(indices) - start)
                if current_batch_size < self.batch_size and self.drop_last:
                    continue
                if not len(transition) or not len(stable):
                    pool = transition if len(transition) else stable
                    batch = draw(pool, current_batch_size, "transition" if len(transition) else "stable")
                else:
                    transition_count = int(round(current_batch_size * self.transition_probability))
                    stable_count = current_batch_size - transition_count
                    batch = draw(transition, transition_count, "transition")
                    batch.extend(draw(stable, stable_count, "stable"))
                rng.shuffle(batch)
                yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return sum(len(indices) // self.batch_size for indices in self.indices_by_record.values())
        return sum((len(indices) + self.batch_size - 1) // self.batch_size for indices in self.indices_by_record.values())


class SequenceDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        history_epochs: int,
        future_horizons: Sequence[int],
        stored_modalities: Sequence[str] = ("EEG", "ECG", "EMG"),
        modalities: Sequence[str] = ("EEG", "ECG", "EMG"),
        cache_records: int = 2,
        normalization_path: str | Path | None = None,
        sequence_stride: int = 1,
        sequence_start_offset: int = 0,
        num_classes: int = 5,
    ) -> None:
        self.manifest_path = str(manifest_path)
        self.history_epochs = int(history_epochs)
        self.future_horizons = tuple(sorted({int(value) for value in future_horizons}))
        self.sequence_stride = int(sequence_stride)
        self.sequence_start_offset = int(sequence_start_offset)
        if (
            self.history_epochs < 1
            or not self.future_horizons
            or self.future_horizons[0] < 1
            or self.sequence_stride < 1
            or self.sequence_start_offset < 0
        ):
            raise ValueError("history, horizons, stride, and start offset are invalid")
        self.stored_modalities = tuple(stored_modalities)
        self.modalities = tuple(modalities)
        self.modality_indices = [self.stored_modalities.index(name) for name in self.modalities]
        self.normalization = load_normalization(normalization_path, self.modalities)
        rows = read_manifest(manifest_path)
        validate_subject_splits(rows)
        self.records: List[Dict[str, object]] = []
        self.index: List[Tuple[int, int]] = []
        self.cache = _RecordCache(cache_records)
        self.future_label_counts = np.zeros(num_classes, dtype=np.int64)
        self.transition_mask: List[Tuple[bool, ...]] = []
        self.transition_window: List[bool] = []

        required_epochs = self.history_epochs + max(self.future_horizons)
        for row in rows:
            if row["split"] != split:
                continue
            path = resolve_record_path(manifest_path, row["npz_path"])
            if not path.exists():
                raise FileNotFoundError(f"processed recording not found: {path}")
            with np.load(path, allow_pickle=False) as archive:
                labels = archive["labels"].astype(np.int64, copy=False)
                epoch_count = int(labels.shape[0])
            record_index = len(self.records)
            self.records.append({**row, "path": path, "epochs": epoch_count})
            for start in range(
                self.sequence_start_offset,
                max(0, epoch_count - required_epochs + 1),
                self.sequence_stride,
            ):
                current_index = start + self.history_epochs - 1
                future_indices = [start + self.history_epochs + horizon - 1 for horizon in self.future_horizons]
                selected_labels = labels[[current_index, *future_indices]]
                if np.any(selected_labels < 0) or np.any(selected_labels >= num_classes):
                    continue
                self.index.append((record_index, start))
                self.future_label_counts += np.bincount(labels[future_indices], minlength=num_classes)
                transition_mask = tuple(bool(value) for value in labels[future_indices] != labels[current_index])
                self.transition_mask.append(transition_mask)
                self.transition_window.append(any(transition_mask))
        if not self.index:
            raise ValueError(f"no valid sequences for split={split}")

    @property
    def transition_window_fraction(self) -> float:
        return float(np.mean(self.transition_window))

    @property
    def transition_fraction_by_horizon(self) -> Dict[str, float]:
        mask = np.asarray(self.transition_mask, dtype=bool)
        return {
            str(horizon): float(mask[:, index].mean())
            for index, horizon in enumerate(self.future_horizons)
        }

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record_index, start = self.index[index]
        meta = self.records[record_index]
        record = self.cache.load(meta["path"])  # type: ignore[arg-type]
        history_slice = slice(start, start + self.history_epochs)
        future_indices = [start + self.history_epochs + horizon - 1 for horizon in self.future_horizons]
        signals = record["signals"]
        present = record["modality_present"]
        history_signals = signals[history_slice][:, self.modality_indices].copy()
        history_present = present[history_slice][:, self.modality_indices].copy()
        future_signals = signals[future_indices][:, self.modality_indices].copy()
        future_present = present[future_indices][:, self.modality_indices].copy()
        if self.normalization is not None:
            mean, std = self.normalization
            history_signals = (history_signals - mean[None, :, None]) / std[None, :, None]
            future_signals = (future_signals - mean[None, :, None]) / std[None, :, None]
        return {
            "history_signals": torch.from_numpy(history_signals),
            "history_present": torch.from_numpy(history_present),
            "history_labels": torch.from_numpy(record["labels"][history_slice].copy()),
            "future_signals": torch.from_numpy(future_signals),
            "future_present": torch.from_numpy(future_present),
            "future_labels": torch.from_numpy(record["labels"][future_indices].copy()),
            "record_id": str(meta["record_id"]),
            "subject": str(meta["subject"]),
            "start_epoch": torch.tensor(start, dtype=torch.long),
        }


class PhysioFeatureSequenceDataset(SequenceDataset):
    """Sequence dataset augmented with standardized current/future physiology targets."""

    def __init__(
        self,
        *args,
        feature_manifest_path: str | Path,
        feature_statistics_path: str | Path,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        statistics_path = Path(feature_statistics_path)
        statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
        self.physiology_feature_names = tuple(str(value) for value in statistics["feature_names"])
        self.physiology_feature_groups = {
            str(group): tuple(str(value) for value in names)
            for group, names in statistics["feature_groups"].items()
        }
        self.physiology_mean = np.asarray(statistics["train_mean"], dtype=np.float32)
        self.physiology_std = np.asarray(statistics["train_std"], dtype=np.float32)
        if (
            self.physiology_mean.shape != (len(self.physiology_feature_names),)
            or self.physiology_std.shape != self.physiology_mean.shape
            or np.any(self.physiology_std <= 0)
        ):
            raise ValueError("invalid physiology feature statistics")

        self.physiology_records: Dict[str, Dict[str, np.ndarray]] = {}
        manifest_path = Path(feature_manifest_path)
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                record_id = str(row["record_id"])
                if record_id in self.physiology_records:
                    raise ValueError(f"duplicate physiology record_id: {record_id}")
                feature_path = Path(row["feature_npz"])
                if not feature_path.is_absolute():
                    feature_path = (manifest_path.parent / feature_path).resolve()
                with np.load(feature_path, allow_pickle=False) as archive:
                    names = tuple(str(value) for value in archive["feature_names"].tolist())
                    if names != self.physiology_feature_names:
                        raise ValueError(f"physiology feature order mismatch: {feature_path}")
                    self.physiology_records[record_id] = {
                        "features": archive["features"].astype(np.float32, copy=True),
                        "valid": archive["valid"].astype(bool, copy=True),
                    }
        missing = [
            str(record["record_id"])
            for record in self.records
            if str(record["record_id"]) not in self.physiology_records
        ]
        if missing:
            raise ValueError(f"missing physiology features for records: {missing[:5]}")

    def __getitem__(self, index: int) -> Dict[str, object]:
        item = super().__getitem__(index)
        record_index, start = self.index[index]
        record_id = str(self.records[record_index]["record_id"])
        cached = self.physiology_records[record_id]
        history_slice = slice(start, start + self.history_epochs)
        current_index = start + self.history_epochs - 1
        future_indices = np.asarray(
            [current_index + horizon for horizon in self.future_horizons], dtype=np.int64
        )
        history = cached["features"][history_slice].copy()
        history_valid = cached["valid"][history_slice].copy() & np.isfinite(history)
        current = history[-1].copy()
        current_valid = history_valid[-1].copy()
        future = cached["features"][future_indices].copy()
        future_valid = cached["valid"][future_indices].copy() & np.isfinite(future)
        history_present = item["history_present"].numpy()
        future_present = item["future_present"].numpy()
        for group, names in self.physiology_feature_groups.items():
            feature_indices = [self.physiology_feature_names.index(name) for name in names]
            if group not in self.modalities:
                history_valid[:, feature_indices] = False
                future_valid[:, feature_indices] = False
                continue
            modality_index = self.modalities.index(group)
            history_valid[:, feature_indices] &= history_present[:, modality_index, None]
            future_valid[:, feature_indices] &= future_present[:, modality_index, None]
        current_valid = history_valid[-1].copy()
        history = (history - self.physiology_mean[None, :]) / self.physiology_std[None, :]
        current = (current - self.physiology_mean) / self.physiology_std
        future = (future - self.physiology_mean[None, :]) / self.physiology_std[None, :]
        history[~history_valid] = 0.0
        current[~current_valid] = 0.0
        future[~future_valid] = 0.0
        item.update(
            {
                "history_physiology": torch.from_numpy(history),
                "history_physiology_valid": torch.from_numpy(history_valid),
                "current_physiology": torch.from_numpy(current),
                "current_physiology_valid": torch.from_numpy(current_valid),
                "future_physiology": torch.from_numpy(future),
                "future_physiology_valid": torch.from_numpy(future_valid),
            }
        )
        return item


def balanced_class_weights(counts: np.ndarray) -> torch.Tensor:
    counts = counts.astype(np.float64)
    positive = counts > 0
    weights = np.zeros_like(counts, dtype=np.float64)
    if positive.any():
        weights[positive] = counts[positive].sum() / counts[positive]
        weights[positive] /= weights[positive].mean()
    return torch.tensor(weights, dtype=torch.float32)
