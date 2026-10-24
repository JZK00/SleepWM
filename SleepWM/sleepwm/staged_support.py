"""Data, observation protocols and metrics shared by staged training."""
from __future__ import annotations

import itertools
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Sampler, Subset

from .data import SleepDataset, load_statistics




class PhysioFeatureSequenceDataset(SleepDataset):
    """Expose original stage-batch fields over validated, timestamped records."""

    def __init__(self, config, split):
        data = config['data']
        if tuple(data.get('modalities', ('EEG', 'ECG', 'EMG'))) != ('EEG', 'ECG', 'EMG'):
            raise ValueError('staged training expects EEG, ECG and EMG, in this order')
        if int(data.get('num_classes', 5)) != 5:
            raise ValueError('staged training expects five sleep stages')
        stride = int(data.get('sequence_stride', 8))
        offset = int(data.get('sequence_start_offset', 0))
        if stride < 1 or offset < 0:
            raise ValueError('sequence_stride must be positive and offset nonnegative')
        super().__init__(
            data['manifest_path'], data['normalization_path'], split,
            history_epochs=int(data['history_epochs']),
            horizons=data['future_horizons'], stride=1 if offset else stride,
        )
        if offset:
            self.index = [(r, s) for r, s in self.index if s >= offset and (s - offset) % stride == 0]
            if not self.index:
                raise ValueError(f'no continuous {split} windows at requested start offset')
        self.future_horizons = self.horizons
        self.feature_names = tuple(self.stats['feature_names'])
        configured_features = config.get('physiology', {}).get('feature_names')
        if configured_features is not None and tuple(configured_features) != self.feature_names:
            raise ValueError('physiology feature names/order differ from prepared data')
        self.future_label_counts = np.zeros(5, dtype=np.int64)
        self.transition_mask, self.transition_window = [], []
        labels = load_label_map(self)
        for row in self.records:
            row['path'] = row['npz_path']
        for record_index, start in self.index:
            values = labels[self.records[record_index]['record_id']]
            current = start + self.history_epochs - 1
            future = values[current + np.asarray(self.horizons)]
            self.future_label_counts += np.bincount(future, minlength=5)
            # Match the original transition-balanced sampling definition.
            changed = tuple(bool(value) for value in future != values[current])
            self.transition_mask.append(changed)
            self.transition_window.append(any(changed))

    @property
    def transition_window_fraction(self):
        return float(np.mean(self.transition_window))

    @property
    def transition_fraction_by_horizon(self):
        mask = np.asarray(self.transition_mask, dtype=bool)
        return {str(h): float(mask[:, i].mean()) for i, h in enumerate(self.horizons)}

    def __getitem__(self, item):
        current = super().__getitem__(item)
        _, start = self.index[item]
        end = start + self.history_epochs
        targets = end - 1 + np.asarray(self.horizons)
        record = self._cache
        signal_mean = np.asarray(self.stats['signal_mean'], dtype=np.float32)
        signal_std = np.asarray(self.stats['signal_std'], dtype=np.float32)
        future_present = record['present'][targets]
        future_signals = (record['signals'][targets] - signal_mean[None, :, None]) / signal_std[None, :, None]
        future_signals = np.where(future_present[..., None], future_signals, 0).astype(np.float32)
        history_valid = record['feature_valid'][start:end]
        feature_mean = np.asarray(self.stats['feature_mean'], dtype=np.float32)
        feature_std = np.asarray(self.stats['feature_std'], dtype=np.float32)
        history_physiology = (record['features'][start:end] - feature_mean) / feature_std
        history_physiology = np.where(history_valid, history_physiology, 0).astype(np.float32)
        return {
            'history_signals': current['signals'],
            'history_present': current['present'],
            'history_labels': torch.from_numpy(record['labels'][start:end].copy()),
            'future_signals': torch.from_numpy(future_signals),
            'future_present': torch.from_numpy(future_present.copy()),
            'future_labels': current['stage'],
            'history_physiology': torch.from_numpy(history_physiology),
            'history_physiology_valid': torch.from_numpy(history_valid.copy()),
            'current_physiology': torch.from_numpy(history_physiology[-1].copy()),
            'current_physiology_valid': torch.from_numpy(history_valid[-1].copy()),
            'future_physiology': current['physiology'],
            'future_physiology_valid': current['physiology_valid'],
            'record_id': current['record_id'], 'subject': current['subject'],
            'start_epoch': torch.tensor(start, dtype=torch.long),
        }


def physio_feature_sequence_dataset(config, split):
    return PhysioFeatureSequenceDataset(config, split)


def load_label_map(dataset):
    """Open labels only for this dataset's selected split."""
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    result = {}
    for record in dataset.records:
        with np.load(record['npz_path'], allow_pickle=False) as archive:
            result[str(record['record_id'])] = archive['labels'].astype(np.int64, copy=True)
    return result


class _BoundedLoader:
    """Limit work for an explicitly requested smoke run without consuming more batches."""

    def __init__(self, loader, maximum):
        self.loader, self.maximum = loader, int(maximum)
        self.dataset, self.batch_sampler = loader.dataset, loader.batch_sampler

    def __iter__(self):
        return itertools.islice(self.loader, self.maximum)

    def __len__(self):
        return min(len(self.loader), self.maximum)


def data_loader(dataset, config, shuffle):
    train = config['train']
    device = resolve_device(str(train.get('device', 'cpu')))
    batch_size = int(train['batch_size'])
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    common = {'num_workers': int(train.get('num_workers', 0)), 'pin_memory': device.type == 'cuda'}
    if shuffle and isinstance(dataset, PhysioFeatureSequenceDataset):
        options = dict(batch_size=batch_size, seed=int(config['experiment']['seed']))
        if train.get('transition_window_probability') is not None:
            sampler = TransitionBalancedRecordBatchSampler(
                dataset, transition_probability=float(train['transition_window_probability']), **options)
        else:
            sampler = RecordBatchSampler(dataset, **options)
        loader = DataLoader(dataset, batch_sampler=sampler, **common)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, **common)
    maximum = train.get('max_batches')
    if maximum is not None:
        if int(maximum) < 1:
            raise ValueError('max_batches must be positive when provided')
        return _BoundedLoader(loader, maximum)
    return loader


def resolve_device(name):
    device = torch.device(name)
    if device.type == 'cuda':
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            raise ValueError('Set CUDA_VISIBLE_DEVICES explicitly before a CUDA run')
        if not torch.cuda.is_available():
            raise ValueError('CUDA was requested but is unavailable; use --device cpu')
    return device


def save_checkpoint(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    statistics_path = payload.get('config', {}).get('data', {}).get('normalization_path')
    if statistics_path:
        statistics = load_statistics(statistics_path)
        payload['normalization'] = statistics
        canonical = json.dumps(statistics, sort_keys=True, separators=(',', ':'), allow_nan=False)
        payload['normalization_sha256'] = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    torch.save(payload, output)


def load_checkpoint(path):
    """Load a trusted checkpoint produced by the training tools."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'checkpoint not found: {path}')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f'checkpoint is not a mapping: {path}')
    return checkpoint


def average_precision(target, score):
    """Tie-invariant AP; use zero for an event-free stage-selection sample."""
    target, score = np.asarray(target), np.asarray(score)
    if target.shape != score.shape or target.ndim != 1 or not target.size:
        raise ValueError('AP expects equally sized, nonempty 1-D target/score arrays')
    if not np.isin(target, (0, 1)).all() or not np.isfinite(score).all():
        raise ValueError('AP requires binary targets and finite scores')
    return float(average_precision_score(target, score)) if target.any() else 0.0




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


def balanced_class_weights(counts: np.ndarray) -> torch.Tensor:
    counts = counts.astype(np.float64)
    positive = counts > 0
    weights = np.zeros_like(counts, dtype=np.float64)
    if positive.any():
        weights[positive] = counts[positive].sum() / counts[positive]
        weights[positive] /= weights[positive].mean()
    return torch.tensor(weights, dtype=torch.float32)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_size(model: torch.nn.Module) -> Dict[str, int]:
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


@dataclass(frozen=True)
class DynamicObservationSpec:
    name: str
    missing_epochs: Mapping[str, int]
    recovery_epochs: int = 0
    profile: str = "hard"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("dynamic observation condition requires a name")
        if self.recovery_epochs < 0:
            raise ValueError("recovery_epochs must be nonnegative")
        if self.profile not in {"hard", "linear_decay"}:
            raise ValueError(f"unsupported observation profile: {self.profile}")
        if not self.missing_epochs:
            raise ValueError("at least one modality interruption is required")
        if any(int(value) < 0 for value in self.missing_epochs.values()):
            raise ValueError("missing durations must be nonnegative")


def dynamic_observation_view(
    history_signals: torch.Tensor,
    history_present: torch.Tensor,
    modalities: Sequence[str],
    spec: DynamicObservationSpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a causal block interruption and optional pre-cutoff recovery."""

    if history_signals.ndim != 4:
        raise ValueError("history_signals must be [batch, epochs, modalities, samples]")
    if history_present.shape != history_signals.shape[:3]:
        raise ValueError("history_present must match the first three signal dimensions")
    if len(modalities) != history_signals.shape[2]:
        raise ValueError("modality names do not match the signal tensor")
    unknown = set(spec.missing_epochs) - set(modalities)
    if unknown:
        raise ValueError(f"unknown interrupted modalities: {sorted(unknown)}")

    epoch_count = history_signals.shape[1]
    stop = max(0, epoch_count - int(spec.recovery_epochs))
    quality = history_present.to(dtype=history_signals.dtype).clone()
    for modality, requested_duration in spec.missing_epochs.items():
        duration = min(int(requested_duration), stop)
        if duration == 0:
            continue
        modality_index = modalities.index(modality)
        start = stop - duration
        if spec.profile == "hard":
            quality[:, start:stop, modality_index] = 0.0
        else:
            decay = torch.linspace(
                1.0 - 1.0 / duration,
                0.0,
                duration,
                device=history_signals.device,
                dtype=history_signals.dtype,
            )
            quality[:, start:stop, modality_index] = decay.reshape(1, -1)

    observed = history_present.bool() & (quality > 0.0)
    signals = history_signals * quality.unsqueeze(-1)
    signals = signals.masked_fill(~observed.unsqueeze(-1), 0.0)
    return signals, observed, quality


def primary_dynamic_observation_specs(
    modalities: Sequence[str],
    duration_epochs: Sequence[int],
    recovery_epochs: Sequence[int],
) -> tuple[DynamicObservationSpec, ...]:
    specs = []
    for modality in modalities:
        for duration in duration_epochs:
            specs.append(
                DynamicObservationSpec(
                    name=f"tail_{modality.lower()}_{int(duration)}ep",
                    missing_epochs={modality: int(duration)},
                )
            )
    for duration in duration_epochs:
        specs.append(
            DynamicObservationSpec(
                name=f"tail_all_{int(duration)}ep",
                missing_epochs={modality: int(duration) for modality in modalities},
            )
        )

    recovery_duration = max(1, min(4, max(int(value) for value in duration_epochs)))
    for recovered in recovery_epochs:
        specs.append(
            DynamicObservationSpec(
                name=f"recover_all_{recovery_duration}ep_after_{int(recovered)}ep",
                missing_epochs={modality: recovery_duration for modality in modalities},
                recovery_epochs=int(recovered),
            )
        )
    specs.append(
        DynamicObservationSpec(
            name="linear_decay_all_4ep",
            missing_epochs={modality: 4 for modality in modalities},
            profile="linear_decay",
        )
    )
    asynchronous = {
        modality: max(1, 4 // (index + 1))
        for index, modality in enumerate(modalities)
    }
    specs.append(
        DynamicObservationSpec(
            name="asynchronous_tail",
            missing_epochs=asynchronous,
        )
    )
    return tuple(specs)


def dynamic_view(
    signals: torch.Tensor,
    present: torch.Tensor,
    modalities: Sequence[str],
    spec: Optional[DynamicObservationSpec],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if spec is None:
        return signals, present, present.to(signals.dtype)
    return dynamic_observation_view(signals, present, modalities, spec)


def route_outputs(model, output: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    base_states = output["belief_base_predicted_states"]
    persistence_delta = (
        output["belief_persistence_trajectory"][:, -1]
        - output["history_epoch_states"][:, -1]
    )
    persistence_states = base_states + persistence_delta.unsqueeze(1)

    persistence_logits = output["belief_base_stage_logits"] + model.stage_head(
        persistence_states
    ) - model.stage_head(base_states)
    direct_current = output["current_stage_logits"]
    persistence_current = direct_current
    if model.current_stage_head is not None:
        base_history = output["belief_base_corrected_history_state"]
        persistence_history = base_history + persistence_delta
        current_delta = model.current_stage_head(
            persistence_history
        ) - model.current_stage_head(base_history)
        persistence_current = direct_current + current_delta
        persistence_logits = persistence_logits + current_delta.unsqueeze(1)

    persistence_task, persistence_physiology_task = model._task_residuals(
        persistence_states, output["observation_reliability"]
    )
    base_task, base_physiology_task = model._task_residuals(
        base_states, output["observation_reliability"]
    )
    persistence_logits = persistence_logits + persistence_task - base_task

    def physiology_from_states(
        states: torch.Tensor, task_residual: torch.Tensor
    ) -> torch.Tensor:
        values = []
        offset = 0
        for group, size in model.physiology_group_sizes.items():
            current = output["current_physiology"][:, offset : offset + size]
            values.append(
                current.unsqueeze(1)
                + model.future_physiology_delta_heads[group](states)
            )
            offset += size
        return torch.cat(values, dim=-1) + task_residual

    base_physiology = physiology_from_states(base_states, base_physiology_task)
    belief_physiology_task = model._task_residuals(
        output["predicted_states"], output["observation_reliability"]
    )[1]
    belief_physiology = physiology_from_states(
        output["predicted_states"], belief_physiology_task
    )
    direct_physiology = (
        output["future_physiology"] - belief_physiology + base_physiology
    )
    persistence_physiology = direct_physiology + physiology_from_states(
        persistence_states, persistence_physiology_task
    ) - base_physiology

    belief_current = output.get("belief_current_stage_logits", direct_current)
    return {
        "direct_incomplete": {
            "states": base_states,
            "stage_logits": output["belief_base_stage_logits"],
            "current_stage_logits": direct_current,
            "future_physiology": direct_physiology,
        },
        "static_persistence": {
            "states": persistence_states,
            "stage_logits": persistence_logits,
            "current_stage_logits": persistence_current,
            "future_physiology": persistence_physiology,
        },
        "recursive_belief": {
            "states": output["predicted_states"],
            "stage_logits": output["stage_logits"],
            "current_stage_logits": belief_current,
            "future_physiology": output["future_physiology"],
        },
    }


def event_at(event: str, previous: int, current: int) -> bool:
    if previous < 0 or current < 0:
        return False
    if event == "transition":
        return previous != current
    if event == "sleep_onset":
        return previous == 0 and current != 0
    if event == "wake_onset":
        return previous != 0 and current == 0
    if event == "rem_onset":
        return previous != 4 and current == 4
    if event == "n3_onset":
        return previous != 3 and current == 3
    raise ValueError(f"unknown event: {event}")


def next_event_offset(labels: np.ndarray, current_index: int, event: str, max_horizon: int) -> int:
    end = min(len(labels) - 1, current_index + max_horizon)
    for index in range(current_index + 1, end + 1):
        if event_at(event, int(labels[index - 1]), int(labels[index])):
            return index - current_index
    return max_horizon + 1


def event_risk_scores(current_probabilities: torch.Tensor, future_probabilities: torch.Tensor) -> dict[str, torch.Tensor]:
    overlap = (current_probabilities.unsqueeze(1) * future_probabilities).sum(dim=-1)
    raw = {
        "transition": 1.0 - overlap,
        "sleep_onset": current_probabilities[:, 0:1] * (1.0 - future_probabilities[:, :, 0]),
        "wake_onset": (1.0 - current_probabilities[:, 0:1]) * future_probabilities[:, :, 0],
        "rem_onset": (1.0 - current_probabilities[:, 4:5]) * future_probabilities[:, :, 4],
        "n3_onset": (1.0 - current_probabilities[:, 3:4]) * future_probabilities[:, :, 3],
    }
    return {name: torch.cummax(value.clamp(0.0, 1.0), dim=1).values for name, value in raw.items()}


def confusion_matrix(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> np.ndarray:
    prediction = logits.argmax(dim=-1).detach().cpu().numpy().reshape(-1)
    truth = labels.detach().cpu().numpy().reshape(-1)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, predicted in zip(truth.tolist(), prediction.tolist()):
        if 0 <= target < num_classes and 0 <= predicted < num_classes:
            matrix[target, predicted] += 1
    return matrix


def metrics_from_confusion(matrix: np.ndarray) -> Dict[str, object]:
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / max(total, 1))
    recalls = []
    per_class_f1 = []
    for class_index in range(matrix.shape[0]):
        true_positive = float(matrix[class_index, class_index])
        false_positive = float(matrix[:, class_index].sum() - true_positive)
        false_negative = float(matrix[class_index, :].sum() - true_positive)
        support = float(matrix[class_index, :].sum())
        if support > 0:
            recalls.append(true_positive / support)
        precision = true_positive / max(true_positive + false_positive, 1.0)
        recall = true_positive / max(true_positive + false_negative, 1.0)
        score = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        per_class_f1.append(float(score))

    row_marginal = matrix.sum(axis=1).astype(np.float64)
    column_marginal = matrix.sum(axis=0).astype(np.float64)
    expected = float((row_marginal * column_marginal).sum() / max(total * total, 1))
    kappa = (accuracy - expected) / max(1.0 - expected, 1e-12)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(per_class_f1)) if per_class_f1 else 0.0,
        "cohen_kappa": float(kappa),
        "per_class_f1": per_class_f1,
        "confusion_matrix": matrix.tolist(),
        "samples": total,
    }


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int = 5) -> Dict[str, object]:
    return metrics_from_confusion(confusion_matrix(logits, labels, num_classes))


def _pearson(prediction: np.ndarray, target: np.ndarray) -> Optional[float]:
    if len(prediction) < 2:
        return None
    x = prediction.astype(np.float64) - float(np.mean(prediction))
    y = target.astype(np.float64) - float(np.mean(target))
    denominator = float(np.sqrt(np.square(x).sum() * np.square(y).sum()))
    return float((x * y).sum() / denominator) if denominator > 1e-12 else None


def standardized_physiology_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    feature_names: Sequence[str],
    feature_groups: Dict[str, Sequence[str]],
    horizons: Sequence[int] | None = None,
) -> Dict[str, object]:
    def available_mean(values, *, required=False):
        values = np.asarray(values, dtype=np.float64)
        available = values[np.isfinite(values)]
        if available.size:
            return float(available.mean())
        if required:
            raise ValueError(
                "No valid finite physiology targets are available for staged evaluation; "
                "check feature extraction, modality availability and the validation split"
            )
        return None

    predicted = prediction.detach().cpu().numpy()
    targets = target.detach().cpu().numpy()
    mask = valid.detach().cpu().numpy().astype(bool)
    if predicted.shape != targets.shape or mask.shape != targets.shape:
        raise ValueError("physiology prediction, target, and validity shapes must match")
    if predicted.shape[-1] != len(feature_names):
        raise ValueError("physiology feature count does not match names")
    if not np.isfinite(predicted[mask]).all() or not np.isfinite(targets[mask]).all():
        raise FloatingPointError("Valid physiology predictions and targets must be finite")
    per_feature = {}
    for feature_index, name in enumerate(feature_names):
        selected = mask[..., feature_index]
        errors = np.abs(
            predicted[..., feature_index][selected] - targets[..., feature_index][selected]
        )
        per_feature[name] = {
            "normalized_mae": float(errors.mean()) if errors.size else None,
            "pearson_r": _pearson(
                predicted[..., feature_index][selected], targets[..., feature_index][selected]
            ),
            "valid_targets": int(selected.sum()),
        }
    by_group = {}
    for group, names in feature_groups.items():
        by_group[group] = {
            "mean_normalized_mae": available_mean(
                [per_feature[name]["normalized_mae"] for name in names]
            ),
            "mean_pearson_r": available_mean(
                [per_feature[name]["pearson_r"] for name in names]
            ),
        }
    result: Dict[str, object] = {
        "all_features": {
            "mean_normalized_mae": available_mean(
                [per_feature[name]["normalized_mae"] for name in feature_names], required=True
            ),
            "mean_pearson_r": available_mean(
                [per_feature[name]["pearson_r"] for name in feature_names]
            ),
        },
        "by_group": by_group,
        "per_feature": per_feature,
    }
    if horizons is not None:
        if predicted.ndim != 3 or predicted.shape[1] != len(horizons):
            raise ValueError("horizon metrics require [samples, horizons, features]")
        by_horizon = {}
        for horizon_index, horizon in enumerate(horizons):
            values = []
            correlations = []
            for feature_index in range(len(feature_names)):
                selected = mask[:, horizon_index, feature_index]
                errors = np.abs(
                    predicted[selected, horizon_index, feature_index]
                    - targets[selected, horizon_index, feature_index]
                )
                values.append(float(errors.mean()) if errors.size else None)
                correlations.append(
                    _pearson(
                        predicted[selected, horizon_index, feature_index],
                        targets[selected, horizon_index, feature_index],
                    )
                )
            by_horizon[str(horizon)] = {
                "mean_normalized_mae": available_mean(values, required=True),
                "mean_pearson_r": available_mean(correlations),
            }
        result["by_horizon"] = by_horizon
    return result
