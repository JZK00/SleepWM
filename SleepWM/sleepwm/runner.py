"""Supervised custom-data example; this is not the paper's training protocol."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from torch.utils.data import DataLoader, Sampler

from . import SleepWM
from .data import SleepDataset
from .features import FEATURE_NAMES

CLAIM_SCOPE = "Supervised custom-data example; does not reproduce paper results."
CONDITIONS = ("complete", "eeg_lost", "all_lost")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_statistics(path):
    statistics = json.loads(Path(path).read_text(encoding="utf-8"))
    if statistics.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Statistics feature names/order differ from this code release")
    for key, size in (("signal_mean", 3), ("signal_std", 3),
                      ("feature_mean", 11), ("feature_std", 11)):
        value = np.asarray(statistics[key], dtype=float)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid normalization field: {key}")
        if key.endswith("std") and (value <= 0).any():
            raise ValueError(f"Normalization standard deviations must be positive: {key}")
    encoded = json.dumps(statistics, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return statistics, hashlib.sha256(encoded).hexdigest()


class RecordBlockSampler(Sampler):
    """Shuffle records and their windows while loading each record as one block."""

    def __init__(self, dataset, seed):
        self.groups = {}
        for index, (record_index, _) in enumerate(dataset.index):
            self.groups.setdefault(record_index, []).append(index)
        self.seed = seed
        self.epoch = 0
        self.length = len(dataset)

    def __len__(self):
        return self.length

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        groups = list(self.groups.values())
        rng.shuffle(groups)
        for group in groups:
            indices = list(group)
            rng.shuffle(indices)
            yield from indices


def make_training_datasets(manifest, statistics, config, stride):
    """Only train and validation signals may be opened by the training entry point."""
    options = dict(history_epochs=config["data"]["history_epochs"],
                   horizons=config["data"]["future_horizons"], stride=stride)
    return tuple(SleepDataset(manifest, statistics, split, **options)
                 for split in ("train", "val"))


def make_loader(dataset, batch_size, seed=None):
    sampler = None if seed is None else RecordBlockSampler(dataset, seed)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      shuffle=False, num_workers=0, drop_last=False)


def to_device(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def observation_mask(present, condition, gap_epochs=4):
    """Apply extra missingness without changing labels or physiological targets."""
    result = present.bool().clone()
    tail = min(gap_epochs, result.shape[1])
    if tail < 1:
        raise ValueError("gap_epochs must be positive")
    if condition == "random":
        for item in range(len(result)):
            # Complete, one lost modality, or all lost; preserve an earlier history.
            choice = int(torch.randint(0, 5, (1,)).item())
            duration = int(torch.randint(1, min(8, result.shape[1]) + 1, (1,)).item())
            if 1 <= choice <= 3:
                result[item, -duration:, choice - 1] = False
            elif choice == 4:
                result[item, -duration:, :] = False
    elif condition == "eeg_lost":
        result[:, -tail:, 0] = False
    elif condition == "all_lost":
        result[:, -tail:, :] = False
    elif condition != "complete":
        raise ValueError(f"Unknown observation condition: {condition}")
    return result


def supervised_loss(outputs, batch):
    for name in ("stage_logits", "future_physiology", "transition_risk"):
        if not torch.isfinite(outputs[name]).all():
            raise FloatingPointError(f"Non-finite model output: {name}")
    stage = F.cross_entropy(outputs["stage_logits"].reshape(-1, 5),
                            batch["stage"].reshape(-1).long())
    valid = batch["physiology_valid"].bool()
    if valid.any():
        physiology = F.smooth_l1_loss(outputs["future_physiology"][valid],
                                     batch["physiology"][valid])
    else:
        physiology = outputs["future_physiology"].sum() * 0.0
    risk = outputs["transition_risk"].clamp(1e-6, 1.0 - 1e-6)
    transition = F.binary_cross_entropy(risk, batch["transition"].float())
    total = stage + physiology + transition
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite training/validation loss")
    return total


def train_epoch(model, loader, optimizer, device, max_batches=None):
    model.train()
    loss_sum, windows, batches = 0.0, 0, 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = to_device(batch, device)
        present = observation_mask(batch["present"], "random")
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["signals"], present)
        loss = supervised_loss(outputs, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0,
                                      error_if_nonfinite=True)
        optimizer.step()
        size = len(batch["signals"])
        loss_sum += float(loss.detach()) * size
        windows += size
        batches += 1
    if not windows:
        raise ValueError("Training has no eligible windows")
    return {"loss": loss_sum / windows, "windows": windows, "batches": batches}


def endpoint_metrics(stage, logits, physiology, prediction, valid, transition, risk):
    predicted_stage = logits.argmax(axis=-1)
    valid = valid.astype(bool)
    event_true, event_score = transition.reshape(-1), risk.reshape(-1)
    positives = int(event_true.sum())
    return {
        "stage_macro_f1": float(f1_score(stage.reshape(-1), predicted_stage.reshape(-1),
                                         labels=list(range(5)), average="macro", zero_division=0)),
        "stage_accuracy": float(accuracy_score(stage.reshape(-1), predicted_stage.reshape(-1))),
        "physiology_mae_standardized": (float(np.abs(physiology[valid] - prediction[valid]).mean())
                                         if valid.any() else None),
        "physiology_valid_values": int(valid.sum()),
        "transition_average_precision": (float(average_precision_score(event_true, event_score))
                                          if positives else None),
        "transition_brier": float(np.square(event_score - event_true).mean()),
        "transition_positive_targets": positives,
        "forecast_targets": int(stage.size),
    }


@torch.no_grad()
def evaluate_model(model, loader, device, horizons, conditions=CONDITIONS,
                   gap_epochs=4, max_batches=None):
    model.eval()
    report = {}
    for condition in conditions:
        values = {name: [] for name in ("stage", "logits", "physiology", "prediction",
                                        "valid", "transition", "risk")}
        loss_sum, windows, batches = 0.0, 0, 0
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = to_device(batch, device)
            present = observation_mask(batch["present"], condition, gap_epochs)
            outputs = model(batch["signals"], present)
            loss = supervised_loss(outputs, batch)
            size = len(batch["signals"])
            loss_sum += float(loss) * size
            windows += size
            batches += 1
            tensors = (batch["stage"], outputs["stage_logits"], batch["physiology"],
                       outputs["future_physiology"], batch["physiology_valid"],
                       batch["transition"], outputs["transition_risk"])
            for key, value in zip(values, tensors):
                values[key].append(value.detach().cpu().numpy())
        if not windows:
            raise ValueError("Evaluation has no eligible windows")
        arrays = [np.concatenate(value, axis=0) for value in values.values()]
        by_horizon = {str(int(horizon) * 30): endpoint_metrics(*(value[:, index] for value in arrays))
                      for index, horizon in enumerate(horizons)}
        report[condition] = {
            "loss": loss_sum / windows, "windows": windows, "batches": batches,
            "overall": endpoint_metrics(*arrays), "by_horizon_seconds": by_horizon,
        }
    return report


def save_checkpoint(path, model, config, statistics, statistics_sha256, metadata):
    torch.save({
        "checkpoint_schema": 1, "claim_scope": metadata.get("claim_scope", CLAIM_SCOPE),
        "model_config": config,
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "normalization": statistics, "normalization_sha256": statistics_sha256,
        "feature_names": list(FEATURE_NAMES), "metadata": metadata,
    }, path)


def load_checkpoint(path, statistics_path, device):
    statistics, fingerprint = read_statistics(statistics_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("checkpoint_schema") != 1:
        raise ValueError("Unsupported checkpoint schema")
    if checkpoint.get("normalization_sha256") != fingerprint or checkpoint.get("normalization") != statistics:
        raise ValueError("Statistics differ from those used to train the checkpoint")
    if checkpoint.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Checkpoint feature names/order differ from this release")
    model = SleepWM(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")
