from __future__ import annotations

import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config, save_config
from .data import (
    EpochDataset,
    PhysioFeatureSequenceDataset,
    RecordBatchSampler,
    SequenceDataset,
    TransitionBalancedRecordBatchSampler,
)
from .metrics import classification_metrics
from .masking import random_full_biased_natural_modality_subset, random_natural_modality_subset


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, count: int = 1) -> None:
        self.total += float(value) * int(count)
        self.count += int(count)

    @property
    def average(self) -> float:
        return self.total / max(self.count, 1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def epoch_dataset(config: Config, split: str) -> EpochDataset:
    data = config["data"]
    return EpochDataset(
        data["manifest_path"],
        split,
        stored_modalities=data["stored_modalities"],
        modalities=data["modalities"],
        num_classes=int(data.get("num_classes", 5)),
        normalization_path=data.get("normalization_path"),
        label_fraction=float(data.get("label_fraction", 1.0)) if split == "train" else 1.0,
        subset_seed=int(data.get("label_subset_seed", config["experiment"]["seed"])),
    )


def sequence_dataset(config: Config, split: str) -> SequenceDataset:
    data = config["data"]
    return SequenceDataset(
        data["manifest_path"],
        split,
        history_epochs=int(data["history_epochs"]),
        future_horizons=data["future_horizons"],
        stored_modalities=data["stored_modalities"],
        modalities=data["modalities"],
        normalization_path=data.get("normalization_path"),
        sequence_stride=int(data.get("sequence_stride", 1)),
        sequence_start_offset=int(data.get("sequence_start_offset", 0)),
        num_classes=int(data.get("num_classes", 5)),
    )


def physio_feature_sequence_dataset(config: Config, split: str) -> PhysioFeatureSequenceDataset:
    data = config["data"]
    physiology = config["physiology"]
    return PhysioFeatureSequenceDataset(
        data["manifest_path"],
        split,
        history_epochs=int(data["history_epochs"]),
        future_horizons=data["future_horizons"],
        stored_modalities=data["stored_modalities"],
        modalities=data["modalities"],
        normalization_path=data.get("normalization_path"),
        sequence_stride=int(data.get("sequence_stride", 1)),
        sequence_start_offset=int(data.get("sequence_start_offset", 0)),
        num_classes=int(data.get("num_classes", 5)),
        feature_manifest_path=physiology["feature_manifest_path"],
        feature_statistics_path=physiology["feature_statistics_path"],
    )


def data_loader(dataset, config: Config, shuffle: bool) -> DataLoader:
    train = config["train"]
    device = resolve_device(str(train.get("device", "cuda")))
    common = {
        "num_workers": int(train.get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    transition_probability = train.get("transition_window_probability")
    if shuffle and isinstance(dataset, SequenceDataset) and transition_probability is not None:
        sampler = TransitionBalancedRecordBatchSampler(
            dataset,
            batch_size=int(train["batch_size"]),
            seed=int(config["experiment"]["seed"]),
            transition_probability=float(transition_probability),
        )
        return DataLoader(dataset, batch_sampler=sampler, **common)
    if shuffle and isinstance(dataset, (EpochDataset, SequenceDataset)):
        sampler = RecordBatchSampler(
            dataset,
            batch_size=int(train["batch_size"]),
            seed=int(config["experiment"]["seed"]),
        )
        return DataLoader(dataset, batch_sampler=sampler, **common)
    return DataLoader(dataset, batch_size=int(train["batch_size"]), shuffle=shuffle, drop_last=False, **common)


def output_directory(config: Config) -> Path:
    directory = Path(config["train"]["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def prepare_run(config: Config) -> Path:
    directory = output_directory(config)
    save_config(directory / "resolved_config.yaml", config)
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "command": sys.argv,
    }
    write_json(directory / "environment.json", metadata)
    return directory


def write_json(path: str | Path, payload: Dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def save_checkpoint(path: str | Path, payload: Dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def load_checkpoint(path: str | Path) -> Dict[str, object]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint is not a mapping: {checkpoint_path}")
    return checkpoint


def load_encoder_state(encoder: torch.nn.Module, checkpoint_path: str | Path) -> None:
    checkpoint = load_checkpoint(checkpoint_path)
    state = checkpoint.get("encoder_state")
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint does not contain encoder_state: {checkpoint_path}")
    encoder.load_state_dict(state, strict=True)


def classification_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    num_classes: int,
    class_weights: Optional[torch.Tensor] = None,
    grad_clip: float = 0.0,
    forced_presence: Optional[torch.Tensor] = None,
    modality_subset_sampling: str = "none",
    full_modality_probability: float = 0.5,
) -> Dict[str, object]:
    training = optimizer is not None
    model.train(training)
    meter = AverageMeter()
    all_logits = []
    all_labels = []
    for batch in loader:
        signals = batch["signals"].to(device=device, dtype=torch.float32)
        labels = batch["label"].to(device=device, dtype=torch.long)
        present = batch["modality_present"].to(device=device, dtype=torch.bool)
        if training and modality_subset_sampling == "uniform_nonempty":
            present = random_natural_modality_subset(present)
        elif training and modality_subset_sampling == "full_biased_nonempty":
            present = random_full_biased_natural_modality_subset(
                present,
                full_modality_probability=full_modality_probability,
            )
        elif modality_subset_sampling != "none":
            raise ValueError(f"unknown modality subset sampling: {modality_subset_sampling}")
        if forced_presence is not None:
            present = present & forced_presence.to(device=device, dtype=torch.bool).reshape(1, -1)
        with torch.set_grad_enabled(training):
            logits = model(signals, present)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        meter.update(float(loss.detach().cpu()), signals.shape[0])
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
    metrics = classification_metrics(torch.cat(all_logits), torch.cat(all_labels), num_classes)
    metrics["loss"] = meter.average
    return metrics


def trainable_parameters(model: torch.nn.Module) -> Iterable[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def model_size(model: torch.nn.Module) -> Dict[str, int]:
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def profile_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> Dict[str, float]:
    model.eval()
    elapsed = 0.0
    samples = 0
    measured_batches = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            signals = batch["signals"].to(device=device, dtype=torch.float32)
            present = batch["modality_present"].to(device=device, dtype=torch.bool)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            model(signals, present)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if batch_index > 0:
                elapsed += time.perf_counter() - start
                samples += signals.shape[0]
                measured_batches += 1
            if measured_batches >= max_batches:
                break
    return {
        "inference_ms_per_sample": 1000.0 * elapsed / max(samples, 1),
        "profiled_samples": float(samples),
    }
