from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict

import yaml


Config = Dict[str, Any]


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    config = _expand_environment(payload)
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    for section in ("experiment", "data", "model", "train"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"missing configuration section: {section}")

    kind = str(config["experiment"].get("kind", ""))
    if kind not in {"baseline", "pretrain", "downstream", "forecast"}:
        raise ValueError(f"unsupported experiment kind: {kind}")

    modalities = config["data"].get("modalities")
    stored = config["data"].get("stored_modalities")
    if not isinstance(modalities, list) or not modalities:
        raise ValueError("data.modalities must be a non-empty list")
    if not isinstance(stored, list) or not stored:
        raise ValueError("data.stored_modalities must be a non-empty list")
    unknown = set(modalities) - set(stored)
    if unknown:
        raise ValueError(f"selected modalities are not stored: {sorted(unknown)}")

    if kind == "pretrain" and "objective" not in config:
        raise ValueError("pretraining config requires objective section")
    if kind == "pretrain":
        teacher_target = str(config["objective"].get("teacher_target", "tokenizer"))
        if teacher_target not in {"tokenizer", "contextual"}:
            raise ValueError("objective.teacher_target must be 'tokenizer' or 'contextual'")
    if kind == "downstream" and "downstream" not in config:
        raise ValueError("downstream config requires downstream section")
    if kind == "downstream":
        sampling = str(config["downstream"].get("modality_subset_sampling", "none"))
        if sampling not in {"none", "uniform_nonempty", "full_biased_nonempty"}:
            raise ValueError(f"unsupported downstream modality subset sampling: {sampling}")
        full_probability = float(config["downstream"].get("full_modality_probability", 0.5))
        if not 0.0 <= full_probability <= 1.0:
            raise ValueError("downstream.full_modality_probability must be in [0, 1]")
    if kind == "forecast":
        if "forecast" not in config:
            raise ValueError("forecast config requires forecast section")
        horizons = config["data"].get("future_horizons", [])
        if not horizons or any(int(horizon) < 1 for horizon in horizons):
            raise ValueError("future_horizons must contain positive epoch offsets")


def with_manifest(config: Config, manifest_path: str | None) -> Config:
    resolved = copy.deepcopy(config)
    if manifest_path:
        resolved["data"]["manifest_path"] = str(Path(manifest_path).resolve())
    return resolved


def with_normalization(config: Config, normalization_path: str | None) -> Config:
    resolved = copy.deepcopy(config)
    if normalization_path:
        resolved["data"]["normalization_path"] = str(Path(normalization_path).resolve())
    return resolved


def save_config(path: str | Path, config: Config) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
