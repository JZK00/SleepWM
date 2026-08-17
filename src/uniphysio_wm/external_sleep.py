from __future__ import annotations

import hashlib
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .preprocessing import ChannelSpec, resolve_channel_spec


MODALITIES = ("EEG", "ECG", "EMG")
ISRUC_CHANNEL_RULES = {
    "EEG": {
        "direct": ("C4-A1", "C4-M1", "C3-A2", "C3-M2"),
        "pairs": (("C4", "A1"), ("C4", "M1"), ("C3", "A2"), ("C3", "M2")),
    },
    "ECG": {"direct": ("X2", "25"), "pairs": ()},
    "EMG": {"direct": ("X1", "24"), "pairs": ()},
}
SLEEP_EDFX_CHANNEL_RULES = {
    "EEG": {"direct": ("EEG Fpz-Cz", "EEG Pz-Oz"), "pairs": ()},
    "ECG": {"direct": (), "pairs": ()},
    "EMG": {"direct": ("EMG submental",), "pairs": ()},
}

ISRUC_STAGE_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}
SLEEP_EDFX_STAGE_MAP = {
    "sleep stage w": 0,
    "sleep stage 1": 1,
    "sleep stage 2": 2,
    "sleep stage 3": 3,
    "sleep stage 4": 3,
    "sleep stage r": 4,
}


def subject_split(
    subject: str,
    seed: int = 20260804,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> str:
    digest = hashlib.sha1(f"{seed}:{subject}".encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(16**12)
    if value < train_fraction:
        return "train"
    if value < train_fraction + val_fraction:
        return "val"
    return "test"


def canonical_channel_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def resolve_channel(ch_names: Sequence[str], requested: str | None) -> str | None:
    if requested is None:
        return None
    canonical = canonical_channel_name(requested)
    return next((name for name in ch_names if canonical_channel_name(name) == canonical), None)


def resolve_external_channels(ch_names: Sequence[str], dataset: str) -> dict[str, ChannelSpec | None]:
    rules = ISRUC_CHANNEL_RULES if dataset == "isruc" else SLEEP_EDFX_CHANNEL_RULES
    return {
        modality: resolve_channel_spec(ch_names, rules[modality]["direct"], rules[modality]["pairs"])
        for modality in MODALITIES
    }


def map_isruc_labels(values: Sequence[int]) -> np.ndarray:
    return np.asarray([ISRUC_STAGE_MAP.get(int(value), -1) for value in values], dtype=np.int64)


def map_sleep_edfx_label(description: str) -> int:
    return SLEEP_EDFX_STAGE_MAP.get(str(description).strip().lower(), -1)


def sleep_edfx_pair_key(path: str | Path) -> str:
    stem = Path(path).stem
    if len(stem) < 6:
        raise ValueError(f"unexpected Sleep-EDF filename: {Path(path).name}")
    return stem[:6]


def sleep_edfx_subject(path: str | Path) -> str:
    stem = Path(path).stem
    if len(stem) < 5:
        raise ValueError(f"unexpected Sleep-EDF filename: {Path(path).name}")
    return f"sleep_edfx:{stem[:5]}"


@contextmanager
def edf_compatible_path(path: str | Path) -> Iterator[Path]:
    """Expose EDF-compatible .rec input through an .edf suffix for older MNE."""

    source = Path(path).resolve()
    if source.suffix.lower() == ".edf":
        yield source
        return
    with tempfile.TemporaryDirectory(prefix="uniphysio_edf_") as directory:
        alias = Path(directory) / f"{source.stem}.edf"
        os.symlink(source, alias)
        yield alias


def expand_sleep_edfx_annotations(
    onsets: Sequence[float],
    durations: Sequence[float],
    descriptions: Sequence[str],
    n_epochs: int,
    epoch_seconds: int = 30,
) -> np.ndarray:
    labels = np.full(int(n_epochs), -1, dtype=np.int64)
    for onset, duration, description in zip(onsets, durations, descriptions):
        label = map_sleep_edfx_label(description)
        if label < 0 or float(duration) <= 0:
            continue
        start = max(0, int(round(float(onset) / epoch_seconds)))
        count = max(1, int(round(float(duration) / epoch_seconds)))
        end = min(n_epochs, start + count)
        labels[start:end] = label
    return labels


def trimmed_sleep_interval(labels: np.ndarray, wake_margin_epochs: int) -> tuple[int, int]:
    asleep = np.flatnonzero((labels >= 1) & (labels <= 4))
    if asleep.size == 0:
        raise ValueError("hypnogram has no scored sleep epochs")
    start = max(0, int(asleep[0]) - int(wake_margin_epochs))
    end = min(int(labels.shape[0]), int(asleep[-1]) + int(wake_margin_epochs) + 1)
    return start, end
