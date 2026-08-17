from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


CAP_STAGE_LABELS = {
    "SLEEP-S0": 0,
    "SLEEP-S1": 1,
    "SLEEP-S2": 2,
    "SLEEP-S3": 3,
    "SLEEP-S4": 3,
    "SLEEP-REM": 4,
}


@dataclass(frozen=True)
class StageEvent:
    onset_seconds: float
    duration_seconds: float
    label: int


@dataclass(frozen=True)
class ChannelSpec:
    sources: tuple[str, ...]
    weights: tuple[float, ...]
    display_name: str


def _canonical_channel_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _find_exact_channel(ch_names: Sequence[str], requested: str) -> str | None:
    canonical = _canonical_channel_name(requested)
    return next((name for name in ch_names if _canonical_channel_name(name) == canonical), None)


def resolve_channel_spec(
    ch_names: Sequence[str],
    direct_candidates: Iterable[str],
    derived_pairs: Iterable[Sequence[str]] = (),
) -> ChannelSpec | None:
    """Resolve a recorded channel or a bipolar signal derived from two channels."""

    for candidate in direct_candidates:
        source = _find_exact_channel(ch_names, candidate)
        if source is not None:
            return ChannelSpec((source,), (1.0,), source)

    for pair in derived_pairs:
        if len(pair) != 2:
            raise ValueError(f"derived channel pairs must contain two names: {pair}")
        positive = _find_exact_channel(ch_names, pair[0])
        negative = _find_exact_channel(ch_names, pair[1])
        if positive is not None and negative is not None:
            return ChannelSpec(
                (positive, negative),
                (1.0, -1.0),
                f"{positive}-{negative}",
            )
    return None


def materialize_channel(signal_by_name: dict[str, np.ndarray], spec: ChannelSpec) -> np.ndarray:
    values = [signal_by_name[source] * weight for source, weight in zip(spec.sources, spec.weights)]
    return np.add.reduce(values, dtype=np.float32).astype(np.float32, copy=False)


def cap_unit_scale(spec: ChannelSpec, original_units: dict[str, str]) -> float:
    """Return the factor needed after MNE conversion to express CAP signals in volts."""

    units = [str(original_units.get(source, "")).strip().lower() for source in spec.sources]
    missing_markers = {"", "n/a", "na", "unknown"}
    missing = [unit in missing_markers for unit in units]
    if all(missing):
        return 1e-6
    if any(missing):
        raise ValueError(f"mixed specified and unspecified units for {spec.display_name}: {units}")
    return 1.0


def _clock_seconds(value: str) -> int:
    parts = re.split(r"[:.]", value.strip())
    if len(parts) < 3:
        raise ValueError(f"invalid CAP clock time: {value!r}")
    hour, minute, second = (int(part) for part in parts[:3])
    if not (0 <= hour < 24 and 0 <= minute < 60 and 0 <= second < 60):
        raise ValueError(f"invalid CAP clock time: {value!r}")
    return hour * 3600 + minute * 60 + second


def parse_cap_stage_events(path: str | Path, recording_start: datetime) -> list[StageEvent]:
    """Parse CAP RemLogic stages and align their wall-clock times to EDF onset."""

    time_index: int | None = None
    event_index: int | None = None
    duration_index: int | None = None
    rows: list[tuple[int, float, int]] = []

    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        columns = [column.strip() for column in line.split("\t")]
        normalized = [column.lower() for column in columns]
        if "time [hh:mm:ss]" in normalized and "event" in normalized:
            time_index = normalized.index("time [hh:mm:ss]")
            event_index = normalized.index("event")
            duration_index = next(
                (index for index, name in enumerate(normalized) if name.startswith("duration")),
                None,
            )
            continue
        if time_index is None or event_index is None or duration_index is None:
            continue
        required_index = max(time_index, event_index, duration_index)
        if len(columns) <= required_index:
            continue
        event_name = columns[event_index].upper()
        label = CAP_STAGE_LABELS.get(event_name)
        if label is None:
            continue
        duration = float(columns[duration_index].replace(",", "."))
        rows.append((_clock_seconds(columns[time_index]), duration, label))

    if not rows:
        raise ValueError(f"no scored CAP sleep stages found in {path}")

    start_clock = recording_start.hour * 3600 + recording_start.minute * 60 + recording_start.second
    first_clock = rows[0][0]
    onset = float(((first_clock - start_clock + 43200) % 86400) - 43200)
    previous_clock = first_clock
    events: list[StageEvent] = []
    for clock, duration, label in rows:
        if events:
            onset += float((clock - previous_clock) % 86400)
        events.append(StageEvent(onset, duration, label))
        previous_clock = clock
    return events
