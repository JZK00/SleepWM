from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from uniphysio_wm.preprocessing import (
    cap_unit_scale,
    materialize_channel,
    parse_cap_stage_events,
    resolve_channel_spec,
)


def test_cap_parser_accepts_header_variants_and_crosses_midnight(tmp_path) -> None:
    annotation = tmp_path / "record.txt"
    annotation.write_text(
        "Sleep Stage\tPosition\tTime [hh:mm:ss]\tEvent\tDuration [s]\tLocation\n"
        "W\tUnknown\t23.59.45\tSLEEP-S0\t30\tEEG\n"
        "S3\tUnknown\t00.00.15\tSLEEP-S3\t30\tEEG\n"
        "S4\tUnknown\t00.00.45\tSLEEP-S4\t30\tEEG\n"
        "R\tUnknown\t00.01.15\tSLEEP-REM\t30\tEEG\n",
        encoding="utf-8",
    )

    events = parse_cap_stage_events(
        annotation,
        datetime(2026, 1, 1, 23, 59, 15, tzinfo=timezone.utc),
    )

    assert [event.onset_seconds for event in events] == [30.0, 60.0, 90.0, 120.0]
    assert [event.label for event in events] == [0, 3, 3, 4]


def test_channel_resolution_prefers_recorded_bipolar_channel() -> None:
    spec = resolve_channel_spec(
        ["C4", "A1", "C4-A1"],
        direct_candidates=["C4-A1"],
        derived_pairs=[("C4", "A1")],
    )

    assert spec is not None
    assert spec.sources == ("C4-A1",)


def test_channel_resolution_and_materialization_support_derived_pair() -> None:
    spec = resolve_channel_spec(
        ["ECG1", "ECG2"],
        direct_candidates=["ECG1-ECG2"],
        derived_pairs=[("ECG1", "ECG2")],
    )

    assert spec is not None
    signal = materialize_channel(
        {
            "ECG1": np.asarray([2.0, 5.0], dtype=np.float32),
            "ECG2": np.asarray([1.0, 3.0], dtype=np.float32),
        },
        spec,
    )
    np.testing.assert_array_equal(signal, np.asarray([1.0, 2.0], dtype=np.float32))


def test_cap_unspecified_units_are_interpreted_as_microvolts() -> None:
    spec = resolve_channel_spec(
        ["CHIN1", "CHIN2"],
        direct_candidates=[],
        derived_pairs=[("CHIN1", "CHIN2")],
    )

    assert spec is not None
    assert cap_unit_scale(spec, {"CHIN1": "n/a", "CHIN2": "n/a"}) == 1e-6
    assert cap_unit_scale(spec, {"CHIN1": "µV", "CHIN2": "µV"}) == 1.0
