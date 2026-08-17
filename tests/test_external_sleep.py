from __future__ import annotations

import numpy as np

from uniphysio_wm.external_sleep import (
    expand_sleep_edfx_annotations,
    map_isruc_labels,
    resolve_channel,
    resolve_external_channels,
    sleep_edfx_pair_key,
    sleep_edfx_subject,
    trimmed_sleep_interval,
)


def test_isruc_label_mapping_merges_to_five_stages() -> None:
    assert map_isruc_labels([0, 1, 2, 3, 5, 6]).tolist() == [0, 1, 2, 3, 4, -1]


def test_channel_resolution_is_separator_insensitive() -> None:
    assert resolve_channel(["C4_A1", "X1", "X2"], "C4-A1") == "C4_A1"
    assert resolve_channel(["EEG Fpz-Cz", "EMG submental"], None) is None


def test_isruc_channel_versions_and_reference_pair_are_supported() -> None:
    renamed = resolve_external_channels(["C4-M1", "24", "25"], "isruc")
    assert [renamed[name].display_name for name in ("EEG", "ECG", "EMG")] == ["C4-M1", "25", "24"]
    derived = resolve_external_channels(["C4", "A1", "X1", "X2"], "isruc")
    assert derived["EEG"].display_name == "C4-A1"


def test_sleep_edfx_pairing_keeps_nights_together_by_subject() -> None:
    assert sleep_edfx_pair_key("SC4001E0-PSG.edf") == "SC4001"
    assert sleep_edfx_pair_key("SC4001EC-Hypnogram.edf") == "SC4001"
    assert sleep_edfx_subject("SC4001E0-PSG.edf") == sleep_edfx_subject("SC4002E0-PSG.edf")


def test_sleep_edfx_annotations_merge_n3_and_ignore_unknown() -> None:
    labels = expand_sleep_edfx_annotations(
        [0, 60, 90],
        [60, 30, 30],
        ["Sleep stage W", "Sleep stage 4", "Sleep stage ?"],
        n_epochs=4,
    )
    assert labels.tolist() == [0, 0, 3, -1]


def test_sleep_edfx_trimming_retains_fixed_wake_margin() -> None:
    labels = np.asarray([0] * 4 + [2, 2] + [0] * 5)
    assert trimmed_sleep_interval(labels, wake_margin_epochs=2) == (2, 8)
