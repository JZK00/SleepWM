import csv
import json

import numpy as np
import pytest

from uniphysio_wm.data import (
    EpochDataset,
    PhysioFeatureSequenceDataset,
    RecordBatchSampler,
    SequenceDataset,
    TransitionBalancedRecordBatchSampler,
    read_manifest,
    validate_subject_splits,
)


def make_manifest(tmp_path):
    rows = []
    for split, subject in (("train", "s1"), ("val", "s2"), ("test", "s3")):
        path = tmp_path / f"{split}.npz"
        signals = np.random.default_rng(1).standard_normal((8, 3, 16)).astype(np.float32)
        labels = np.arange(8, dtype=np.int64) % 5
        np.savez(path, signals=signals, labels=labels)
        rows.append({"record_id": split, "subject": subject, "split": split, "npz_path": path.name})
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["record_id", "subject", "split", "npz_path"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_epoch_and_sequence_dataset_contract(tmp_path) -> None:
    manifest = make_manifest(tmp_path)
    epoch_dataset = EpochDataset(manifest, "train", modalities=("EEG", "EMG"))
    sample = epoch_dataset[0]
    assert sample["signals"].shape == (2, 16)
    assert sample["modality_present"].tolist() == [True, True]

    sequence_dataset = SequenceDataset(
        manifest,
        "train",
        history_epochs=3,
        future_horizons=(1, 2),
    )
    sequence = sequence_dataset[0]
    assert sequence["history_signals"].shape == (3, 3, 16)
    assert sequence["history_labels"].tolist() == [0, 1, 2]
    assert sequence["future_signals"].shape == (2, 3, 16)
    assert sequence["future_labels"].tolist() == [3, 4]

    strided = SequenceDataset(
        manifest,
        "train",
        history_epochs=3,
        future_horizons=(1, 2),
        sequence_stride=2,
    )
    assert len(strided) == 2
    assert int(strided.future_label_counts.sum()) == 4

    offset = SequenceDataset(
        manifest,
        "train",
        history_epochs=3,
        future_horizons=(1, 2),
        sequence_start_offset=2,
    )
    assert [start for _, start in offset.index] == [2, 3]


def test_physio_feature_sequence_dataset_standardizes_and_masks_targets(tmp_path) -> None:
    manifest = make_manifest(tmp_path)
    feature_rows = []
    feature_names = tuple(f"f{index}" for index in range(11))
    for split in ("train", "val", "test"):
        features = np.tile(np.arange(11, dtype=np.float32), (8, 1))
        features += np.arange(8, dtype=np.float32)[:, None]
        valid = np.ones_like(features, dtype=bool)
        valid[1, 6] = False
        features[1, 6] = np.nan
        valid[3, 6] = False
        features[3, 6] = np.nan
        feature_path = tmp_path / f"{split}_features.npz"
        np.savez(feature_path, features=features, valid=valid, feature_names=feature_names)
        feature_rows.append(
            {
                "record_id": split,
                "subject": f"s_{split}",
                "split": split,
                "feature_npz": str(feature_path),
            }
        )
    feature_manifest = tmp_path / "feature_manifest.csv"
    with feature_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0]))
        writer.writeheader()
        writer.writerows(feature_rows)
    statistics = tmp_path / "feature_statistics.json"
    statistics.write_text(
        json.dumps(
            {
                "feature_names": feature_names,
                "feature_groups": {
                    "EEG": feature_names[:5],
                    "ECG": feature_names[5:8],
                    "EMG": feature_names[8:],
                },
                "train_mean": [0.0] * 11,
                "train_std": [2.0] * 11,
            }
        ),
        encoding="utf-8",
    )
    dataset = PhysioFeatureSequenceDataset(
        manifest,
        "train",
        history_epochs=3,
        future_horizons=(1, 2),
        feature_manifest_path=feature_manifest,
        feature_statistics_path=statistics,
    )
    sample = dataset[0]
    assert sample["history_physiology"].shape == (3, 11)
    assert not bool(sample["history_physiology_valid"][1, 6])
    assert float(sample["history_physiology"][1, 6]) == 0.0
    np.testing.assert_allclose(sample["current_physiology"].numpy(), np.arange(11) / 2.0 + 1.0)
    assert not bool(sample["future_physiology_valid"][0, 6])
    assert float(sample["future_physiology"][0, 6]) == 0.0


def test_subject_leakage_is_rejected(tmp_path) -> None:
    manifest = make_manifest(tmp_path)
    rows = read_manifest(manifest)
    rows[1]["subject"] = rows[0]["subject"]
    with pytest.raises(ValueError, match="multiple splits"):
        validate_subject_splits(rows)


def test_train_statistics_are_applied_by_modality(tmp_path) -> None:
    manifest = make_manifest(tmp_path)
    stats = tmp_path / "normalization.json"
    stats.write_text(
        json.dumps(
            {
                "modalities": ["EEG", "ECG", "EMG"],
                "mean": [1.0, 2.0, 3.0],
                "std": [2.0, 4.0, 8.0],
            }
        ),
        encoding="utf-8",
    )
    raw_dataset = EpochDataset(manifest, "train", modalities=("EEG", "EMG"))
    normalized_dataset = EpochDataset(
        manifest,
        "train",
        modalities=("EEG", "EMG"),
        normalization_path=stats,
    )
    raw = raw_dataset[0]["signals"].numpy()
    normalized = normalized_dataset[0]["signals"].numpy()
    np.testing.assert_allclose(normalized[0], (raw[0] - 1.0) / 2.0)
    np.testing.assert_allclose(normalized[1], (raw[1] - 3.0) / 8.0)


def test_record_batch_sampler_keeps_each_batch_within_one_record(tmp_path) -> None:
    manifest = make_manifest(tmp_path)
    dataset = EpochDataset(manifest, "train")
    sampler = RecordBatchSampler(dataset, batch_size=3, seed=7)
    for batch in sampler:
        record_indices = {dataset.index[index][0] for index in batch}
        assert len(record_indices) == 1

    sequence_dataset = SequenceDataset(manifest, "train", history_epochs=2, future_horizons=(1,))
    sequence_sampler = RecordBatchSampler(sequence_dataset, batch_size=3, seed=7)
    for batch in sequence_sampler:
        record_indices = {sequence_dataset.index[index][0] for index in batch}
        assert len(record_indices) == 1


def test_label_fraction_is_stratified_and_deterministic(tmp_path) -> None:
    manifest = make_manifest(tmp_path)
    first = EpochDataset(manifest, "train", label_fraction=0.5, subset_seed=11)
    second = EpochDataset(manifest, "train", label_fraction=0.5, subset_seed=11)
    assert first.index == second.index
    assert len(first) == 5
    assert np.all(first.label_counts[np.flatnonzero(first.label_counts)] >= 1)


def test_transition_balanced_sampler_preserves_epoch_size_and_batch_ratio(tmp_path) -> None:
    manifest = make_manifest(tmp_path)
    train_path = tmp_path / "train.npz"
    with np.load(train_path, allow_pickle=False) as archive:
        signals = archive["signals"].copy()
    labels = np.asarray([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.int64)
    np.savez(train_path, signals=signals, labels=labels)
    dataset = SequenceDataset(manifest, "train", history_epochs=2, future_horizons=(1,))
    assert dataset.transition_window == [False, True, False, False, False, True]

    sampler = TransitionBalancedRecordBatchSampler(
        dataset,
        batch_size=4,
        seed=7,
        transition_probability=0.5,
    )
    batches = list(sampler)
    assert sum(len(batch) for batch in batches) == len(dataset)
    for batch in batches:
        transition_count = sum(dataset.transition_window[index] for index in batch)
        assert transition_count == round(len(batch) * 0.5)
        assert len({dataset.index[index][0] for index in batch}) == 1
