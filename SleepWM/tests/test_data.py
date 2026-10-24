import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from prepare_data import prepare
from sleepwm.data import SleepDataset, load_record, read_manifest, window_starts


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write_record(self, name, offset=0, n=28):
        t = np.arange(3840) / 128
        signal = 1e-5 * np.sin(2 * np.pi * 8 * t) + offset
        signals = np.broadcast_to(signal, (n, 3, 3840)).copy().astype(np.float32)
        present = np.ones((n, 3), dtype=bool)
        path = self.root / f"{name}.npz"
        np.savez(path, signals=signals, labels=np.arange(n, dtype=np.int64) % 5,
                 modality_present=present, epoch_start_seconds=np.arange(n) * 30.0)
        return path

    def manifest(self, rows, name="raw.csv"):
        path = self.root / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["record_id", "subject", "split", "npz_path"])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def row(self, split, offset=0):
        return dict(record_id=split, subject=f"demo:{split}", split=split,
                    npz_path=str(self.write_record(split, offset)))

    def test_continuity_and_unknown_labels_are_not_compressed(self):
        labels = np.zeros(40, dtype=np.int64)
        onsets = np.arange(40) * 30.0
        onsets[21:] += 30
        self.assertEqual(list(window_starts(labels, onsets, 20, (1, 2, 4), 1)), [])
        onsets = np.arange(40) * 30.0
        labels[21] = -1
        self.assertEqual(list(window_starts(labels, onsets, 20, (1, 2, 4), 1)), [])

    def test_targets_and_transition_include_intermediate_epochs(self):
        rows = [self.row("train"), self.row("val", 1e-4)]
        with np.load(rows[0]["npz_path"]) as raw_record:
            arrays = dict(raw_record)
        # Last history stage is 4; +30 is also 4, +60 changes to 1,
        # and +120 returns to 4. Cumulative transition stays positive.
        arrays["labels"][20:24] = [4, 1, 4, 4]
        np.savez(rows[0]["npz_path"], **arrays)
        raw = self.manifest(rows)
        prepared = prepare(raw, self.root / "prepared")
        data = SleepDataset(prepared / "manifest.csv", prepared / "statistics.json", "train")
        sample = data[0]
        self.assertEqual(sample["stage"].tolist(), [4, 1, 4])  # indices 20, 21, 23
        self.assertEqual(sample["transition"].tolist(), [0., 1., 1.])
        self.assertEqual(tuple(sample["signals"].shape), (20, 3, 3840))
        self.assertEqual(tuple(sample["physiology"].shape), (3, 11))
        self.assertEqual(data.records[0]["split"], "train")
        self.assertEqual(data.index[0], (0, 0))
        self.assertLess(abs(data.stats["signal_mean"][0]), 1e-8)  # val offset not fitted

    def test_split_isolation_and_subject_overlap(self):
        train = self.row("train")
        val = dict(record_id="val", subject="demo:val", split="val", npz_path=str(self.root / "not_there.npz"))
        path = self.manifest([train, val])
        prepared = prepare(path, self.root / "train_only", splits=("train",))
        # Keep an inaccessible val row in the prepared manifest: construction
        # must inspect its split metadata but never open its raw/cache files.
        with (prepared / "manifest.csv").open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["record_id", "subject", "split", "npz_path", "features_path"])
            writer.writerow({**val, "features_path": "unavailable_val_features.npz"})
        data = SleepDataset(prepared / "manifest.csv", prepared / "statistics.json", "train")
        self.assertGreater(len(data), 0)
        val["subject"] = train["subject"]
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            read_manifest(self.manifest([train, val], "overlap.csv"))

    def test_test_requires_frozen_statistics_and_never_refits(self):
        path = self.manifest([self.row("train"), self.row("val"), self.row("test", 0.5)])
        prepared = prepare(path, self.root / "prepared")
        with self.assertRaisesRegex(ValueError, "requires --statistics"):
            prepare(path, self.root / "bad_test", splits=("test",))
        test = prepare(path, self.root / "test", splits=("test",), statistics=prepared / "statistics.json")
        self.assertEqual(json.loads((test / "statistics.json").read_text()),
                         json.loads((prepared / "statistics.json").read_text()))
        with self.assertRaises(FileExistsError):
            prepare(path, prepared)

    def test_absent_nonfinite_allowed_available_rejected(self):
        path = self.write_record("missing")
        with np.load(path) as raw:
            arrays = dict(raw)
        arrays["signals"][:, 2] = np.nan
        arrays["modality_present"][:, 2] = False
        np.savez(path, **arrays)
        self.assertTrue((load_record(path)["signals"][:, 2] == 0).all())
        arrays["modality_present"][0, 2] = True
        np.savez(path, **arrays)
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            load_record(path)


if __name__ == "__main__":
    unittest.main()
