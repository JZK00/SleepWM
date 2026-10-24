"""Semantic checks for the custom-data training/evaluation workflow."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from sleepwm import runner
from sleepwm.features import FEATURE_NAMES


class TinyModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head = torch.nn.Linear(1, 17)

    def forward(self, signals, present):
        result = self.head(signals.mean((1, 2, 3)).reshape(-1, 1))[:, None, :]
        return {"stage_logits": result[:, :, :5], "future_physiology": result[:, :, 5:16],
                "transition_risk": torch.sigmoid(result[:, :, 16])}


class TrainingTests(unittest.TestCase):
    def test_parameters_change_and_checkpoint_roundtrip(self):
        runner.set_seed(7)
        model = TinyModel({})
        batch = {"signals": torch.randn(3, 2, 3, 8), "present": torch.ones(3, 2, 3, dtype=torch.bool),
                 "stage": torch.tensor([[0], [1], [2]]), "physiology": torch.ones(3, 1, 11),
                 "physiology_valid": torch.ones(3, 1, 11, dtype=torch.bool),
                 "transition": torch.tensor([[0.], [1.], [0.]])}
        before = model.head.weight.detach().clone()
        result = runner.train_epoch(model, [batch], torch.optim.Adam(model.parameters(), lr=.01), "cpu")
        self.assertEqual(result["windows"], 3)
        self.assertFalse(torch.equal(before, model.head.weight))
        stats = {"feature_names": list(FEATURE_NAMES), "signal_mean": [0.] * 3,
                 "signal_std": [1.] * 3, "feature_mean": [0.] * 11, "feature_std": [1.] * 11}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "statistics.json").write_text(json.dumps(stats), encoding="utf-8")
            _, fingerprint = runner.read_statistics(path / "statistics.json")
            runner.save_checkpoint(path / "best.pt", model, {}, stats, fingerprint, {"smoke_test": True})
            with patch.object(runner, "SleepWM", TinyModel):
                reloaded, _ = runner.load_checkpoint(path / "best.pt", path / "statistics.json", "cpu")
            expected, actual = model(batch["signals"], batch["present"]), reloaded(batch["signals"], batch["present"])
            for key in expected:
                self.assertTrue(torch.equal(expected[key], actual[key]), key)
            stats["signal_mean"][0] = 1.
            (path / "statistics.json").write_text(json.dumps(stats), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Statistics differ"):
                runner.load_checkpoint(path / "best.pt", path / "statistics.json", "cpu")

    def test_training_constructs_no_test_dataset(self):
        calls = []
        def fake_dataset(manifest, statistics, split, **options):
            self.assertNotEqual(split, "test")
            calls.append(split)
            return split
        with patch.object(runner, "SleepDataset", fake_dataset):
            datasets = runner.make_training_datasets("manifest.csv", "statistics.json",
                                                     {"data": {"history_epochs": 20, "future_horizons": [1, 2, 4]}}, 8)
        self.assertEqual(calls, ["train", "val"])
        self.assertEqual(datasets, ("train", "val"))

    def test_average_precision_ties_and_invalid_physiology(self):
        stage = np.array([0, 1])
        logits = np.eye(5)[stage]
        physiology = np.array([[1., np.nan], [3., np.nan]])
        prediction = np.array([[2., 100.], [3., -100.]])
        valid = np.array([[True, False], [True, False]])
        event = np.array([1., 0.])
        risk = np.array([.5, .5])
        result = runner.endpoint_metrics(stage, logits, physiology, prediction, valid, event, risk)
        reversed_result = runner.endpoint_metrics(stage[::-1], logits[::-1], physiology[::-1],
                                                 prediction[::-1], valid[::-1], event[::-1], risk[::-1])
        self.assertEqual(result["transition_average_precision"], .5)
        self.assertEqual(result["transition_average_precision"], reversed_result["transition_average_precision"])
        self.assertEqual(result["physiology_mae_standardized"], .5)
        no_positive = runner.endpoint_metrics(stage, logits, physiology, prediction, valid, event * 0, risk)
        self.assertIsNone(no_positive["transition_average_precision"])


if __name__ == "__main__":
    unittest.main()
