from __future__ import annotations

import torch

from evaluate_external_zero_shot import calibration_metrics, predicted_persistence_metrics


def test_calibration_is_near_perfect_for_confident_correct_predictions() -> None:
    logits = torch.tensor([[12.0, -12.0], [-12.0, 12.0]])
    metrics = calibration_metrics(logits, torch.tensor([0, 1]))
    assert metrics["ece_15"] < 1e-6
    assert metrics["brier"] < 1e-6
    assert metrics["nll"] < 1e-6


def test_predicted_persistence_repeats_current_logits() -> None:
    current = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
    future = torch.tensor([[0, 0], [1, 0]])
    metrics = predicted_persistence_metrics(current, future, horizons=(1, 2), num_classes=2)
    assert metrics["by_horizon"]["1"]["accuracy"] == 1.0
    assert metrics["by_horizon"]["2"]["accuracy"] == 0.5
