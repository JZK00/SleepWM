import torch

from uniphysio_wm.metrics import classification_metrics, forecast_subgroup_metrics


def test_classification_metrics_include_paper_fields() -> None:
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0], [0.0, 2.0]])
    labels = torch.tensor([0, 1, 0])
    metrics = classification_metrics(logits, labels, num_classes=2)
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert -1.0 <= metrics["cohen_kappa"] <= 1.0
    assert len(metrics["per_class_f1"]) == 2
    assert metrics["samples"] == 3


def test_forecast_subgroups_use_current_label_at_each_horizon() -> None:
    labels = torch.tensor([[0, 1], [0, 1]])
    current = torch.tensor([0, 1])
    logits = torch.nn.functional.one_hot(labels, num_classes=2).float()
    groups = forecast_subgroup_metrics(logits, labels, current, horizons=(1, 2), num_classes=2)
    assert groups["stable"]["all_horizons"]["samples"] == 2
    assert groups["transition"]["all_horizons"]["samples"] == 2
    assert groups["stable"]["by_horizon"]["1"]["samples"] == 1
    assert groups["transition"]["by_horizon"]["2"]["samples"] == 1
    assert groups["stable"]["fraction"] == 0.5
