from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch


def _pearson(prediction: np.ndarray, target: np.ndarray) -> float:
    if len(prediction) < 2:
        return float("nan")
    x = prediction.astype(np.float64) - float(np.mean(prediction))
    y = target.astype(np.float64) - float(np.mean(target))
    denominator = float(np.sqrt(np.square(x).sum() * np.square(y).sum()))
    return float((x * y).sum() / denominator) if denominator > 1e-12 else float("nan")


def standardized_physiology_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    feature_names: Sequence[str],
    feature_groups: Dict[str, Sequence[str]],
    horizons: Sequence[int] | None = None,
) -> Dict[str, object]:
    predicted = prediction.detach().cpu().numpy()
    targets = target.detach().cpu().numpy()
    mask = valid.detach().cpu().numpy().astype(bool)
    if predicted.shape != targets.shape or mask.shape != targets.shape:
        raise ValueError("physiology prediction, target, and validity shapes must match")
    if predicted.shape[-1] != len(feature_names):
        raise ValueError("physiology feature count does not match names")
    per_feature = {}
    for feature_index, name in enumerate(feature_names):
        selected = mask[..., feature_index]
        errors = np.abs(
            predicted[..., feature_index][selected] - targets[..., feature_index][selected]
        )
        per_feature[name] = {
            "normalized_mae": float(errors.mean()),
            "pearson_r": _pearson(
                predicted[..., feature_index][selected], targets[..., feature_index][selected]
            ),
            "valid_targets": int(selected.sum()),
        }
    by_group = {}
    for group, names in feature_groups.items():
        by_group[group] = {
            "mean_normalized_mae": float(
                np.mean([per_feature[name]["normalized_mae"] for name in names])
            ),
            "mean_pearson_r": float(
                np.nanmean([per_feature[name]["pearson_r"] for name in names])
            ),
        }
    result: Dict[str, object] = {
        "all_features": {
            "mean_normalized_mae": float(
                np.mean([per_feature[name]["normalized_mae"] for name in feature_names])
            ),
            "mean_pearson_r": float(
                np.nanmean([per_feature[name]["pearson_r"] for name in feature_names])
            ),
        },
        "by_group": by_group,
        "per_feature": per_feature,
    }
    if horizons is not None:
        if predicted.ndim != 3 or predicted.shape[1] != len(horizons):
            raise ValueError("horizon metrics require [samples, horizons, features]")
        by_horizon = {}
        for horizon_index, horizon in enumerate(horizons):
            values = []
            correlations = []
            for feature_index in range(len(feature_names)):
                selected = mask[:, horizon_index, feature_index]
                values.append(
                    float(
                        np.abs(
                            predicted[selected, horizon_index, feature_index]
                            - targets[selected, horizon_index, feature_index]
                        ).mean()
                    )
                )
                correlations.append(
                    _pearson(
                        predicted[selected, horizon_index, feature_index],
                        targets[selected, horizon_index, feature_index],
                    )
                )
            by_horizon[str(horizon)] = {
                "mean_normalized_mae": float(np.mean(values)),
                "mean_pearson_r": float(np.nanmean(correlations)),
            }
        result["by_horizon"] = by_horizon
    return result
