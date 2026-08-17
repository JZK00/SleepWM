from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch


def confusion_matrix(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> np.ndarray:
    prediction = logits.argmax(dim=-1).detach().cpu().numpy().reshape(-1)
    truth = labels.detach().cpu().numpy().reshape(-1)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, predicted in zip(truth.tolist(), prediction.tolist()):
        if 0 <= target < num_classes and 0 <= predicted < num_classes:
            matrix[target, predicted] += 1
    return matrix


def metrics_from_confusion(matrix: np.ndarray) -> Dict[str, object]:
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / max(total, 1))
    recalls = []
    per_class_f1 = []
    for class_index in range(matrix.shape[0]):
        true_positive = float(matrix[class_index, class_index])
        false_positive = float(matrix[:, class_index].sum() - true_positive)
        false_negative = float(matrix[class_index, :].sum() - true_positive)
        support = float(matrix[class_index, :].sum())
        if support > 0:
            recalls.append(true_positive / support)
        precision = true_positive / max(true_positive + false_positive, 1.0)
        recall = true_positive / max(true_positive + false_negative, 1.0)
        score = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        per_class_f1.append(float(score))

    row_marginal = matrix.sum(axis=1).astype(np.float64)
    column_marginal = matrix.sum(axis=0).astype(np.float64)
    expected = float((row_marginal * column_marginal).sum() / max(total * total, 1))
    kappa = (accuracy - expected) / max(1.0 - expected, 1e-12)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(per_class_f1)) if per_class_f1 else 0.0,
        "cohen_kappa": float(kappa),
        "per_class_f1": per_class_f1,
        "confusion_matrix": matrix.tolist(),
        "samples": total,
    }


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int = 5) -> Dict[str, object]:
    return metrics_from_confusion(confusion_matrix(logits, labels, num_classes))


def forecast_subgroup_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    current_labels: torch.Tensor,
    horizons: Sequence[int],
    num_classes: int = 5,
) -> Dict[str, object]:
    transition = labels != current_labels.reshape(-1, 1)
    groups = {}
    for name, mask in (("stable", ~transition), ("transition", transition)):
        groups[name] = {
            "all_horizons": classification_metrics(logits[mask], labels[mask], num_classes),
            "by_horizon": {
                str(horizon): classification_metrics(
                    logits[:, index][mask[:, index]],
                    labels[:, index][mask[:, index]],
                    num_classes,
                )
                for index, horizon in enumerate(horizons)
            },
            "fraction": float(mask.float().mean()),
        }
    return groups


def factorized_transition_metrics(
    change_logits: torch.Tensor,
    destination_logits: torch.Tensor,
    labels: torch.Tensor,
    current_labels: torch.Tensor,
    horizons: Sequence[int],
    num_classes: int = 5,
) -> Dict[str, object]:
    transition = labels != current_labels.reshape(-1, 1)
    binary_logits = torch.stack((torch.zeros_like(change_logits), change_logits), dim=-1)
    return {
        "change_detection": {
            "all_horizons": classification_metrics(binary_logits, transition.long(), 2),
            "by_horizon": {
                str(horizon): classification_metrics(
                    binary_logits[:, index], transition[:, index].long(), 2
                )
                for index, horizon in enumerate(horizons)
            },
        },
        "destination_on_transition": {
            "all_horizons": classification_metrics(
                destination_logits[transition], labels[transition], num_classes
            ),
            "by_horizon": {
                str(horizon): classification_metrics(
                    destination_logits[:, index][transition[:, index]],
                    labels[:, index][transition[:, index]],
                    num_classes,
                )
                for index, horizon in enumerate(horizons)
            },
        },
    }


def reconstruction_metrics(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    selected_prediction = prediction[mask]
    selected_target = target[mask]
    mse = float((selected_prediction - selected_target).pow(2).mean().detach().cpu())
    pred_flat = selected_prediction.detach().float().reshape(-1)
    target_flat = selected_target.detach().float().reshape(-1)
    pred_centered = pred_flat - pred_flat.mean()
    target_centered = target_flat - target_flat.mean()
    denominator = pred_centered.norm() * target_centered.norm()
    correlation = float((pred_centered @ target_centered / denominator.clamp_min(1e-12)).cpu())
    return {"mse": mse, "correlation": correlation}
