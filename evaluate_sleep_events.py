from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from uniphysio_wm.engine import data_loader, load_checkpoint, resolve_device, sequence_dataset, write_json  # noqa: E402
from uniphysio_wm.mainline import build_mainline_model  # noqa: E402


EVENTS = ("transition", "sleep_onset", "wake_onset", "rem_onset", "n3_onset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate sleep-event time-to-event and early warning.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--dataset-id", choices=("isruc", "sleep_edfx"), required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=(1, 2, 4, 10, 14))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-batches", type=int, default=0)
    return parser.parse_args()


def event_at(event: str, previous: int, current: int) -> bool:
    if previous < 0 or current < 0:
        return False
    if event == "transition":
        return previous != current
    if event == "sleep_onset":
        return previous == 0 and current != 0
    if event == "wake_onset":
        return previous != 0 and current == 0
    if event == "rem_onset":
        return previous != 4 and current == 4
    if event == "n3_onset":
        return previous != 3 and current == 3
    raise ValueError(f"unknown event: {event}")


def at_risk(event: str, current: int) -> bool:
    if current < 0:
        return False
    if event == "transition":
        return True
    if event == "sleep_onset":
        return current == 0
    if event == "wake_onset":
        return current != 0
    if event == "rem_onset":
        return current != 4
    if event == "n3_onset":
        return current != 3
    raise ValueError(f"unknown event: {event}")


def next_event_offset(labels: np.ndarray, current_index: int, event: str, max_horizon: int) -> int:
    end = min(len(labels) - 1, current_index + max_horizon)
    for index in range(current_index + 1, end + 1):
        if event_at(event, int(labels[index - 1]), int(labels[index])):
            return index - current_index
    return max_horizon + 1


def event_risk_scores(current_probabilities: torch.Tensor, future_probabilities: torch.Tensor) -> dict[str, torch.Tensor]:
    overlap = (current_probabilities.unsqueeze(1) * future_probabilities).sum(dim=-1)
    raw = {
        "transition": 1.0 - overlap,
        "sleep_onset": current_probabilities[:, 0:1] * (1.0 - future_probabilities[:, :, 0]),
        "wake_onset": (1.0 - current_probabilities[:, 0:1]) * future_probabilities[:, :, 0],
        "rem_onset": (1.0 - current_probabilities[:, 4:5]) * future_probabilities[:, :, 4],
        "n3_onset": (1.0 - current_probabilities[:, 3:4]) * future_probabilities[:, :, 3],
    }
    return {name: torch.cummax(value.clamp(0.0, 1.0), dim=1).values for name, value in raw.items()}


def select_f1_threshold(target: np.ndarray, score: np.ndarray) -> float:
    if target.size == 0 or np.unique(target).size < 2:
        return 0.5
    thresholds = np.unique(score)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        prediction = score >= threshold
        tp = int(np.sum(prediction & target))
        fp = int(np.sum(prediction & ~target))
        fn = int(np.sum(~prediction & target))
        f1 = 2.0 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def roc_auc(target: np.ndarray, score: np.ndarray) -> float | None:
    positive = int(target.sum())
    negative = int(target.size - positive)
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    ranks = np.empty(target.size, dtype=np.float64)
    start = 0
    while start < target.size:
        end = start + 1
        while end < target.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    rank_sum = float(ranks[target].sum())
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def average_precision(target: np.ndarray, score: np.ndarray) -> float | None:
    positive = int(target.sum())
    if positive == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    sorted_target = target[order].astype(np.float64)
    precision = np.cumsum(sorted_target) / np.arange(1, target.size + 1)
    return float(precision[sorted_target.astype(bool)].sum() / positive)


def binary_metrics(target: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = score >= threshold
    tp = int(np.sum(prediction & target))
    fp = int(np.sum(prediction & ~target))
    fn = int(np.sum(~prediction & target))
    tn = int(np.sum(~prediction & ~target))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return {
        "auroc": roc_auc(target, score),
        "auprc": average_precision(target, score),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "samples": int(target.size),
        "event_fraction": float(target.mean()) if target.size else 0.0,
    }


def load_label_map(dataset) -> dict[str, np.ndarray]:
    result = {}
    for record in dataset.records:
        with np.load(record["path"], allow_pickle=False) as archive:
            result[str(record["record_id"])] = archive["labels"].astype(np.int64, copy=True)
    return result


@torch.inference_mode()
def collect_predictions(model, dataset, loader, device, horizons: Sequence[int], history_epochs: int, max_batches: int) -> dict[str, Any]:
    label_map = load_label_map(dataset)
    record_ids: list[str] = []
    origins: list[int] = []
    scores = {event: [] for event in EVENTS}
    model.eval()
    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        signals = batch["history_signals"].to(device, non_blocking=True)
        present = batch["history_present"].to(device, non_blocking=True)
        output = model.rollout_context_horizons(signals, present, horizons)
        risks = event_risk_scores(
            output["current_stage_logits"].softmax(dim=-1),
            output["stage_logits"].softmax(dim=-1),
        )
        for event in EVENTS:
            scores[event].append(risks[event].cpu())
        record_ids.extend(str(value) for value in batch["record_id"])
        origins.extend((batch["start_epoch"] + history_epochs - 1).tolist())

    score_arrays = {event: torch.cat(values).numpy() for event, values in scores.items()}
    max_horizon = max(horizons)
    current_labels = np.asarray(
        [label_map[record_id][origin] for record_id, origin in zip(record_ids, origins)],
        dtype=np.int64,
    )
    tte = {
        event: np.asarray(
            [next_event_offset(label_map[record_id], origin, event, max_horizon) for record_id, origin in zip(record_ids, origins)],
            dtype=np.int64,
        )
        for event in EVENTS
    }
    risk_masks = {
        event: np.asarray([at_risk(event, int(label)) for label in current_labels], dtype=bool)
        for event in EVENTS
    }
    return {
        "record_ids": np.asarray(record_ids),
        "origins": np.asarray(origins, dtype=np.int64),
        "scores": score_arrays,
        "tte": tte,
        "at_risk": risk_masks,
    }


def evaluate_event_bundle(bundle: dict[str, Any], horizons: Sequence[int], thresholds: dict[str, float], epoch_seconds: int) -> dict[str, Any]:
    results = {}
    max_horizon = max(horizons)
    for event in EVENTS:
        mask = bundle["at_risk"][event]
        scores = bundle["scores"][event][mask]
        tte = bundle["tte"][event][mask]
        record_ids = bundle["record_ids"][mask]
        origins = bundle["origins"][mask]
        threshold = thresholds[event]
        by_horizon = {}
        for index, horizon in enumerate(horizons):
            target = tte <= horizon
            by_horizon[str(horizon * epoch_seconds)] = binary_metrics(target, scores[:, index], threshold)

        target_max = tte <= max_horizon
        predicted_indices = np.argmax(scores >= threshold, axis=1)
        detected = (scores >= threshold).any(axis=1)
        predicted_tte = np.asarray(horizons, dtype=np.int64)[predicted_indices]
        true_detected = target_max & detected
        mae = (
            float(np.mean(np.abs(predicted_tte[true_detected] - tte[true_detected])) * epoch_seconds)
            if true_detected.any()
            else None
        )
        event_keys = {
            (str(record_id), int(origin + offset))
            for record_id, origin, offset, positive in zip(record_ids, origins, tte, target_max)
            if positive
        }
        lead_by_event: dict[tuple[str, int], int] = {}
        for record_id, origin, offset, positive, alert in zip(record_ids, origins, tte, target_max, detected):
            if positive and alert:
                key = (str(record_id), int(origin + offset))
                lead_by_event[key] = max(lead_by_event.get(key, 0), int(offset) * epoch_seconds)
        lead_values = list(lead_by_event.values())
        results[event] = {
            "threshold": float(threshold),
            "by_horizon_seconds": by_horizon,
            "max_horizon": binary_metrics(target_max, scores[:, -1], threshold),
            "tte_mae_seconds_detected_events": mae,
            "event_coverage": float(len(lead_by_event) / max(len(event_keys), 1)),
            "covered_events": len(lead_by_event),
            "eligible_events": len(event_keys),
            "median_lead_seconds": float(np.median(lead_values)) if lead_values else None,
            "mean_lead_seconds": float(np.mean(lead_values)) if lead_values else None,
        }
    return results


def main() -> int:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError("checkpoint lacks resolved config")
    config = copy.deepcopy(checkpoint_config)
    config["data"]["manifest_path"] = str(Path(args.manifest).resolve())
    config["data"]["normalization_path"] = str(Path(args.normalization).resolve())
    config["data"]["future_horizons"] = [int(value) for value in args.horizons]
    config["data"]["sequence_stride"] = 1
    config["train"]["batch_size"] = int(args.batch_size)
    config["train"]["num_workers"] = int(args.workers)
    config["train"]["device"] = str(args.device)
    device = resolve_device(args.device)
    model = build_mainline_model(config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    horizons = tuple(int(value) for value in args.horizons)
    history_epochs = int(config["data"]["history_epochs"])

    validation = sequence_dataset(config, "val")
    validation_bundle = collect_predictions(
        model,
        validation,
        data_loader(validation, config, shuffle=False),
        device,
        horizons,
        history_epochs,
        args.max_batches,
    )
    thresholds = {}
    for event in EVENTS:
        mask = validation_bundle["at_risk"][event]
        target = validation_bundle["tte"][event][mask] <= max(horizons)
        thresholds[event] = select_f1_threshold(target, validation_bundle["scores"][event][mask, -1])

    test = sequence_dataset(config, "test")
    test_bundle = collect_predictions(
        model,
        test,
        data_loader(test, config, shuffle=False),
        device,
        horizons,
        history_epochs,
        args.max_batches,
    )
    epoch_seconds = int(config["data"]["epoch_seconds"])
    payload = {
        "protocol": "stage8_sleep_event_v1",
        "dataset_id": args.dataset_id,
        "event_definitions": list(EVENTS),
        "horizons_seconds": [horizon * epoch_seconds for horizon in horizons],
        "threshold_source": "external_validation_only",
        "thresholds": thresholds,
        "validation_sequences": len(validation_bundle["origins"]),
        "test_sequences": len(test_bundle["origins"]),
        "test_results": evaluate_event_bundle(test_bundle, horizons, thresholds, epoch_seconds),
        "hmc_cap_test_accessed": False,
        "driving_risk_claim": False,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
