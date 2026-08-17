from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
import torch.nn.functional as F

from uniphysio_wm.probabilistic_waveform import (
    _band_log_power,
    _eeg_targets,
    _emg_targets,
    _qrs_events,
    _rr_centers,
    _rr_targets,
    _soft_event_targets,
)


def _finite_mean(values: torch.Tensor) -> float:
    selected = values[torch.isfinite(values)]
    return float(selected.mean()) if selected.numel() else float("nan")


def _correlation_rows(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    x = first.double() - first.double().mean(dim=-1, keepdim=True)
    y = second.double() - second.double().mean(dim=-1, keepdim=True)
    denominator = x.square().sum(dim=-1).sqrt() * y.square().sum(dim=-1).sqrt()
    result = (x * y).sum(dim=-1) / denominator.clamp_min(1e-12)
    return torch.where(denominator > 1e-12, result, torch.nan)


def _binary_f1(prediction: torch.Tensor, target: torch.Tensor) -> float:
    predicted = prediction.bool()
    targets = target.bool()
    true_positive = (predicted & targets).sum().double()
    precision = true_positive / predicted.sum().double().clamp_min(1.0)
    recall = true_positive / targets.sum().double().clamp_min(1.0)
    return float(2.0 * precision * recall / (precision + recall).clamp_min(1e-12))


def _event_scores(
    predicted_events: torch.Tensor,
    target_events: torch.Tensor,
    sample_rate: int,
) -> Dict[str, float]:
    tolerance = max(1, round(0.08 * sample_rate))
    expanded_target = F.max_pool1d(
        target_events.float().unsqueeze(1),
        kernel_size=2 * tolerance + 1,
        stride=1,
        padding=tolerance,
    ).squeeze(1).bool()
    expanded_prediction = F.max_pool1d(
        predicted_events.float().unsqueeze(1),
        kernel_size=2 * tolerance + 1,
        stride=1,
        padding=tolerance,
    ).squeeze(1).bool()
    precision_tp = (predicted_events & expanded_target).sum().double()
    recall_tp = (target_events & expanded_prediction).sum().double()
    precision = precision_tp / predicted_events.sum().double().clamp_min(1.0)
    recall = recall_tp / target_events.sum().double().clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predicted_count_per_record": float(predicted_events.sum(-1).float().mean()),
        "target_count_per_record": float(target_events.sum(-1).float().mean()),
    }


def _ridge_fit(features: torch.Tensor, target: torch.Tensor, alpha: float) -> dict:
    x = features.double()
    y = target.double()
    mean = x.mean(dim=0)
    scale = x.std(dim=0).clamp_min(1e-6)
    standardized = (x - mean) / scale
    design = torch.cat((torch.ones(len(x), 1, dtype=x.dtype), standardized), dim=1)
    penalty = torch.eye(design.shape[1], dtype=x.dtype) * float(alpha)
    penalty[0, 0] = 0.0
    beta = torch.linalg.solve(design.T @ design + penalty, design.T @ y)
    return {
        "feature_mean": mean.float(),
        "feature_scale": scale.float(),
        "beta": beta.float(),
    }


def _ridge_predict(state: dict, features: torch.Tensor) -> torch.Tensor:
    x = (features.float() - state["feature_mean"]) / state["feature_scale"]
    return state["beta"][0] + x @ state["beta"][1:]


def _logistic_fit(features: torch.Tensor, target: torch.Tensor) -> dict:
    maximum_rows = 200_000
    if len(features) > maximum_rows:
        indices = torch.linspace(0, len(features) - 1, maximum_rows).long()
        features = features[indices]
        target = target[indices]
    x = features.float()
    y = target.float()
    mean = x.mean(dim=0)
    scale = x.std(dim=0).clamp_min(1e-6)
    design = torch.cat((torch.ones(len(x), 1), (x - mean) / scale), dim=1)
    weights = torch.zeros(design.shape[1], requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weights], lr=1.0, max_iter=40, tolerance_grad=1e-8, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(design @ weights, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    probability = torch.sigmoid(design @ weights.detach())
    thresholds = torch.linspace(0.05, 0.95, 91)
    scores = torch.tensor([_binary_f1(probability >= value, y.bool()) for value in thresholds])
    threshold = thresholds[int(scores.argmax())]
    return {
        "feature_mean": mean,
        "feature_scale": scale,
        "weights": weights.detach(),
        "decision_threshold": threshold,
        "fit_prevalence": y.mean(),
        "fit_f1": scores.max(),
    }


def _logistic_predict(state: dict, features: torch.Tensor) -> torch.Tensor:
    x = (features.float() - state["feature_mean"]) / state["feature_scale"]
    design = torch.cat((torch.ones(len(x), 1), x), dim=1)
    return torch.sigmoid(design @ state["weights"])


def _residual_quantiles(residual: torch.Tensor) -> torch.Tensor:
    values = residual.reshape(-1, residual.shape[-1]).float()
    levels = torch.linspace(0.01, 0.99, 99)
    return torch.quantile(values, levels, dim=0)


def _interval_coverage(
    mean: torch.Tensor,
    target: torch.Tensor,
    quantiles: torch.Tensor,
    low_index: int,
    high_index: int,
) -> float:
    lower = mean + quantiles[low_index]
    upper = mean + quantiles[high_index]
    return float(((target >= lower) & (target <= upper)).float().mean())


def _sample_coverage(
    mean: torch.Tensor,
    target: torch.Tensor,
    quantiles: torch.Tensor,
    seed: int,
    sample_count: int,
    lower_probability: float,
    upper_probability: float,
) -> float:
    generator = torch.Generator().manual_seed(int(seed))
    strata = torch.arange(int(sample_count), dtype=torch.float32).unsqueeze(-1)
    jitter = torch.rand(
        int(sample_count), quantiles.shape[1], generator=generator
    )
    levels = (strata + jitter) / float(sample_count)
    indices = (levels * quantiles.shape[0]).long().clamp_max(quantiles.shape[0] - 1)
    feature_indices = torch.arange(quantiles.shape[1]).unsqueeze(0)
    draws = quantiles[indices, feature_indices]
    lower = torch.quantile(draws, lower_probability, dim=0)
    upper = torch.quantile(draws, upper_probability, dim=0)
    return float(((target >= mean + lower) & (target <= mean + upper)).float().mean())


def _rr_mean_from_head(probability: dict, selected: torch.Tensor, patch_count: int) -> torch.Tensor:
    rr_logits = probability.get("rr_logits")
    if rr_logits is None:
        return probability["rr_mean_seconds"][selected, :patch_count].mean(dim=1)
    logits = rr_logits[selected, :patch_count].mean(dim=1)
    centers = _rr_centers(logits)
    return (logits.softmax(dim=-1) * centers).sum(dim=-1)


def _rr_features(
    generated: torch.Tensor,
    recent: torch.Tensor,
    probability: dict,
    selected: torch.Tensor,
    sample_rate: int,
    samples: int,
    patch_count: int,
) -> tuple[torch.Tensor, dict]:
    generated_events = _qrs_events(generated[selected, :samples], sample_rate)
    recent_events = _qrs_events(recent[selected], sample_rate)
    generated_rr, generated_valid = _rr_targets(generated_events, sample_rate)
    recent_rr, recent_valid = _rr_targets(recent_events, sample_rate)
    head_rr = _rr_mean_from_head(probability, selected, patch_count)
    generated_filled = torch.where(generated_valid, generated_rr, head_rr)
    recent_filled = torch.where(recent_valid, recent_rr, head_rr)
    features = torch.stack(
        (
            recent_filled,
            generated_filled,
            head_rr,
            recent_valid.float(),
            generated_valid.float(),
        ),
        dim=-1,
    )
    return features, {
        "generated_events": generated_events,
        "generated_rr": generated_rr,
        "generated_rr_valid": generated_valid,
        "recent_events": recent_events,
        "recent_rr": recent_rr,
        "recent_rr_valid": recent_valid,
        "head_rr": head_rr,
    }


def _gaussian_event_basis(events: torch.Tensor, sample_rate: int, sigma_seconds: float) -> torch.Tensor:
    sigma = max(1.0, float(sigma_seconds) * sample_rate)
    radius = max(1, math.ceil(4.0 * sigma))
    positions = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (positions / sigma).square()).view(1, 1, -1)
    return F.conv1d(events.float().unsqueeze(1), kernel, padding=radius).squeeze(1).clamp_max(1.0)


def _fit_event_probability(
    generated_events: torch.Tensor,
    target_events: torch.Tensor,
    sample_rate: int,
) -> dict:
    maximum_rows = min(len(generated_events), 4096)
    indices = torch.linspace(0, len(generated_events) - 1, maximum_rows).long()
    generated_events = generated_events[indices]
    target = _soft_event_targets(target_events[indices], sample_rate).float()
    sigma_seconds = 0.04
    basis = _gaussian_event_basis(generated_events, sample_rate, sigma_seconds)
    design = torch.stack((torch.ones_like(basis), basis), dim=-1).reshape(-1, 2).double()
    response = target.reshape(-1).double()
    beta = torch.linalg.lstsq(design, response).solution.float()
    floor = beta[0].clamp(1e-4, 0.25)
    gain = beta[1].clamp(0.0, 1.0 - floor)
    return {"sigma_seconds": sigma_seconds, "floor": floor, "gain": gain}


def _event_probability(state: dict, events: torch.Tensor, sample_rate: int) -> torch.Tensor:
    basis = _gaussian_event_basis(events, sample_rate, float(state["sigma_seconds"]))
    return (state["floor"] + state["gain"] * basis).clamp(1e-4, 1.0 - 1e-4)


def _event_distances(
    predicted_events: torch.Tensor, target_events: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    target_to_prediction = []
    symmetric = []
    for predicted_row, target_row in zip(predicted_events, target_events):
        predicted = predicted_row.nonzero(as_tuple=False).flatten().float()
        target = target_row.nonzero(as_tuple=False).flatten().float()
        if predicted.numel() and target.numel():
            distances = (predicted[:, None] - target[None, :]).abs()
            target_to_prediction.append(distances.min(dim=0).values)
            symmetric.append(torch.cat((distances.min(dim=0).values, distances.min(dim=1).values)))
    empty = predicted_events.new_empty(0, dtype=torch.float32)
    return (
        torch.cat(target_to_prediction) if target_to_prediction else empty,
        torch.cat(symmetric) if symmetric else empty,
    )


def _emg_regression_features(
    generated_descriptor: torch.Tensor,
    head_descriptor: torch.Tensor,
    recent_descriptor: torch.Tensor,
) -> torch.Tensor:
    patch_count = generated_descriptor.shape[1]
    position = torch.linspace(0.0, 1.0, patch_count).expand(len(generated_descriptor), -1)
    recent_mean = recent_descriptor.mean(dim=1, keepdim=True).expand(-1, patch_count)
    return torch.stack((generated_descriptor, head_descriptor, recent_mean, position), dim=-1)


def _burst_features(
    envelope: torch.Tensor,
    rms: torch.Tensor,
    old_logits: torch.Tensor,
) -> torch.Tensor:
    envelope_z = (envelope - envelope.mean(dim=1, keepdim=True)) / envelope.std(
        dim=1, keepdim=True
    ).clamp_min(1e-6)
    rms_z = (rms - rms.mean(dim=1, keepdim=True)) / rms.std(
        dim=1, keepdim=True
    ).clamp_min(1e-6)
    return torch.stack((envelope_z, rms_z, old_logits), dim=-1)


def fit_renderer_probability_calibration(
    generated: torch.Tensor,
    probabilities: Dict[str, Dict[str, torch.Tensor]],
    recent: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    modalities: Sequence[str],
    sample_rate: int,
    horizons_seconds: Sequence[int],
    patch_samples: int,
    ridge_alpha: float = 1e-3,
) -> dict:
    modality_index = {name: index for index, name in enumerate(modalities)}
    state = {"by_horizon_seconds": {}, "fit_split": "train", "version": 1}
    for seconds in horizons_seconds:
        samples = int(seconds) * int(sample_rate)
        patch_count = samples // int(patch_samples)
        horizon = {}

        selected = valid[:, modality_index["EEG"]].bool()
        eeg_target = _eeg_targets(
            target[selected, modality_index["EEG"], :samples], patch_samples, sample_rate
        )
        eeg_mean = probabilities["EEG"]["spectral_mean"][selected, :patch_count]
        horizon["EEG"] = {"residual_quantiles": _residual_quantiles(eeg_target - eeg_mean)}

        selected = valid[:, modality_index["ECG"]].bool()
        target_events = _qrs_events(target[selected, modality_index["ECG"], :samples], sample_rate)
        features, rr_info = _rr_features(
            generated[:, modality_index["ECG"]],
            recent[:, modality_index["ECG"]],
            probabilities["ECG"],
            selected,
            sample_rate,
            samples,
            patch_count,
        )
        target_rr, target_rr_valid = _rr_targets(target_events, sample_rate)
        rr_model = _ridge_fit(features[target_rr_valid], target_rr[target_rr_valid], ridge_alpha)
        fitted_rr = _ridge_predict(rr_model, features).clamp(0.25, 2.0)
        distances, _ = _event_distances(rr_info["generated_events"], target_events)
        horizon["ECG"] = {
            "rr_model": rr_model,
            "rr_residual_quantiles": _residual_quantiles(
                (target_rr[target_rr_valid] - fitted_rr[target_rr_valid]).unsqueeze(-1)
            ),
            "event_probability": _fit_event_probability(
                rr_info["generated_events"], target_events, sample_rate
            ),
            "timing_quantiles_samples": torch.quantile(
                distances, torch.tensor([0.80, 0.95])
            ) if distances.numel() else torch.tensor([float("nan"), float("nan")]),
        }

        selected = valid[:, modality_index["EMG"]].bool()
        target_emg = target[selected, modality_index["EMG"], :samples]
        generated_emg = generated[selected, modality_index["EMG"], :samples]
        recent_emg = recent[selected, modality_index["EMG"]]
        target_envelope, target_rms, target_burst = _emg_targets(
            target_emg, patch_samples, sample_rate
        )
        generated_envelope, generated_rms, _ = _emg_targets(
            generated_emg, patch_samples, sample_rate
        )
        recent_envelope, recent_rms, _ = _emg_targets(
            recent_emg, patch_samples, sample_rate
        )
        old_envelope = probabilities["EMG"]["envelope_mean"][selected, :patch_count]
        old_rms = probabilities["EMG"]["rms_mean"][selected, :patch_count]
        envelope_features = _emg_regression_features(
            generated_envelope, old_envelope, recent_envelope
        )
        rms_features = _emg_regression_features(generated_rms, old_rms, recent_rms)
        envelope_model = _ridge_fit(
            envelope_features.reshape(-1, envelope_features.shape[-1]),
            target_envelope.reshape(-1),
            ridge_alpha,
        )
        rms_model = _ridge_fit(
            rms_features.reshape(-1, rms_features.shape[-1]),
            target_rms.reshape(-1),
            ridge_alpha,
        )
        calibrated_envelope = _ridge_predict(
            envelope_model, envelope_features.reshape(-1, envelope_features.shape[-1])
        ).reshape_as(target_envelope).clamp_min(0.0)
        calibrated_rms = _ridge_predict(
            rms_model, rms_features.reshape(-1, rms_features.shape[-1])
        ).reshape_as(target_rms).clamp_min(0.0)
        burst_features = _burst_features(
            calibrated_envelope,
            calibrated_rms,
            probabilities["EMG"]["burst_logits"][selected, :patch_count],
        )
        burst_model = _logistic_fit(
            burst_features.reshape(-1, burst_features.shape[-1]), target_burst.reshape(-1)
        )
        horizon["EMG"] = {
            "envelope_model": envelope_model,
            "rms_model": rms_model,
            "burst_model": burst_model,
            "envelope_residual_quantiles": _residual_quantiles(
                (target_envelope - calibrated_envelope).unsqueeze(-1)
            ),
            "rms_residual_quantiles": _residual_quantiles(
                (target_rms - calibrated_rms).unsqueeze(-1)
            ),
        }
        state["by_horizon_seconds"][str(seconds)] = horizon
    return state


def _probability_metrics_for_horizon(
    state: dict,
    generated: torch.Tensor,
    probabilities: Dict[str, Dict[str, torch.Tensor]],
    recent: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    modality_index: dict,
    sample_rate: int,
    seconds: int,
    patch_samples: int,
    seed: int,
    sample_count: int,
) -> dict:
    samples = seconds * sample_rate
    patch_count = samples // patch_samples
    metrics = {}

    selected = valid[:, modality_index["EEG"]].bool()
    eeg_target = _eeg_targets(target[selected, modality_index["EEG"], :samples], patch_samples, sample_rate)
    eeg_recent = _eeg_targets(recent[selected, modality_index["EEG"], :samples], patch_samples, sample_rate)
    eeg_mean = probabilities["EEG"]["spectral_mean"][selected, :patch_count]
    eeg_quantiles = state["EEG"]["residual_quantiles"]
    metrics["EEG"] = {
        "spectral_mean_mae": float((eeg_mean - eeg_target).abs().mean()),
        "recent_spectral_mae": float((eeg_recent - eeg_target).abs().mean()),
        "coverage_80": _interval_coverage(eeg_mean, eeg_target, eeg_quantiles, 9, 89),
        "coverage_95": _interval_coverage(eeg_mean, eeg_target, eeg_quantiles, 1, 97),
        "sample64_coverage_80": _sample_coverage(
            eeg_mean, eeg_target, eeg_quantiles, seed + seconds, sample_count, 0.10, 0.90
        ),
        "sample64_coverage_95": _sample_coverage(
            eeg_mean, eeg_target, eeg_quantiles, seed + 100 + seconds, sample_count, 0.025, 0.975
        ),
    }

    selected = valid[:, modality_index["ECG"]].bool()
    target_waveform = target[selected, modality_index["ECG"], :samples]
    recent_waveform = recent[selected, modality_index["ECG"], :samples]
    target_events = _qrs_events(target_waveform, sample_rate)
    baseline_events = _qrs_events(recent_waveform, sample_rate)
    features, rr_info = _rr_features(
        generated[:, modality_index["ECG"]], recent[:, modality_index["ECG"]], probabilities["ECG"],
        selected, sample_rate, samples, patch_count
    )
    calibrated_rr = _ridge_predict(state["ECG"]["rr_model"], features).clamp(0.25, 2.0)
    target_rr, target_rr_valid = _rr_targets(target_events, sample_rate)
    direct_recent_rr = rr_info["recent_rr"]
    direct_recent_valid = rr_info["recent_rr_valid"]
    generated_rr = rr_info["generated_rr"]
    generated_rr_valid = rr_info["generated_rr_valid"]
    event_probability = _event_probability(
        state["ECG"]["event_probability"], rr_info["generated_events"], sample_rate
    )
    soft_target = _soft_event_targets(target_events, sample_rate)
    soft_baseline = _soft_event_targets(baseline_events, sample_rate)
    rr_quantiles = state["ECG"]["rr_residual_quantiles"]
    paired_recent = target_rr_valid & direct_recent_valid
    paired_generated = target_rr_valid & generated_rr_valid
    distances, symmetric_distances = _event_distances(rr_info["generated_events"], target_events)
    timing_quantiles = state["ECG"]["timing_quantiles_samples"]
    event_scores = _event_scores(rr_info["generated_events"], target_events, sample_rate)
    baseline_scores = _event_scores(baseline_events, target_events, sample_rate)
    metrics["ECG"] = {
        "qrs_brier": float((event_probability - soft_target.float()).square().mean()),
        "recent_qrs_brier": float((soft_baseline - soft_target.float()).square().mean()),
        "qrs_event_precision": event_scores["precision"],
        "qrs_event_recall": event_scores["recall"],
        "qrs_event_f1": event_scores["f1"],
        "recent_qrs_event_f1": baseline_scores["f1"],
        "qrs_timing_chamfer_ms": _finite_mean(symmetric_distances) * 1000.0 / sample_rate,
        "qrs_timing_80_coverage": float((distances <= timing_quantiles[0]).float().mean()) if distances.numel() else float("nan"),
        "qrs_timing_95_coverage": float((distances <= timing_quantiles[1]).float().mean()) if distances.numel() else float("nan"),
        "rr_calibrated_mae_ms": float((calibrated_rr[target_rr_valid] - target_rr[target_rr_valid]).abs().mean() * 1000.0),
        "rr_original_head_mae_ms": float((rr_info["head_rr"][target_rr_valid] - target_rr[target_rr_valid]).abs().mean() * 1000.0),
        "rr_recent_mae_ms": float((direct_recent_rr[paired_recent] - target_rr[paired_recent]).abs().mean() * 1000.0) if paired_recent.any() else float("nan"),
        "rr_generated_mae_ms": float((generated_rr[paired_generated] - target_rr[paired_generated]).abs().mean() * 1000.0) if paired_generated.any() else float("nan"),
        "rr_coverage_80": _interval_coverage(calibrated_rr[target_rr_valid, None], target_rr[target_rr_valid, None], rr_quantiles, 9, 89),
        "rr_coverage_95": _interval_coverage(calibrated_rr[target_rr_valid, None], target_rr[target_rr_valid, None], rr_quantiles, 1, 97),
        "rr_sample64_coverage_80": _sample_coverage(calibrated_rr[target_rr_valid, None], target_rr[target_rr_valid, None], rr_quantiles, seed + 200 + seconds, sample_count, 0.10, 0.90),
        "rr_sample64_coverage_95": _sample_coverage(calibrated_rr[target_rr_valid, None], target_rr[target_rr_valid, None], rr_quantiles, seed + 300 + seconds, sample_count, 0.025, 0.975),
        "predicted_qrs_count_per_record": event_scores["predicted_count_per_record"],
        "target_qrs_count_per_record": event_scores["target_count_per_record"],
    }

    selected = valid[:, modality_index["EMG"]].bool()
    target_emg = target[selected, modality_index["EMG"], :samples]
    generated_emg = generated[selected, modality_index["EMG"], :samples]
    recent_emg = recent[selected, modality_index["EMG"]]
    target_envelope, target_rms, target_burst = _emg_targets(target_emg, patch_samples, sample_rate)
    generated_envelope, generated_rms, _ = _emg_targets(generated_emg, patch_samples, sample_rate)
    recent_envelope, recent_rms, recent_burst = _emg_targets(recent_emg, patch_samples, sample_rate)
    old_envelope = probabilities["EMG"]["envelope_mean"][selected, :patch_count]
    old_rms = probabilities["EMG"]["rms_mean"][selected, :patch_count]
    envelope_features = _emg_regression_features(generated_envelope, old_envelope, recent_envelope)
    rms_features = _emg_regression_features(generated_rms, old_rms, recent_rms)
    calibrated_envelope = _ridge_predict(
        state["EMG"]["envelope_model"], envelope_features.reshape(-1, envelope_features.shape[-1])
    ).reshape_as(target_envelope).clamp_min(0.0)
    calibrated_rms = _ridge_predict(
        state["EMG"]["rms_model"], rms_features.reshape(-1, rms_features.shape[-1])
    ).reshape_as(target_rms).clamp_min(0.0)
    burst_features = _burst_features(
        calibrated_envelope, calibrated_rms, probabilities["EMG"]["burst_logits"][selected, :patch_count]
    )
    burst_probability = _logistic_predict(
        state["EMG"]["burst_model"], burst_features.reshape(-1, burst_features.shape[-1])
    ).reshape_as(target_envelope)
    burst_threshold = float(state["EMG"]["burst_model"]["decision_threshold"])
    envelope_quantiles = state["EMG"]["envelope_residual_quantiles"]
    rms_quantiles = state["EMG"]["rms_residual_quantiles"]
    recent_envelope_prefix = recent_envelope[:, :patch_count]
    recent_rms_prefix = recent_rms[:, :patch_count]
    recent_burst_prefix = recent_burst[:, :patch_count]
    metrics["EMG"] = {
        "envelope_mean_mae": float((calibrated_envelope - target_envelope).abs().mean()),
        "original_head_envelope_mae": float((old_envelope - target_envelope).abs().mean()),
        "recent_envelope_mae": float((recent_envelope_prefix - target_envelope).abs().mean()),
        "envelope_correlation": _finite_mean(_correlation_rows(calibrated_envelope, target_envelope)),
        "rms_mean_mae": float((calibrated_rms - target_rms).abs().mean()),
        "original_head_rms_mae": float((old_rms - target_rms).abs().mean()),
        "recent_rms_mae": float((recent_rms_prefix - target_rms).abs().mean()),
        "envelope_coverage_80": _interval_coverage(calibrated_envelope[..., None], target_envelope[..., None], envelope_quantiles, 9, 89),
        "envelope_coverage_95": _interval_coverage(calibrated_envelope[..., None], target_envelope[..., None], envelope_quantiles, 1, 97),
        "envelope_sample64_coverage_80": _sample_coverage(calibrated_envelope[..., None], target_envelope[..., None], envelope_quantiles, seed + 400 + seconds, sample_count, 0.10, 0.90),
        "envelope_sample64_coverage_95": _sample_coverage(calibrated_envelope[..., None], target_envelope[..., None], envelope_quantiles, seed + 500 + seconds, sample_count, 0.025, 0.975),
        "rms_coverage_80": _interval_coverage(calibrated_rms[..., None], target_rms[..., None], rms_quantiles, 9, 89),
        "rms_coverage_95": _interval_coverage(calibrated_rms[..., None], target_rms[..., None], rms_quantiles, 1, 97),
        "burst_brier": float((burst_probability - target_burst.float()).square().mean()),
        "burst_f1": _binary_f1(burst_probability >= burst_threshold, target_burst),
        "original_head_burst_f1": _binary_f1(
            probabilities["EMG"]["burst_logits"][selected, :patch_count].sigmoid() >= 0.5, target_burst
        ),
        "recent_burst_f1": _binary_f1(recent_burst_prefix, target_burst),
        "burst_decision_threshold": burst_threshold,
    }
    return metrics


def evaluate_renderer_probability_calibration(
    calibration: dict,
    generated: torch.Tensor,
    probabilities: Dict[str, Dict[str, torch.Tensor]],
    recent: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    modalities: Sequence[str],
    sample_rate: int,
    horizons_seconds: Sequence[int],
    patch_samples: int,
    seed: int,
    sample_count: int = 64,
) -> dict:
    modality_index = {name: index for index, name in enumerate(modalities)}
    by_horizon = {}
    for seconds in horizons_seconds:
        by_horizon[str(seconds)] = _probability_metrics_for_horizon(
            calibration["by_horizon_seconds"][str(seconds)],
            generated,
            probabilities,
            recent,
            target,
            valid,
            modality_index,
            sample_rate,
            int(seconds),
            patch_samples,
            seed,
            sample_count,
        )
    return {"by_horizon_seconds": by_horizon, "sample_count": int(sample_count)}


def renderer_probability_gate(metrics: dict) -> dict:
    improved = {"EEG": [], "ECG_QRS": [], "ECG_RR": [], "EMG_envelope": [], "EMG_burst": []}
    calibrated = {"EEG": [], "ECG_RR": [], "EMG_envelope": []}
    for horizon, values in metrics["by_horizon_seconds"].items():
        eeg = values["EEG"]
        ecg = values["ECG"]
        emg = values["EMG"]
        if eeg["spectral_mean_mae"] < eeg["recent_spectral_mae"]:
            improved["EEG"].append(horizon)
        if ecg["qrs_event_f1"] > ecg["recent_qrs_event_f1"] and ecg["qrs_brier"] < ecg["recent_qrs_brier"]:
            improved["ECG_QRS"].append(horizon)
        if ecg["rr_calibrated_mae_ms"] < ecg["rr_original_head_mae_ms"]:
            improved["ECG_RR"].append(horizon)
        if emg["envelope_mean_mae"] < min(emg["original_head_envelope_mae"], emg["recent_envelope_mae"]):
            improved["EMG_envelope"].append(horizon)
        if emg["burst_f1"] > max(emg["original_head_burst_f1"], emg["recent_burst_f1"]):
            improved["EMG_burst"].append(horizon)
        if (
            0.70 <= eeg["coverage_80"] <= 0.90
            and 0.90 <= eeg["coverage_95"] <= 0.99
            and 0.70 <= eeg["sample64_coverage_80"] <= 0.90
            and 0.90 <= eeg["sample64_coverage_95"] <= 0.99
        ):
            calibrated["EEG"].append(horizon)
        if (
            0.70 <= ecg["rr_coverage_80"] <= 0.90
            and 0.90 <= ecg["rr_coverage_95"] <= 0.99
            and 0.70 <= ecg["rr_sample64_coverage_80"] <= 0.90
            and 0.90 <= ecg["rr_sample64_coverage_95"] <= 0.99
        ):
            calibrated["ECG_RR"].append(horizon)
        if (
            0.70 <= emg["envelope_coverage_80"] <= 0.90
            and 0.90 <= emg["envelope_coverage_95"] <= 0.99
            and 0.70 <= emg["envelope_sample64_coverage_80"] <= 0.90
            and 0.90 <= emg["envelope_sample64_coverage_95"] <= 0.99
        ):
            calibrated["EMG_envelope"].append(horizon)
    result = {"improved_horizons": improved, "calibrated_horizons": calibrated}
    result["passed"] = bool(
        len(improved["EEG"]) >= 2
        and len(improved["ECG_QRS"]) >= 2
        and len(improved["ECG_RR"]) >= 2
        and len(improved["EMG_envelope"]) >= 2
        and len(improved["EMG_burst"]) >= 2
        and all(len(values) >= 2 for values in calibrated.values())
    )
    return result
