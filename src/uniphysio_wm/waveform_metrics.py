from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn.functional as F


def _mean_sample_correlation(prediction: torch.Tensor, target: torch.Tensor) -> float:
    x = prediction.double() - prediction.double().mean(dim=-1, keepdim=True)
    y = target.double() - target.double().mean(dim=-1, keepdim=True)
    denominator = x.square().sum(dim=-1).sqrt() * y.square().sum(dim=-1).sqrt()
    correlation = (x * y).sum(dim=-1) / denominator.clamp_min(1e-12)
    return float(correlation.mean())


def _mean_spectral_correlation(prediction: torch.Tensor, target: torch.Tensor) -> float:
    predicted = torch.log1p(torch.fft.rfft(prediction.double(), dim=-1, norm="ortho").abs())
    targets = torch.log1p(torch.fft.rfft(target.double(), dim=-1, norm="ortho").abs())
    return _mean_sample_correlation(predicted, targets)


def _envelope(signals: torch.Tensor, kernel: int, derivative: bool) -> torch.Tensor:
    values = signals[..., 1:] - signals[..., :-1] if derivative else signals
    values = values.square() if derivative else values.abs()
    kernel = max(1, min(int(kernel), values.shape[-1]))
    return F.avg_pool1d(
        values.reshape(-1, 1, values.shape[-1]),
        kernel_size=kernel,
        stride=1,
        padding=kernel // 2,
    )[..., : values.shape[-1]].reshape(*values.shape)


def _qrs_events(signals: torch.Tensor, sample_rate: int) -> torch.Tensor:
    envelope = _envelope(signals, round(0.12 * sample_rate), derivative=True)
    threshold = envelope.mean(dim=-1, keepdim=True) + envelope.std(
        dim=-1, keepdim=True
    )
    refractory = max(3, round(0.25 * sample_rate))
    if refractory % 2 == 0:
        refractory += 1
    local_max = F.max_pool1d(
        envelope.reshape(-1, 1, envelope.shape[-1]),
        kernel_size=refractory,
        stride=1,
        padding=refractory // 2,
    ).reshape_as(envelope)
    return (envelope >= local_max) & (envelope > threshold)


def _event_f1(prediction: torch.Tensor, target: torch.Tensor, sample_rate: int) -> float:
    predicted_events = _qrs_events(prediction, sample_rate)
    target_events = _qrs_events(target, sample_rate)
    return _binary_event_f1(predicted_events, target_events, sample_rate)


def _binary_event_f1(
    predicted_events: torch.Tensor,
    target_events: torch.Tensor,
    sample_rate: int,
) -> float:
    tolerance = max(1, round(0.08 * sample_rate))
    kernel = 2 * tolerance + 1
    expanded_target = F.max_pool1d(
        target_events.float().reshape(-1, 1, target_events.shape[-1]),
        kernel_size=kernel,
        stride=1,
        padding=tolerance,
    ).reshape_as(target_events)
    expanded_prediction = F.max_pool1d(
        predicted_events.float().reshape(-1, 1, predicted_events.shape[-1]),
        kernel_size=kernel,
        stride=1,
        padding=tolerance,
    ).reshape_as(predicted_events)
    true_positive_precision = (predicted_events & expanded_target.bool()).sum().double()
    true_positive_recall = (target_events & expanded_prediction.bool()).sum().double()
    precision = true_positive_precision / predicted_events.sum().double().clamp_min(1.0)
    recall = true_positive_recall / target_events.sum().double().clamp_min(1.0)
    return float(2.0 * precision * recall / (precision + recall).clamp_min(1e-12))


def _binary_f1(prediction: torch.Tensor, target: torch.Tensor) -> float:
    predicted = prediction.bool()
    targets = target.bool()
    true_positive = (predicted & targets).sum().double()
    precision = true_positive / predicted.sum().double().clamp_min(1.0)
    recall = true_positive / targets.sum().double().clamp_min(1.0)
    return float(2.0 * precision * recall / (precision + recall).clamp_min(1e-12))


def _band_log_power(
    patches: torch.Tensor, sample_rate: int
) -> torch.Tensor:
    spectrum = torch.fft.rfft(patches.double(), dim=-1, norm="ortho")
    power = spectrum.real.square() + spectrum.imag.square()
    frequencies = torch.fft.rfftfreq(
        patches.shape[-1], d=1.0 / sample_rate, device=patches.device
    )
    values = []
    for low, high in ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0)):
        selected = (frequencies >= low) & (frequencies < high)
        if selected.any():
            values.append(torch.log1p(power[..., selected].mean(dim=-1)))
        else:
            values.append(power.sum(dim=-1) * 0.0)
    return torch.stack(values, dim=-1)


def _rr_interval_mae_ms(
    prediction: torch.Tensor, target: torch.Tensor, sample_rate: int
) -> float:
    predicted_events = _qrs_events(prediction, sample_rate)
    target_events = _qrs_events(target, sample_rate)
    errors = []
    for predicted_row, target_row in zip(predicted_events, target_events):
        predicted_positions = predicted_row.nonzero(as_tuple=False).flatten()
        target_positions = target_row.nonzero(as_tuple=False).flatten()
        if len(predicted_positions) < 2 or len(target_positions) < 2:
            continue
        predicted_rr = (predicted_positions[1:] - predicted_positions[:-1]).double()
        target_rr = (target_positions[1:] - target_positions[:-1]).double()
        errors.append((predicted_rr.median() - target_rr.median()).abs())
    if not errors:
        return float(prediction.shape[-1] * 1000.0 / sample_rate)
    return float(torch.stack(errors).mean() * 1000.0 / sample_rate)


def waveform_forecast_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    modalities: Sequence[str],
    sample_rate: int,
    horizons_seconds: Sequence[int],
) -> Dict[str, object]:
    predicted = prediction.detach().cpu().float()
    targets = target.detach().cpu().float()
    mask = valid.detach().cpu().bool()
    if predicted.shape != targets.shape or predicted.ndim != 3:
        raise ValueError("waveform prediction and target shapes must match")
    if mask.shape != predicted.shape[:2] or predicted.shape[1] != len(modalities):
        raise ValueError("waveform modality validity shape is invalid")
    by_horizon = {}
    modality_mae = {modality: [] for modality in modalities}
    for seconds in horizons_seconds:
        samples = int(seconds) * int(sample_rate)
        if samples < 2 or samples > predicted.shape[-1]:
            raise ValueError("waveform metric horizon is invalid")
        horizon_modalities = {}
        for modality_index, modality in enumerate(modalities):
            selected = mask[:, modality_index]
            if not selected.any():
                continue
            predicted_segment = predicted[selected, modality_index, :samples]
            target_segment = targets[selected, modality_index, :samples]
            mae = float((predicted_segment - target_segment).abs().mean())
            predicted_spectrum = torch.log1p(
                torch.fft.rfft(predicted_segment.double(), dim=-1, norm="ortho").abs()
            )
            target_spectrum = torch.log1p(
                torch.fft.rfft(target_segment.double(), dim=-1, norm="ortho").abs()
            )
            result = {
                "standardized_mae": mae,
                "waveform_correlation": _mean_sample_correlation(
                    predicted_segment, target_segment
                ),
                "log_spectral_mae": float(
                    (predicted_spectrum - target_spectrum).abs().mean()
                ),
                "spectral_correlation": _mean_spectral_correlation(
                    predicted_segment, target_segment
                ),
                "valid_sequences": int(selected.sum()),
            }
            if modality == "ECG":
                result["qrs_event_f1"] = _event_f1(
                    predicted_segment, target_segment, sample_rate
                )
                result["rr_interval_mae_ms"] = _rr_interval_mae_ms(
                    predicted_segment, target_segment, sample_rate
                )
                result["qrs_envelope_correlation"] = _mean_sample_correlation(
                    _envelope(predicted_segment, round(0.12 * sample_rate), True),
                    _envelope(target_segment, round(0.12 * sample_rate), True),
                )
            elif modality == "EMG":
                result["envelope_correlation"] = _mean_sample_correlation(
                    _envelope(predicted_segment, round(0.25 * sample_rate), False),
                    _envelope(target_segment, round(0.25 * sample_rate), False),
                )
            horizon_modalities[modality] = result
            modality_mae[modality].append(mae)
        by_horizon[str(seconds)] = {
            "mean_standardized_mae": float(
                sum(value["standardized_mae"] for value in horizon_modalities.values())
                / len(horizon_modalities)
            ),
            "by_modality": horizon_modalities,
        }
    by_modality = {
        modality: {"mean_standardized_mae": float(sum(values) / len(values))}
        for modality, values in modality_mae.items()
        if values
    }
    return {
        "all": {
            "mean_standardized_mae": float(
                sum(value["mean_standardized_mae"] for value in by_horizon.values())
                / len(by_horizon)
            )
        },
        "by_horizon_seconds": by_horizon,
        "by_modality": by_modality,
    }


def structured_head_metrics(
    predictions: Dict[str, Dict[str, torch.Tensor]],
    target: torch.Tensor,
    valid: torch.Tensor,
    sample_rate: int,
    horizons_seconds: Sequence[int],
    patch_samples: int,
) -> Dict[str, object]:
    structures = {
        modality: {name: value.detach().cpu().float() for name, value in values.items()}
        for modality, values in predictions.items()
    }
    targets = target.detach().cpu().float()
    mask = valid.detach().cpu().bool()
    modality_index = {"EEG": 0, "ECG": 1, "EMG": 2}
    by_horizon = {}
    for seconds in horizons_seconds:
        samples = int(seconds) * int(sample_rate)
        if samples % int(patch_samples):
            raise ValueError("structured metric horizon must align to patches")
        patch_count = samples // int(patch_samples)
        horizon = {}
        eeg_selected = mask[:, modality_index["EEG"]]
        eeg_target = targets[eeg_selected, modality_index["EEG"], :samples]
        eeg_patches = eeg_target.reshape(-1, patch_count, patch_samples)
        eeg_centered = eeg_patches - eeg_patches.mean(dim=-1, keepdim=True)
        eeg_spectrum = torch.log1p(
            torch.fft.rfft(eeg_centered.double(), dim=-1, norm="ortho").abs()
        )
        horizon["EEG"] = {
            "head_log_spectral_mae": float(
                (
                    structures["EEG"]["log_spectrum"][eeg_selected, :patch_count]
                    - eeg_spectrum
                )
                .abs()
                .mean()
            ),
            "head_band_log_power_mae": float(
                (
                    structures["EEG"]["band_log_power"][eeg_selected, :patch_count]
                    - _band_log_power(eeg_centered, sample_rate)
                )
                .abs()
                .mean()
            ),
        }
        ecg_selected = mask[:, modality_index["ECG"]]
        ecg_target = targets[ecg_selected, modality_index["ECG"], :samples]
        ecg_events = F.pad(_qrs_events(ecg_target, sample_rate), (1, 0), value=False)
        predicted_events = (
            structures["ECG"]["qrs_logits"][ecg_selected, :samples].sigmoid() >= 0.5
        )
        rr_errors = []
        predicted_rr = structures["ECG"]["rr_seconds"][
            ecg_selected, :patch_count
        ].mean(dim=1)
        for row_index, events in enumerate(ecg_events):
            positions = events.nonzero(as_tuple=False).flatten()
            if len(positions) >= 2:
                target_rr = (
                    (positions[1:] - positions[:-1]).double().median() / sample_rate
                )
                rr_errors.append((predicted_rr[row_index].double() - target_rr).abs())
        horizon["ECG"] = {
            "head_qrs_event_f1": _binary_event_f1(
                predicted_events, ecg_events, sample_rate
            ),
            "head_rr_interval_mae_ms": float(
                torch.stack(rr_errors).mean() * 1000.0
                if rr_errors
                else samples * 1000.0 / sample_rate
            ),
        }
        emg_selected = mask[:, modality_index["EMG"]]
        emg_target = targets[emg_selected, modality_index["EMG"], :samples]
        emg_patches = emg_target.reshape(-1, patch_count, patch_samples)
        emg_envelope = _envelope(
            emg_target, round(0.25 * sample_rate), derivative=False
        ).reshape(-1, patch_count, patch_samples).mean(dim=-1)
        emg_rms = emg_patches.square().mean(dim=-1).clamp_min(1e-8).sqrt()
        burst_threshold = emg_envelope.mean(dim=-1, keepdim=True) + emg_envelope.std(
            dim=-1, keepdim=True
        )
        emg_burst = emg_envelope > burst_threshold
        horizon["EMG"] = {
            "head_envelope_correlation": _mean_sample_correlation(
                structures["EMG"]["envelope"][emg_selected, :patch_count],
                emg_envelope,
            ),
            "head_rms_mae": float(
                (
                    structures["EMG"]["rms"][emg_selected, :patch_count] - emg_rms
                )
                .abs()
                .mean()
            ),
            "head_burst_f1": _binary_f1(
                structures["EMG"]["burst_logits"][emg_selected, :patch_count]
                .sigmoid()
                >= 0.5,
                emg_burst,
            ),
        }
        by_horizon[str(seconds)] = horizon
    return {"by_horizon_seconds": by_horizon}


def waveform_gate_result(
    model_metrics: Dict[str, object],
    baseline_metrics: Dict[str, object],
    required_relative_mae_improvement: float = 0.02,
) -> Dict[str, object]:
    model_score = float(model_metrics["all"]["mean_standardized_mae"])
    baseline_score = float(baseline_metrics["all"]["mean_standardized_mae"])
    primary_metrics = {
        "EEG": ("log_spectral_mae", "lower"),
        "ECG": ("qrs_event_f1", "higher"),
        "EMG": ("envelope_correlation", "higher"),
    }
    improved_horizons = {}
    for modality, (metric, direction) in primary_metrics.items():
        improved = []
        for horizon, model_horizon in model_metrics["by_horizon_seconds"].items():
            model_value = float(model_horizon["by_modality"][modality][metric])
            baseline_value = float(
                baseline_metrics["by_horizon_seconds"][horizon]["by_modality"][modality][metric]
            )
            if (direction == "lower" and model_value < baseline_value) or (
                direction == "higher" and model_value > baseline_value
            ):
                improved.append(horizon)
        improved_horizons[modality] = improved
    improved_modalities = [
        modality for modality, horizons in improved_horizons.items() if len(horizons) >= 2
    ]
    result = {
        "required_relative_mae_improvement": required_relative_mae_improvement,
        "overall_mae_threshold": baseline_score
        * (1.0 - required_relative_mae_improvement),
        "overall_mae_improved": model_score
        < baseline_score * (1.0 - required_relative_mae_improvement),
        "primary_metric_improved_horizons": improved_horizons,
        "modalities_improved_on_at_least_two_horizons": improved_modalities,
    }
    result["passed"] = bool(
        result["overall_mae_improved"] and len(improved_modalities) >= 2
    )
    return result


def structured_waveform_gate_result(
    model_metrics: Dict[str, object],
    baseline_metrics: Dict[str, object],
    required_relative_mae_improvement: float = 0.02,
) -> Dict[str, object]:
    model_score = float(model_metrics["all"]["mean_standardized_mae"])
    baseline_score = float(baseline_metrics["all"]["mean_standardized_mae"])
    improved_horizons = {"EEG": [], "ECG": [], "EMG": []}
    ecg_metric_by_horizon = {}
    for horizon, model_horizon in model_metrics["by_horizon_seconds"].items():
        baseline_horizon = baseline_metrics["by_horizon_seconds"][horizon]
        model_by_modality = model_horizon["by_modality"]
        baseline_by_modality = baseline_horizon["by_modality"]
        if float(model_by_modality["EEG"]["log_spectral_mae"]) < float(
            baseline_by_modality["EEG"]["log_spectral_mae"]
        ):
            improved_horizons["EEG"].append(horizon)
        qrs_improved = float(model_by_modality["ECG"]["qrs_event_f1"]) > float(
            baseline_by_modality["ECG"]["qrs_event_f1"]
        )
        rr_improved = float(model_by_modality["ECG"]["rr_interval_mae_ms"]) < float(
            baseline_by_modality["ECG"]["rr_interval_mae_ms"]
        )
        ecg_metric_by_horizon[horizon] = {
            "qrs_event_f1_improved": qrs_improved,
            "rr_interval_mae_improved": rr_improved,
        }
        if qrs_improved or rr_improved:
            improved_horizons["ECG"].append(horizon)
        if float(model_by_modality["EMG"]["envelope_correlation"]) > float(
            baseline_by_modality["EMG"]["envelope_correlation"]
        ):
            improved_horizons["EMG"].append(horizon)
    improved_modalities = [
        modality for modality, horizons in improved_horizons.items() if len(horizons) >= 2
    ]
    overall_threshold = baseline_score * (1.0 - required_relative_mae_improvement)
    result = {
        "required_relative_mae_improvement": required_relative_mae_improvement,
        "overall_mae_threshold": overall_threshold,
        "overall_mae_improved": model_score < overall_threshold,
        "primary_metric_improved_horizons": improved_horizons,
        "ecg_metric_improvement_by_horizon": ecg_metric_by_horizon,
        "modalities_improved_on_at_least_two_horizons": improved_modalities,
        "ecg_required": True,
    }
    result["passed"] = bool(
        result["overall_mae_improved"]
        and len(improved_modalities) >= 2
        and "ECG" in improved_modalities
    )
    return result
