from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
import torch.nn.functional as F


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
    return F.pad((envelope >= local_max) & (envelope > threshold), (1, 0), value=False)


def _soft_event_targets(events: torch.Tensor, sample_rate: int) -> torch.Tensor:
    tolerance = max(1, round(0.08 * sample_rate))
    return F.max_pool1d(
        events.float().reshape(-1, 1, events.shape[-1]),
        kernel_size=2 * tolerance + 1,
        stride=1,
        padding=tolerance,
    ).reshape_as(events)


def _probability_events(probabilities: torch.Tensor, sample_rate: int) -> torch.Tensor:
    refractory = max(3, round(0.25 * sample_rate))
    if refractory % 2 == 0:
        refractory += 1
    local_max = F.max_pool1d(
        probabilities.reshape(-1, 1, probabilities.shape[-1]),
        kernel_size=refractory,
        stride=1,
        padding=refractory // 2,
    ).reshape_as(probabilities)
    return (probabilities >= local_max) & (probabilities >= 0.5)


def _event_f1(
    predicted_events: torch.Tensor, target_events: torch.Tensor, sample_rate: int
) -> float:
    tolerance = max(1, round(0.08 * sample_rate))
    expanded_target = F.max_pool1d(
        target_events.float().reshape(-1, 1, target_events.shape[-1]),
        kernel_size=2 * tolerance + 1,
        stride=1,
        padding=tolerance,
    ).reshape_as(target_events)
    expanded_prediction = F.max_pool1d(
        predicted_events.float().reshape(-1, 1, predicted_events.shape[-1]),
        kernel_size=2 * tolerance + 1,
        stride=1,
        padding=tolerance,
    ).reshape_as(predicted_events)
    precision_tp = (predicted_events & expanded_target.bool()).sum().double()
    recall_tp = (target_events & expanded_prediction.bool()).sum().double()
    precision = precision_tp / predicted_events.sum().double().clamp_min(1.0)
    recall = recall_tp / target_events.sum().double().clamp_min(1.0)
    return float(2.0 * precision * recall / (precision + recall).clamp_min(1e-12))


def _binary_f1(prediction: torch.Tensor, target: torch.Tensor) -> float:
    predicted = prediction.bool()
    targets = target.bool()
    true_positive = (predicted & targets).sum().double()
    precision = true_positive / predicted.sum().double().clamp_min(1.0)
    recall = true_positive / targets.sum().double().clamp_min(1.0)
    return float(2.0 * precision * recall / (precision + recall).clamp_min(1e-12))


def _rr_targets(
    events: torch.Tensor, sample_rate: int
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = []
    valid = []
    for row in events:
        positions = row.nonzero(as_tuple=False).flatten()
        if len(positions) >= 2:
            targets.append(
                (positions[1:] - positions[:-1]).float().median() / sample_rate
            )
            valid.append(True)
        else:
            targets.append(row.float().sum() * 0.0)
            valid.append(False)
    return torch.stack(targets), torch.tensor(valid, device=events.device)


def _band_log_power(patches: torch.Tensor, sample_rate: int) -> torch.Tensor:
    spectrum = torch.fft.rfft(patches, dim=-1, norm="ortho")
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


def _eeg_targets(
    waveform: torch.Tensor, patch_samples: int, sample_rate: int
) -> torch.Tensor:
    patches = waveform.reshape(waveform.shape[0], -1, patch_samples)
    centered = patches - patches.mean(dim=-1, keepdim=True)
    spectrum = torch.log1p(
        torch.fft.rfft(centered, dim=-1, norm="ortho").abs()
    )
    return torch.cat((spectrum, _band_log_power(centered, sample_rate)), dim=-1)


def _emg_targets(
    waveform: torch.Tensor, patch_samples: int, sample_rate: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    patch_count = waveform.shape[-1] // patch_samples
    patches = waveform.reshape(waveform.shape[0], patch_count, patch_samples)
    envelope = _envelope(
        waveform, round(0.25 * sample_rate), derivative=False
    ).reshape(waveform.shape[0], patch_count, patch_samples).mean(dim=-1)
    rms = patches.square().mean(dim=-1).clamp_min(1e-8).sqrt()
    threshold = envelope.mean(dim=-1, keepdim=True) + envelope.std(
        dim=-1, keepdim=True
    )
    return envelope, rms, envelope > threshold


def _normal_nll(
    mean: torch.Tensor, scale: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    z = (target - mean) / scale.clamp_min(1e-6)
    return (0.5 * z.square() + scale.clamp_min(1e-6).log() + 0.5 * math.log(2 * math.pi)).mean()


def _normal_crps(
    mean: torch.Tensor, scale: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    sigma = scale.clamp_min(1e-6)
    z = (target - mean) / sigma
    phi = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    return sigma * (z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def _focal_binary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    targets = target.to(logits.dtype)
    probability = logits.sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability_target = probability * targets + (1.0 - probability) * (1.0 - targets)
    alpha_target = float(alpha) * targets + (1.0 - float(alpha)) * (1.0 - targets)
    return (alpha_target * (1.0 - probability_target).pow(float(gamma)) * cross_entropy).mean()


def _refractory_qrs_loss(
    logits: torch.Tensor, events: torch.Tensor, sample_rate: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    targets = events.to(logits.dtype)
    positives = targets.sum().clamp_min(1.0)
    negatives = targets.numel() - positives
    positive_weight = (negatives / positives).clamp(1.0, 64.0)
    balanced_bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=positive_weight
    )
    probability = logits.sigmoid()
    intersection = (probability * targets).sum()
    dice_loss = 1.0 - (2.0 * intersection + 1.0) / (
        probability.sum() + targets.sum() + 1.0
    )
    refractory = max(1, round(0.25 * sample_rate))
    pooled = F.max_pool1d(
        probability.reshape(-1, 1, probability.shape[-1]),
        kernel_size=2 * refractory + 1,
        stride=1,
        padding=refractory,
    ).reshape_as(probability)
    conflict = (probability * (pooled - probability).clamp_min(0.0)).mean()
    return 0.5 * balanced_bce + 0.4 * dice_loss + 0.1 * conflict, balanced_bce, conflict


def _rr_centers(logits: torch.Tensor) -> torch.Tensor:
    return torch.linspace(
        0.25, 2.0, logits.shape[-1], device=logits.device, dtype=logits.dtype
    )


def _rr_distribution_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    centers = _rr_centers(logits)
    spacing = float((centers[1] - centers[0]).detach())
    target_scale = max(0.05, spacing)
    soft_target = torch.exp(
        -0.5 * ((centers.unsqueeze(0) - target.unsqueeze(-1)) / target_scale).square()
    )
    soft_target = soft_target / soft_target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return -(soft_target * logits.log_softmax(dim=-1)).sum(dim=-1).mean()


def _categorical_quantile(
    probabilities: torch.Tensor, centers: torch.Tensor, quantile: float
) -> torch.Tensor:
    indices = (probabilities.cumsum(dim=-1) >= float(quantile)).long().argmax(dim=-1)
    return centers[indices]


def probabilistic_waveform_loss(
    predictions: Dict[str, Dict[str, torch.Tensor]],
    target: torch.Tensor,
    valid: torch.Tensor,
    modalities: Sequence[str],
    sample_rate: int,
    horizons_seconds: Sequence[int],
    patch_samples: int,
) -> Dict[str, torch.Tensor]:
    modality_indices = {modality: index for index, modality in enumerate(modalities)}
    losses = {"EEG": [], "ECG": [], "EMG": []}
    components = {
        "eeg_nll": [],
        "qrs_bce": [],
        "qrs_refractory": [],
        "rr_nll": [],
        "emg_nll": [],
        "burst_focal": [],
    }
    for seconds in horizons_seconds:
        samples = int(seconds) * int(sample_rate)
        if samples % int(patch_samples):
            raise ValueError("probabilistic waveform horizon must align to patches")
        patch_count = samples // int(patch_samples)
        eeg_selected = valid[:, modality_indices["EEG"]]
        eeg_target = _eeg_targets(
            target[eeg_selected, modality_indices["EEG"], :samples],
            patch_samples,
            sample_rate,
        )
        eeg_nll = _normal_nll(
            predictions["EEG"]["spectral_mean"][eeg_selected, :patch_count],
            predictions["EEG"]["spectral_scale"][eeg_selected, :patch_count],
            eeg_target,
        )
        losses["EEG"].append(eeg_nll)
        components["eeg_nll"].append(eeg_nll)

        ecg_selected = valid[:, modality_indices["ECG"]]
        ecg_target = target[ecg_selected, modality_indices["ECG"], :samples]
        ecg_events = _qrs_events(ecg_target, sample_rate)
        qrs_logits = predictions["ECG"]["qrs_logits"][ecg_selected, :samples]
        rr_target, rr_valid = _rr_targets(ecg_events, sample_rate)
        if "rr_logits" in predictions["ECG"]:
            qrs_loss, qrs_bce, qrs_refractory = _refractory_qrs_loss(
                qrs_logits, ecg_events, sample_rate
            )
            rr_logits = predictions["ECG"]["rr_logits"][
                ecg_selected, :patch_count
            ].mean(dim=1)
            rr_nll = (
                _rr_distribution_loss(rr_logits[rr_valid], rr_target[rr_valid])
                if rr_valid.any()
                else qrs_loss * 0.0
            )
        else:
            soft_events = _soft_event_targets(ecg_events, sample_rate)
            qrs_bce = F.binary_cross_entropy_with_logits(qrs_logits, soft_events)
            qrs_refractory = qrs_bce * 0.0
            qrs_loss = qrs_bce
            rr_mean = predictions["ECG"]["rr_mean_seconds"][
                ecg_selected, :patch_count
            ].mean(dim=1)
            rr_scale = predictions["ECG"]["rr_scale_seconds"][
                ecg_selected, :patch_count
            ].mean(dim=1)
            rr_nll = (
                _normal_nll(rr_mean[rr_valid], rr_scale[rr_valid], rr_target[rr_valid])
                if rr_valid.any()
                else qrs_bce * 0.0
            )
        losses["ECG"].append(0.5 * (qrs_loss + rr_nll))
        components["qrs_bce"].append(qrs_bce)
        components["qrs_refractory"].append(qrs_refractory)
        components["rr_nll"].append(rr_nll)

        emg_selected = valid[:, modality_indices["EMG"]]
        emg_target = target[emg_selected, modality_indices["EMG"], :samples]
        envelope, rms, burst = _emg_targets(emg_target, patch_samples, sample_rate)
        envelope_nll = _normal_nll(
            predictions["EMG"]["envelope_mean"][emg_selected, :patch_count],
            predictions["EMG"]["envelope_scale"][emg_selected, :patch_count],
            envelope,
        )
        rms_nll = _normal_nll(
            predictions["EMG"]["rms_mean"][emg_selected, :patch_count],
            predictions["EMG"]["rms_scale"][emg_selected, :patch_count],
            rms,
        )
        burst_focal = _focal_binary_loss(
            predictions["EMG"]["burst_logits"][emg_selected, :patch_count], burst
        )
        emg_nll = 0.5 * (envelope_nll + rms_nll)
        losses["EMG"].append(0.5 * (emg_nll + burst_focal))
        components["emg_nll"].append(emg_nll)
        components["burst_focal"].append(burst_focal)
    group_losses = {group: torch.stack(values).mean() for group, values in losses.items()}
    total = torch.stack(tuple(group_losses.values())).mean()
    return {
        "loss": total,
        **{f"{group.lower()}_loss": value for group, value in group_losses.items()},
        **{name: torch.stack(values).mean() for name, values in components.items()},
    }


def _coverage(
    mean: torch.Tensor, scale: torch.Tensor, target: torch.Tensor, z_value: float
) -> float:
    lower = mean - float(z_value) * scale
    upper = mean + float(z_value) * scale
    return float(((target >= lower) & (target <= upper)).float().mean())


def probabilistic_waveform_metrics(
    predictions: Dict[str, Dict[str, torch.Tensor]],
    target: torch.Tensor,
    baseline_waveform: torch.Tensor,
    valid: torch.Tensor,
    modalities: Sequence[str],
    sample_rate: int,
    horizons_seconds: Sequence[int],
    patch_samples: int,
) -> Dict[str, object]:
    probability = {
        modality: {name: value.detach().cpu().float() for name, value in values.items()}
        for modality, values in predictions.items()
    }
    targets = target.detach().cpu().float()
    baseline = baseline_waveform.detach().cpu().float()
    mask = valid.detach().cpu().bool()
    modality_indices = {modality: index for index, modality in enumerate(modalities)}
    by_horizon = {}
    for seconds in horizons_seconds:
        samples = int(seconds) * int(sample_rate)
        patch_count = samples // int(patch_samples)
        horizon = {}

        eeg_selected = mask[:, modality_indices["EEG"]]
        eeg_target = _eeg_targets(
            targets[eeg_selected, modality_indices["EEG"], :samples],
            patch_samples,
            sample_rate,
        )
        eeg_baseline = _eeg_targets(
            baseline[eeg_selected, modality_indices["EEG"], :samples],
            patch_samples,
            sample_rate,
        )
        eeg_mean = probability["EEG"]["spectral_mean"][eeg_selected, :patch_count]
        eeg_scale = probability["EEG"]["spectral_scale"][eeg_selected, :patch_count]
        horizon["EEG"] = {
            "spectral_mean_mae": float((eeg_mean - eeg_target).abs().mean()),
            "baseline_spectral_mae": float((eeg_baseline - eeg_target).abs().mean()),
            "spectral_nll": float(_normal_nll(eeg_mean, eeg_scale, eeg_target)),
            "spectral_crps": float(_normal_crps(eeg_mean, eeg_scale, eeg_target).mean()),
            "coverage_80": _coverage(eeg_mean, eeg_scale, eeg_target, 1.2815515655),
            "coverage_95": _coverage(eeg_mean, eeg_scale, eeg_target, 1.9599639845),
        }

        ecg_selected = mask[:, modality_indices["ECG"]]
        ecg_target = targets[ecg_selected, modality_indices["ECG"], :samples]
        ecg_baseline = baseline[ecg_selected, modality_indices["ECG"], :samples]
        target_events = _qrs_events(ecg_target, sample_rate)
        baseline_events = _qrs_events(ecg_baseline, sample_rate)
        soft_target = _soft_event_targets(target_events, sample_rate)
        soft_baseline = _soft_event_targets(baseline_events, sample_rate)
        qrs_probability = probability["ECG"]["qrs_logits"][ecg_selected, :samples].sigmoid()
        predicted_events = _probability_events(qrs_probability, sample_rate)
        rr_target, rr_valid = _rr_targets(target_events, sample_rate)
        baseline_rr, baseline_rr_valid = _rr_targets(baseline_events, sample_rate)
        rr_logits = probability["ECG"].get("rr_logits")
        if rr_logits is not None:
            rr_logits = rr_logits[ecg_selected, :patch_count].mean(dim=1)
            rr_probability = rr_logits.softmax(dim=-1)
            rr_centers = _rr_centers(rr_logits)
            rr_mean = (rr_probability * rr_centers).sum(dim=-1)
            rr_scale = (
                rr_probability * (rr_centers - rr_mean.unsqueeze(-1)).square()
            ).sum(dim=-1).clamp_min(1e-6).sqrt()
        else:
            rr_probability = None
            rr_centers = None
            rr_mean = probability["ECG"]["rr_mean_seconds"][
                ecg_selected, :patch_count
            ].mean(dim=1)
            rr_scale = probability["ECG"]["rr_scale_seconds"][
                ecg_selected, :patch_count
            ].mean(dim=1)
        paired_rr = rr_valid & baseline_rr_valid
        ecg_metrics = {
            "qrs_brier": float((qrs_probability - soft_target).square().mean()),
            "qrs_exact_brier": float(
                (qrs_probability - target_events.float()).square().mean()
            ),
            "baseline_qrs_brier": float((soft_baseline - soft_target).square().mean()),
            "qrs_event_f1": _event_f1(predicted_events, target_events, sample_rate),
            "baseline_qrs_event_f1": _event_f1(
                baseline_events, target_events, sample_rate
            ),
            "rr_mean_mae_ms": float(
                (rr_mean[rr_valid] - rr_target[rr_valid]).abs().mean() * 1000.0
            ),
            "baseline_rr_mae_ms": float(
                (baseline_rr[paired_rr] - rr_target[paired_rr]).abs().mean() * 1000.0
            ),
            "predicted_qrs_count_per_record": float(predicted_events.sum(dim=-1).float().mean()),
            "target_qrs_count_per_record": float(target_events.sum(dim=-1).float().mean()),
        }
        if rr_probability is not None and rr_centers is not None:
            selected_probability = rr_probability[rr_valid]
            selected_target = rr_target[rr_valid]
            cdf = selected_probability.cumsum(dim=-1)
            observed_cdf = (
                rr_centers.unsqueeze(0) >= selected_target.unsqueeze(-1)
            ).to(cdf.dtype)
            bin_width = float(rr_centers[1] - rr_centers[0])
            ecg_metrics.update(
                {
                    "rr_nll": float(
                        _rr_distribution_loss(rr_logits[rr_valid], selected_target)
                    ),
                    "rr_crps_ms": float(
                        ((cdf - observed_cdf).square().sum(dim=-1) * bin_width).mean()
                        * 1000.0
                    ),
                    "rr_coverage_80": float(
                        (
                            (selected_target >= _categorical_quantile(selected_probability, rr_centers, 0.10))
                            & (selected_target <= _categorical_quantile(selected_probability, rr_centers, 0.90))
                        ).float().mean()
                    ),
                    "rr_coverage_95": float(
                        (
                            (selected_target >= _categorical_quantile(selected_probability, rr_centers, 0.025))
                            & (selected_target <= _categorical_quantile(selected_probability, rr_centers, 0.975))
                        ).float().mean()
                    ),
                }
            )
        else:
            ecg_metrics.update(
                {
                    "rr_nll": float(
                        _normal_nll(rr_mean[rr_valid], rr_scale[rr_valid], rr_target[rr_valid])
                    ),
                    "rr_crps_ms": float(
                        _normal_crps(rr_mean[rr_valid], rr_scale[rr_valid], rr_target[rr_valid]).mean()
                        * 1000.0
                    ),
                    "rr_coverage_80": _coverage(
                        rr_mean[rr_valid], rr_scale[rr_valid], rr_target[rr_valid], 1.2815515655
                    ),
                    "rr_coverage_95": _coverage(
                        rr_mean[rr_valid], rr_scale[rr_valid], rr_target[rr_valid], 1.9599639845
                    ),
                }
            )
        horizon["ECG"] = ecg_metrics

        emg_selected = mask[:, modality_indices["EMG"]]
        emg_target = targets[emg_selected, modality_indices["EMG"], :samples]
        emg_baseline = baseline[emg_selected, modality_indices["EMG"], :samples]
        envelope, rms, burst = _emg_targets(emg_target, patch_samples, sample_rate)
        baseline_envelope, _, baseline_burst = _emg_targets(
            emg_baseline, patch_samples, sample_rate
        )
        envelope_mean = probability["EMG"]["envelope_mean"][emg_selected, :patch_count]
        envelope_scale = probability["EMG"]["envelope_scale"][emg_selected, :patch_count]
        burst_probability = probability["EMG"]["burst_logits"][emg_selected, :patch_count].sigmoid()
        horizon["EMG"] = {
            "envelope_mean_mae": float((envelope_mean - envelope).abs().mean()),
            "baseline_envelope_mae": float((baseline_envelope - envelope).abs().mean()),
            "envelope_nll": float(_normal_nll(envelope_mean, envelope_scale, envelope)),
            "envelope_crps": float(_normal_crps(envelope_mean, envelope_scale, envelope).mean()),
            "envelope_coverage_80": _coverage(
                envelope_mean, envelope_scale, envelope, 1.2815515655
            ),
            "envelope_coverage_95": _coverage(
                envelope_mean, envelope_scale, envelope, 1.9599639845
            ),
            "burst_brier": float((burst_probability - burst.float()).square().mean()),
            "baseline_burst_brier": float(
                (baseline_burst.float() - burst.float()).square().mean()
            ),
            "burst_f1": _binary_f1(burst_probability >= 0.5, burst),
            "baseline_burst_f1": _binary_f1(baseline_burst, burst),
        }
        by_horizon[str(seconds)] = horizon
    return {"by_horizon_seconds": by_horizon}


def probabilistic_waveform_gate_result(
    waveform_metrics: Dict[str, object],
    baseline_waveform_metrics: Dict[str, object],
    probability_metrics: Dict[str, object],
    required_relative_mae_improvement: float = 0.02,
) -> Dict[str, object]:
    model_mae = float(waveform_metrics["all"]["mean_standardized_mae"])
    baseline_mae = float(
        baseline_waveform_metrics["all"]["mean_standardized_mae"]
    )
    ecg_horizons = []
    eeg_horizons = []
    emg_horizons = []
    calibrated_horizons = []
    for horizon, values in probability_metrics["by_horizon_seconds"].items():
        ecg = values["ECG"]
        brier_improved = float(ecg["qrs_brier"]) < float(ecg["baseline_qrs_brier"])
        event_or_rr_improved = (
            float(ecg["qrs_event_f1"]) > float(ecg["baseline_qrs_event_f1"])
            or float(ecg["rr_mean_mae_ms"]) < float(ecg["baseline_rr_mae_ms"])
        )
        if brier_improved and event_or_rr_improved:
            ecg_horizons.append(horizon)
        if float(values["EEG"]["spectral_mean_mae"]) < float(
            values["EEG"]["baseline_spectral_mae"]
        ):
            eeg_horizons.append(horizon)
        if float(values["EMG"]["envelope_mean_mae"]) < float(
            values["EMG"]["baseline_envelope_mae"]
        ):
            emg_horizons.append(horizon)
        coverage = float(ecg["rr_coverage_80"])
        if 0.70 <= coverage <= 0.90:
            calibrated_horizons.append(horizon)
    threshold = baseline_mae * (1.0 - required_relative_mae_improvement)
    result = {
        "overall_waveform_mae_threshold": threshold,
        "overall_waveform_mae_improved": model_mae < threshold,
        "ecg_probability_improved_horizons": ecg_horizons,
        "eeg_spectral_mean_improved_horizons": eeg_horizons,
        "emg_envelope_mean_improved_horizons": emg_horizons,
        "rr_80pct_calibrated_horizons": calibrated_horizons,
    }
    result["passed"] = bool(
        result["overall_waveform_mae_improved"]
        and len(ecg_horizons) >= 2
        and (len(eeg_horizons) >= 2 or len(emg_horizons) >= 2)
        and len(calibrated_horizons) >= 2
    )
    return result


def refractory_ecg_gate_result(
    waveform_metrics: Dict[str, object],
    baseline_waveform_metrics: Dict[str, object],
    probability_metrics: Dict[str, object],
    required_relative_mae_improvement: float = 0.02,
) -> Dict[str, object]:
    result = probabilistic_waveform_gate_result(
        waveform_metrics,
        baseline_waveform_metrics,
        probability_metrics,
        required_relative_mae_improvement,
    )
    brier_horizons = []
    event_horizons = []
    rr_horizons = []
    for horizon, values in probability_metrics["by_horizon_seconds"].items():
        ecg = values["ECG"]
        if float(ecg["qrs_brier"]) < float(ecg["baseline_qrs_brier"]):
            brier_horizons.append(horizon)
        if float(ecg["qrs_event_f1"]) > float(ecg["baseline_qrs_event_f1"]):
            event_horizons.append(horizon)
        if float(ecg["rr_mean_mae_ms"]) < float(ecg["baseline_rr_mae_ms"]):
            rr_horizons.append(horizon)
    result.update(
        {
            "qrs_brier_improved_horizons": brier_horizons,
            "qrs_event_f1_improved_horizons": event_horizons,
            "rr_mean_mae_improved_horizons": rr_horizons,
        }
    )
    result["passed"] = bool(
        result["overall_waveform_mae_improved"]
        and len(brier_horizons) >= 2
        and len(event_horizons) >= 2
        and len(rr_horizons) >= 2
        and (
            len(result["eeg_spectral_mean_improved_horizons"]) >= 2
            or len(result["emg_envelope_mean_improved_horizons"]) >= 2
        )
        and len(result["rr_80pct_calibrated_horizons"]) >= 2
    )
    return result
