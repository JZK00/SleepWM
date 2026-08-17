from __future__ import annotations

from typing import Dict, Sequence

import torch

from uniphysio_wm.probabilistic_waveform import (
    _band_log_power,
    _emg_targets,
    _qrs_events,
    _rr_targets,
)
from uniphysio_wm.renderer_probability_calibration import (
    _binary_f1,
    _correlation_rows,
    _event_distances,
    _event_scores,
    _finite_mean,
)


def _log_psd_mae(prediction: torch.Tensor, target: torch.Tensor) -> float:
    predicted = torch.log1p(torch.fft.rfft(prediction.double(), dim=-1, norm="ortho").abs())
    targets = torch.log1p(torch.fft.rfft(target.double(), dim=-1, norm="ortho").abs())
    return float((predicted - targets).abs().mean())


def _rr_sequences(events: torch.Tensor, sample_rate: int) -> list[torch.Tensor]:
    sequences = []
    for row in events:
        positions = row.nonzero(as_tuple=False).flatten().float()
        sequences.append((positions[1:] - positions[:-1]) / sample_rate)
    return sequences


def _rr_and_hrv_metrics(
    predicted_events: torch.Tensor,
    target_events: torch.Tensor,
    sample_rate: int,
) -> Dict[str, float | int | None]:
    predicted_sequences = _rr_sequences(predicted_events, sample_rate)
    target_sequences = _rr_sequences(target_events, sample_rate)
    median_errors = []
    rmssd_errors = []
    for predicted, target in zip(predicted_sequences, target_sequences):
        if predicted.numel() and target.numel():
            median_errors.append((predicted.median() - target.median()).abs())
        if predicted.numel() >= 2 and target.numel() >= 2:
            predicted_rmssd = predicted.diff().square().mean().sqrt()
            target_rmssd = target.diff().square().mean().sqrt()
            rmssd_errors.append((predicted_rmssd - target_rmssd).abs())
    return {
        "median_rr_mae_ms": float(torch.stack(median_errors).mean() * 1000.0)
        if median_errors
        else None,
        "rmssd_mae_ms": float(torch.stack(rmssd_errors).mean() * 1000.0)
        if rmssd_errors
        else None,
        "rr_valid_records": len(median_errors),
        "rmssd_valid_records": len(rmssd_errors),
    }


def _beat_templates(
    signals: torch.Tensor,
    events: torch.Tensor,
    sample_rate: int,
) -> tuple[list[torch.Tensor], list[int]]:
    left = max(1, round(0.20 * sample_rate))
    right = max(1, round(0.40 * sample_rate))
    templates = []
    rows = []
    for row_index, (signal, event_row) in enumerate(zip(signals, events)):
        beats = []
        for position in event_row.nonzero(as_tuple=False).flatten():
            start = int(position) - left
            stop = int(position) + right
            if start >= 0 and stop <= signal.shape[-1]:
                segment = signal[start:stop]
                beats.append(segment - 0.5 * (segment[0] + segment[-1]))
        if beats:
            templates.append(torch.stack(beats).median(dim=0).values)
            rows.append(row_index)
    return templates, rows


def _beat_morphology_correlation(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int,
) -> tuple[float, int]:
    predicted_events = _qrs_events(prediction, sample_rate)
    target_events = _qrs_events(target, sample_rate)
    predicted_templates, predicted_rows = _beat_templates(
        prediction, predicted_events, sample_rate
    )
    target_templates, target_rows = _beat_templates(target, target_events, sample_rate)
    predicted_by_row = dict(zip(predicted_rows, predicted_templates))
    target_by_row = dict(zip(target_rows, target_templates))
    common = sorted(set(predicted_by_row).intersection(target_by_row))
    if not common:
        return float("nan"), 0
    predicted = torch.stack([predicted_by_row[index] for index in common])
    targets = torch.stack([target_by_row[index] for index in common])
    return _finite_mean(_correlation_rows(predicted, targets)), len(common)


def _coupling_descriptors(
    waveforms: torch.Tensor,
    sample_rate: int,
    patch_samples: int,
) -> Dict[str, torch.Tensor]:
    patch_count = waveforms.shape[-1] // patch_samples
    eeg_patches = waveforms[:, 0].reshape(len(waveforms), patch_count, patch_samples)
    eeg_centered = eeg_patches - eeg_patches.mean(dim=-1, keepdim=True)
    eeg = _band_log_power(eeg_centered, sample_rate).mean(dim=-1)
    ecg_events = _qrs_events(waveforms[:, 1], sample_rate)
    ecg = ecg_events.reshape(len(waveforms), patch_count, patch_samples).float().sum(dim=-1)
    emg_patches = waveforms[:, 2].reshape(len(waveforms), patch_count, patch_samples)
    emg = emg_patches.square().mean(dim=-1).clamp_min(1e-8).sqrt()
    return {"EEG": eeg, "ECG": ecg, "EMG": emg}


def _paired_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    selected = torch.isfinite(first) & torch.isfinite(second)
    if int(selected.sum()) < 2:
        return float("nan")
    x = first[selected].double()
    y = second[selected].double()
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
    return float((x * y).sum() / denominator.clamp_min(1e-12))


def _coupling_metrics(
    generated: torch.Tensor,
    target: torch.Tensor,
    recent: torch.Tensor,
    sample_rate: int,
    patch_samples: int,
) -> dict:
    generated_descriptors = _coupling_descriptors(generated, sample_rate, patch_samples)
    target_descriptors = _coupling_descriptors(target, sample_rate, patch_samples)
    recent_descriptors = _coupling_descriptors(recent, sample_rate, patch_samples)
    by_pair = {}
    for first, second in (("EEG", "ECG"), ("EEG", "EMG"), ("ECG", "EMG")):
        target_coupling = _correlation_rows(target_descriptors[first], target_descriptors[second])
        generated_coupling = _correlation_rows(
            generated_descriptors[first], generated_descriptors[second]
        )
        recent_coupling = _correlation_rows(recent_descriptors[first], recent_descriptors[second])
        generated_error = (generated_coupling - target_coupling).abs()
        recent_error = (recent_coupling - target_coupling).abs()
        by_pair[f"{first}-{second}"] = {
            "generated_coupling_mae": _finite_mean(generated_error),
            "recent_coupling_mae": _finite_mean(recent_error),
            "generated_target_coupling_correlation": _paired_correlation(
                generated_coupling, target_coupling
            ),
            "valid_records": int(
                (torch.isfinite(generated_error) & torch.isfinite(recent_error)).sum()
            ),
        }
    return {"by_pair": by_pair}


def comprehensive_physiology_metrics(
    generated: torch.Tensor,
    target: torch.Tensor,
    recent: torch.Tensor,
    valid: torch.Tensor,
    modalities: Sequence[str],
    sample_rate: int,
    horizons_seconds: Sequence[int],
    patch_samples: int,
) -> dict:
    modality_index = {name: index for index, name in enumerate(modalities)}
    by_horizon = {}
    amplitude = {}
    for modality, index in modality_index.items():
        selected = valid[:, index].bool()
        generated_rms = generated[selected, index].square().mean(dim=-1).sqrt().mean()
        target_rms = target[selected, index].square().mean(dim=-1).sqrt().mean()
        amplitude[modality] = {
            "generated_rms": float(generated_rms),
            "target_rms": float(target_rms),
            "generated_to_target_ratio": float(generated_rms / target_rms.clamp_min(1e-8)),
        }

    for seconds in horizons_seconds:
        samples = int(seconds) * int(sample_rate)
        patch_count = samples // int(patch_samples)
        horizon = {}

        selected = valid[:, modality_index["EEG"]].bool()
        prediction = generated[selected, modality_index["EEG"], :samples]
        truth = target[selected, modality_index["EEG"], :samples]
        baseline = recent[selected, modality_index["EEG"], :samples]
        prediction_patches = prediction.reshape(len(prediction), patch_count, patch_samples)
        target_patches = truth.reshape(len(truth), patch_count, patch_samples)
        baseline_patches = baseline.reshape(len(baseline), patch_count, patch_samples)
        prediction_band = _band_log_power(
            prediction_patches - prediction_patches.mean(dim=-1, keepdim=True), sample_rate
        )
        target_band = _band_log_power(
            target_patches - target_patches.mean(dim=-1, keepdim=True), sample_rate
        )
        baseline_band = _band_log_power(
            baseline_patches - baseline_patches.mean(dim=-1, keepdim=True), sample_rate
        )
        band_names = ("delta", "theta", "alpha", "beta")
        horizon["EEG"] = {
            "log_psd_mae": _log_psd_mae(prediction, truth),
            "recent_log_psd_mae": _log_psd_mae(baseline, truth),
            "band_log_power_mae": float((prediction_band - target_band).abs().mean()),
            "recent_band_log_power_mae": float((baseline_band - target_band).abs().mean()),
            "band_log_power_mae_by_band": {
                name: float((prediction_band[..., index] - target_band[..., index]).abs().mean())
                for index, name in enumerate(band_names)
            },
        }

        selected = valid[:, modality_index["ECG"]].bool()
        prediction = generated[selected, modality_index["ECG"], :samples]
        truth = target[selected, modality_index["ECG"], :samples]
        baseline = recent[selected, modality_index["ECG"], :samples]
        prediction_events = _qrs_events(prediction, sample_rate)
        target_events = _qrs_events(truth, sample_rate)
        baseline_events = _qrs_events(baseline, sample_rate)
        event_scores = _event_scores(prediction_events, target_events, sample_rate)
        baseline_scores = _event_scores(baseline_events, target_events, sample_rate)
        timing, chamfer = _event_distances(prediction_events, target_events)
        morphology, morphology_records = _beat_morphology_correlation(
            prediction, truth, sample_rate
        )
        baseline_morphology, baseline_morphology_records = _beat_morphology_correlation(
            baseline, truth, sample_rate
        )
        horizon["ECG"] = {
            "qrs_precision": event_scores["precision"],
            "qrs_recall": event_scores["recall"],
            "qrs_f1": event_scores["f1"],
            "recent_qrs_f1": baseline_scores["f1"],
            "qrs_target_nearest_timing_mae_ms": _finite_mean(timing) * 1000.0 / sample_rate,
            "qrs_chamfer_ms": _finite_mean(chamfer) * 1000.0 / sample_rate,
            "beat_morphology_correlation": morphology,
            "beat_morphology_valid_records": morphology_records,
            "recent_beat_morphology_correlation": baseline_morphology,
            "recent_beat_morphology_valid_records": baseline_morphology_records,
            **_rr_and_hrv_metrics(prediction_events, target_events, sample_rate),
            "recent_rr_hrv": _rr_and_hrv_metrics(
                baseline_events, target_events, sample_rate
            ),
        }

        selected = valid[:, modality_index["EMG"]].bool()
        prediction = generated[selected, modality_index["EMG"], :samples]
        truth = target[selected, modality_index["EMG"], :samples]
        baseline = recent[selected, modality_index["EMG"], :samples]
        prediction_envelope, prediction_rms, prediction_burst = _emg_targets(
            prediction, patch_samples, sample_rate
        )
        target_envelope, target_rms, target_burst = _emg_targets(
            truth, patch_samples, sample_rate
        )
        baseline_envelope, baseline_rms, baseline_burst = _emg_targets(
            baseline, patch_samples, sample_rate
        )
        horizon["EMG"] = {
            "envelope_mae": float((prediction_envelope - target_envelope).abs().mean()),
            "recent_envelope_mae": float((baseline_envelope - target_envelope).abs().mean()),
            "envelope_correlation": _finite_mean(
                _correlation_rows(prediction_envelope, target_envelope)
            ),
            "recent_envelope_correlation": _finite_mean(
                _correlation_rows(baseline_envelope, target_envelope)
            ),
            "rms_mae": float((prediction_rms - target_rms).abs().mean()),
            "recent_rms_mae": float((baseline_rms - target_rms).abs().mean()),
            "burst_f1": _binary_f1(prediction_burst, target_burst),
            "recent_burst_f1": _binary_f1(baseline_burst, target_burst),
        }

        all_selected = valid[:, [modality_index[name] for name in ("EEG", "ECG", "EMG")]].all(dim=1)
        if patch_count >= 4 and all_selected.any():
            horizon["cross_modal_coupling"] = _coupling_metrics(
                generated[all_selected, :, :samples],
                target[all_selected, :, :samples],
                recent[all_selected, :, :samples],
                sample_rate,
                patch_samples,
            )
        else:
            horizon["cross_modal_coupling"] = {
                "status": "insufficient_patches",
                "minimum_patches": 4,
                "observed_patches": patch_count,
            }
        by_horizon[str(seconds)] = horizon
    return {"by_horizon_seconds": by_horizon, "amplitude": amplitude}


def physiology_report_summary(metrics: dict) -> dict:
    improvements = {
        "EEG_log_PSD": [],
        "EEG_band_power": [],
        "ECG_QRS": [],
        "ECG_morphology": [],
        "EMG_envelope": [],
        "EMG_burst": [],
        "coupling_pairs": [],
    }
    required_fields_complete = True
    for horizon, values in metrics["by_horizon_seconds"].items():
        eeg = values["EEG"]
        ecg = values["ECG"]
        emg = values["EMG"]
        if eeg["log_psd_mae"] < eeg["recent_log_psd_mae"]:
            improvements["EEG_log_PSD"].append(horizon)
        if eeg["band_log_power_mae"] < eeg["recent_band_log_power_mae"]:
            improvements["EEG_band_power"].append(horizon)
        if ecg["qrs_f1"] > ecg["recent_qrs_f1"]:
            improvements["ECG_QRS"].append(horizon)
        if ecg["beat_morphology_correlation"] > ecg["recent_beat_morphology_correlation"]:
            improvements["ECG_morphology"].append(horizon)
        if emg["envelope_correlation"] > emg["recent_envelope_correlation"]:
            improvements["EMG_envelope"].append(horizon)
        if emg["burst_f1"] > emg["recent_burst_f1"]:
            improvements["EMG_burst"].append(horizon)
        coupling = values["cross_modal_coupling"]
        for pair, pair_values in coupling.get("by_pair", {}).items():
            if pair_values["generated_coupling_mae"] < pair_values["recent_coupling_mae"]:
                improvements["coupling_pairs"].append(f"{horizon}:{pair}")
        required_fields_complete &= all(
            key in eeg for key in ("log_psd_mae", "band_log_power_mae")
        ) and all(
            key in ecg
            for key in ("qrs_f1", "median_rr_mae_ms", "rmssd_mae_ms", "beat_morphology_correlation")
        ) and all(
            key in emg for key in ("envelope_correlation", "rms_mae", "burst_f1")
        )
    noncollapsed = all(
        0.25 <= values["generated_to_target_ratio"] <= 2.5
        for values in metrics["amplitude"].values()
    )
    return {
        "required_metric_families_complete": bool(required_fields_complete),
        "noncollapsed_amplitude": noncollapsed,
        "improved_horizons_or_pairs": improvements,
        "report_complete": bool(required_fields_complete),
        "passed": bool(required_fields_complete and noncollapsed),
    }
