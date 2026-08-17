from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F

from .probabilistic_waveform import _event_f1, _qrs_events, _rr_distribution_loss


def _patch_event_targets(
    events: torch.Tensor, patch_samples: int
) -> tuple[torch.Tensor, torch.Tensor]:
    patches = events.reshape(events.shape[0], -1, int(patch_samples))
    presence = patches.any(dim=-1)
    local_positions = torch.arange(
        patch_samples, device=events.device, dtype=torch.float32
    )
    counts = patches.sum(dim=-1).clamp_min(1).to(torch.float32)
    event_position = (
        patches.to(torch.float32) * local_positions.view(1, 1, -1)
    ).sum(dim=-1) / counts
    offset = event_position - 0.5 * (int(patch_samples) - 1)
    return presence, offset


def _rr_sequence_targets(
    events: torch.Tensor, patch_samples: int, sample_rate: int
) -> tuple[torch.Tensor, torch.Tensor]:
    patch_count = events.shape[-1] // int(patch_samples)
    targets = torch.zeros(
        events.shape[0], patch_count, device=events.device, dtype=torch.float32
    )
    valid = torch.zeros_like(targets, dtype=torch.bool)
    patch_centers = (
        (torch.arange(patch_count, device=events.device, dtype=torch.float32) + 0.5)
        * int(patch_samples)
    )
    for row_index, row in enumerate(events):
        positions = row.nonzero(as_tuple=False).flatten().to(torch.float32)
        if len(positions) < 2:
            continue
        intervals = (positions[1:] - positions[:-1]) / int(sample_rate)
        interval_centers = 0.5 * (positions[1:] + positions[:-1])
        nearest = (patch_centers[:, None] - interval_centers[None, :]).abs().argmin(
            dim=-1
        )
        targets[row_index] = intervals[nearest]
        valid[row_index] = True
    return targets, valid


def _predicted_event_geometry(
    prediction: Dict[str, torch.Tensor], patch_samples: int
) -> tuple[torch.Tensor, torch.Tensor]:
    qrs_logits = prediction["qrs_logits"].reshape(
        prediction["qrs_logits"].shape[0], -1, int(patch_samples)
    )
    local_positions = torch.arange(
        patch_samples, device=qrs_logits.device, dtype=qrs_logits.dtype
    )
    local_centers = ((qrs_logits / 0.25).softmax(dim=-1) * local_positions).sum(
        dim=-1
    )
    patch_starts = int(patch_samples) * torch.arange(
        qrs_logits.shape[1], device=qrs_logits.device, dtype=qrs_logits.dtype
    )
    centers = (
        patch_starts.unsqueeze(0)
        + local_centers
        + prediction["event_offset_samples"]
    )
    weights = prediction["event_hazard_logits"].sigmoid()
    return centers, weights


def _event_slot_targets(
    events: torch.Tensor,
    recent_events: torch.Tensor,
    slot_count: int,
    sample_rate: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    presence = torch.zeros(
        events.shape[0], slot_count, device=events.device, dtype=torch.bool
    )
    centers = torch.zeros(
        events.shape[0], slot_count, device=events.device, dtype=torch.float32
    )
    rr_target = torch.zeros_like(centers)
    rr_valid = torch.zeros_like(presence)
    for row_index, (future_row, recent_row) in enumerate(
        zip(events, recent_events)
    ):
        future_positions = future_row.nonzero(as_tuple=False).flatten().to(torch.float32)
        count = min(len(future_positions), int(slot_count))
        if count == 0:
            continue
        presence[row_index, :count] = True
        centers[row_index, :count] = future_positions[:count]
        recent_positions = recent_row.nonzero(as_tuple=False).flatten()
        if len(recent_positions):
            elapsed = recent_row.shape[-1] - 1 - int(recent_positions[-1])
            rr_target[row_index, 0] = (
                future_positions[0] + elapsed
            ) / int(sample_rate)
            rr_valid[row_index, 0] = True
        if count >= 2:
            rr_target[row_index, 1:count] = (
                future_positions[1:count] - future_positions[: count - 1]
            ) / int(sample_rate)
            rr_valid[row_index, 1:count] = True
    return presence, centers, rr_target, rr_valid


def _recursive_event_geometry(
    prediction: Dict[str, torch.Tensor],
    recent: torch.Tensor,
    sample_rate: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rr_logits = prediction["rr_logits"]
    rr_centers = torch.linspace(
        0.25,
        2.0,
        rr_logits.shape[-1],
        device=rr_logits.device,
        dtype=rr_logits.dtype,
    )
    rr_samples = (rr_logits.softmax(dim=-1) * rr_centers).sum(dim=-1) * int(
        sample_rate
    )
    recent_events = _qrs_events(recent, sample_rate)
    elapsed = []
    recent_rr = []
    for row_index, row in enumerate(recent_events):
        positions = row.nonzero(as_tuple=False).flatten()
        if len(positions):
            elapsed.append(
                recent.new_tensor(recent.shape[-1] - 1 - int(positions[-1]))
            )
            if len(positions) >= 2:
                recent_rr.append(
                    (positions[1:] - positions[:-1]).float().median()
                )
            else:
                recent_rr.append(rr_samples[row_index, 0].detach())
        else:
            elapsed.append(rr_samples[row_index, 0].detach())
            recent_rr.append(rr_samples[row_index, 0].detach())
    elapsed_samples = torch.stack(elapsed)
    if "rr_residual_gate_logits" in prediction:
        gate = prediction["rr_residual_gate_logits"].sigmoid()
        recent_rr_samples = torch.stack(recent_rr).unsqueeze(-1)
        rr_samples = recent_rr_samples + gate * (rr_samples - recent_rr_samples)
    base_centers = rr_samples.cumsum(dim=-1) - elapsed_samples.unsqueeze(-1)
    search_centers = base_centers + prediction["event_offset_samples"]
    qrs_logits = prediction["qrs_logits"]
    sample_positions = torch.arange(
        qrs_logits.shape[-1], device=qrs_logits.device, dtype=qrs_logits.dtype
    )
    tolerance = max(1.0, 0.12 * int(sample_rate))
    scores = qrs_logits.unsqueeze(1) / 0.5 - 0.5 * (
        (sample_positions.view(1, 1, -1) - search_centers.unsqueeze(-1))
        / tolerance
    ).square()
    centers = (scores.softmax(dim=-1) * sample_positions).sum(dim=-1)
    return (
        centers,
        prediction["event_hazard_logits"].sigmoid(),
        base_centers,
        rr_samples / int(sample_rate),
    )


def _soft_chamfer_loss(
    centers: torch.Tensor,
    weights: torch.Tensor,
    target_events: torch.Tensor,
    sample_rate: int,
    horizon_samples: Optional[int] = None,
    late_event_weight: float = 1.0,
) -> torch.Tensor:
    losses = []
    temperature = max(1.0, 0.04 * int(sample_rate))
    for row_centers, row_weights, row_events in zip(
        centers, weights, target_events
    ):
        targets = row_events.nonzero(as_tuple=False).flatten().to(row_centers.dtype)
        if len(targets) == 0:
            losses.append(row_weights.mean())
            continue
        distances = (row_centers[:, None] - targets[None, :]).abs()
        if horizon_samples is not None and float(late_event_weight) > 1.0:
            temporal_weights = 1.0 + (float(late_event_weight) - 1.0) * (
                row_centers.clamp(0, int(horizon_samples) - 1)
                / max(1.0, float(int(horizon_samples) - 1))
            )
            target_weights = 1.0 + (float(late_event_weight) - 1.0) * (
                targets / max(1.0, float(int(horizon_samples) - 1))
            )
        else:
            temporal_weights = torch.ones_like(row_weights)
            target_weights = torch.ones_like(targets)
        normalized_weights = row_weights * temporal_weights
        normalized_weights = normalized_weights / normalized_weights.sum().clamp_min(1e-6)
        predicted_to_target = (
            normalized_weights * distances.min(dim=-1).values
        ).sum()
        target_to_prediction = -temperature * torch.logsumexp(
            normalized_weights.clamp_min(1e-8).log().unsqueeze(-1)
            - distances / temperature,
            dim=0,
        )
        losses.append(
            (
                predicted_to_target
                + (target_to_prediction * target_weights).sum()
                / target_weights.sum().clamp_min(1e-6)
            )
            / (2.0 * int(sample_rate))
        )
    return torch.stack(losses).mean()


def _beat_morphology_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_events: torch.Tensor,
    sample_rate: int,
) -> torch.Tensor:
    left = max(1, round(0.12 * int(sample_rate)))
    right = max(1, round(0.20 * int(sample_rate)))
    predicted_beats = []
    target_beats = []
    for row_prediction, row_target, row_events in zip(
        prediction, target, target_events
    ):
        for position in row_events.nonzero(as_tuple=False).flatten():
            start = int(position) - left
            stop = int(position) + right
            if start >= 0 and stop <= row_target.shape[-1]:
                predicted_beats.append(row_prediction[start:stop])
                target_beats.append(row_target[start:stop])
    if not predicted_beats:
        return prediction.sum() * 0.0
    return F.smooth_l1_loss(
        torch.stack(predicted_beats), torch.stack(target_beats)
    )


def shift_aware_shape_time_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int,
    maximum_shift_seconds: float = 0.12,
    temperature: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DILATE-inspired local shape matching with an explicit timing penalty."""
    if prediction.ndim != 2 or target.shape != prediction.shape:
        raise ValueError("ECG prediction and target must be [batch, samples]")
    kernel = max(1, round(0.08 * int(sample_rate)))
    predicted_envelope = F.avg_pool1d(
        prediction.diff(dim=-1).abs().unsqueeze(1),
        kernel_size=kernel,
        stride=1,
        padding=kernel // 2,
    ).squeeze(1)[..., : prediction.shape[-1] - 1]
    target_envelope = F.avg_pool1d(
        target.diff(dim=-1).abs().unsqueeze(1),
        kernel_size=kernel,
        stride=1,
        padding=kernel // 2,
    ).squeeze(1)[..., : target.shape[-1] - 1]
    maximum_shift = max(1, round(float(maximum_shift_seconds) * int(sample_rate)))
    shifts = torch.arange(
        -maximum_shift,
        maximum_shift + 1,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    costs = []
    for shift in range(-maximum_shift, maximum_shift + 1):
        if shift > 0:
            predicted_values = predicted_envelope[:, shift:]
            target_values = target_envelope[:, :-shift]
        elif shift < 0:
            predicted_values = predicted_envelope[:, :shift]
            target_values = target_envelope[:, -shift:]
        else:
            predicted_values = predicted_envelope
            target_values = target_envelope
        costs.append(
            F.smooth_l1_loss(
                predicted_values, target_values, reduction="none"
            ).mean(dim=-1)
        )
    costs = torch.stack(costs, dim=-1)
    scale = costs.detach().median(dim=-1, keepdim=True).values.clamp_min(1e-4)
    assignment = torch.softmax(
        -costs / (float(temperature) * scale), dim=-1
    )
    shape_loss = (assignment * costs).sum(dim=-1).mean()
    timing_loss = (
        assignment
        * (shifts.abs() / max(1.0, float(maximum_shift))).unsqueeze(0)
    ).sum(dim=-1).mean()
    return shape_loss, timing_loss


def ecg_event_alignment_loss(
    prediction: Dict[str, torch.Tensor],
    predicted_waveform: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    modalities: Sequence[str],
    sample_rate: int,
    horizons_seconds: Sequence[int],
    patch_samples: int,
    recent_waveform: Optional[torch.Tensor] = None,
    recursive_events: bool = False,
    hazard_weight: float = 0.5,
    offset_weight: float = 0.25,
    chamfer_weight: float = 1.0,
    rr_sequence_weight: float = 0.5,
    morphology_weight: float = 0.25,
    shift_shape_weight: float = 0.0,
    shift_time_weight: float = 0.0,
    shift_max_seconds: float = 0.12,
    shift_temperature: float = 0.05,
    horizon_weights: Optional[Sequence[float]] = None,
    late_event_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    ecg_index = tuple(modalities).index("ECG")
    selected = valid[:, ecg_index]
    components = {
        "hazard_loss": [],
        "offset_loss": [],
        "chamfer_loss": [],
        "rr_sequence_loss": [],
        "morphology_loss": [],
        "shift_shape_loss": [],
        "shift_time_loss": [],
    }
    if horizon_weights is None:
        horizon_loss_weights = [1.0] * len(horizons_seconds)
    else:
        horizon_loss_weights = [float(value) for value in horizon_weights]
        if len(horizon_loss_weights) != len(horizons_seconds):
            raise ValueError("horizon_weights must match horizons_seconds")
        if min(horizon_loss_weights) <= 0.0:
            raise ValueError("horizon_weights must be positive")
    for seconds in horizons_seconds:
        samples = int(seconds) * int(sample_rate)
        patch_count = samples // int(patch_samples)
        target_waveform = target[selected, ecg_index, :samples]
        generated_waveform = predicted_waveform[selected, ecg_index, :samples]
        events = _qrs_events(target_waveform, sample_rate)
        if recursive_events:
            if recent_waveform is None:
                raise ValueError("recursive ECG alignment requires recent waveform")
            recent_ecg = recent_waveform[selected, ecg_index]
            recent_events = _qrs_events(recent_ecg, sample_rate)
            presence, target_centers, rr_target, rr_valid = _event_slot_targets(
                events, recent_events, patch_count, sample_rate
            )
        else:
            presence, target_offset = _patch_event_targets(events, patch_samples)
        hazard_logits = prediction["event_hazard_logits"][selected, :patch_count]
        positives = presence.sum().clamp_min(1.0)
        negatives = presence.numel() - positives
        positive_weight = (negatives / positives).clamp(1.0, 8.0)
        slot_weights = torch.linspace(
            1.0,
            float(late_event_weight),
            patch_count,
            device=hazard_logits.device,
            dtype=hazard_logits.dtype,
        )
        hazard_values = F.binary_cross_entropy_with_logits(
            hazard_logits,
            presence.to(hazard_logits.dtype),
            pos_weight=positive_weight,
            reduction="none",
        )
        components["hazard_loss"].append(
            (hazard_values * slot_weights.unsqueeze(0)).sum()
            / (slot_weights.sum() * hazard_values.shape[0]).clamp_min(1e-6)
        )
        horizon_prediction = {
            name: value[selected, :patch_count]
            if name != "qrs_logits"
            else value[selected, :samples]
            for name, value in prediction.items()
        }
        if recursive_events:
            centers, weights, _, effective_rr = _recursive_event_geometry(
                horizon_prediction, recent_ecg, sample_rate
            )
            if presence.any():
                offset_values = F.smooth_l1_loss(
                    centers[presence] / int(sample_rate),
                    target_centers[presence] / int(sample_rate),
                    reduction="none",
                )
                offset_weights = slot_weights.unsqueeze(0).expand_as(presence)[presence]
                offset_loss = (offset_values * offset_weights).sum() / offset_weights.sum()
            else:
                offset_loss = centers.sum() * 0.0
            components["offset_loss"].append(offset_loss)
        else:
            predicted_offset = prediction["event_offset_samples"][selected, :patch_count]
            components["offset_loss"].append(
                F.smooth_l1_loss(
                    predicted_offset[presence] / int(patch_samples),
                    target_offset[presence] / int(patch_samples),
                )
                if presence.any()
                else predicted_offset.sum() * 0.0
            )
            centers, weights = _predicted_event_geometry(
                horizon_prediction, patch_samples
            )
        components["chamfer_loss"].append(
            _soft_chamfer_loss(
                centers,
                weights,
                events,
                sample_rate,
                horizon_samples=samples,
                late_event_weight=late_event_weight,
            )
        )
        if not recursive_events:
            rr_target, rr_valid = _rr_sequence_targets(
                events, patch_samples, sample_rate
            )
        rr_logits = prediction["rr_logits"][selected, :patch_count]
        if recursive_events and "rr_residual_gate_logits" in horizon_prediction:
            if rr_valid.any():
                rr_values = F.smooth_l1_loss(
                    effective_rr[rr_valid] / 0.1,
                    rr_target[rr_valid] / 0.1,
                    reduction="none",
                )
                rr_weights = slot_weights.unsqueeze(0).expand_as(rr_valid)[rr_valid]
                rr_sequence_loss = (rr_values * rr_weights).sum() / rr_weights.sum()
            else:
                rr_sequence_loss = rr_logits.sum() * 0.0
        else:
            rr_sequence_loss = (
                _rr_distribution_loss(rr_logits[rr_valid], rr_target[rr_valid])
                if rr_valid.any()
                else rr_logits.sum() * 0.0
            )
        components["rr_sequence_loss"].append(rr_sequence_loss)
        components["morphology_loss"].append(
            _beat_morphology_loss(
                generated_waveform, target_waveform, events, sample_rate
            )
        )
        shape_loss, time_loss = shift_aware_shape_time_loss(
            generated_waveform,
            target_waveform,
            sample_rate,
            maximum_shift_seconds=shift_max_seconds,
            temperature=shift_temperature,
        )
        components["shift_shape_loss"].append(shape_loss)
        components["shift_time_loss"].append(time_loss)
    horizon_weight_tensor = target.new_tensor(horizon_loss_weights)
    reduced = {
        name: (torch.stack(values) * horizon_weight_tensor).sum()
        / horizon_weight_tensor.sum()
        for name, values in components.items()
    }
    total = (
        float(hazard_weight) * reduced["hazard_loss"]
        + float(offset_weight) * reduced["offset_loss"]
        + float(chamfer_weight) * reduced["chamfer_loss"]
        + float(rr_sequence_weight) * reduced["rr_sequence_loss"]
        + float(morphology_weight) * reduced["morphology_loss"]
        + float(shift_shape_weight) * reduced["shift_shape_loss"]
        + float(shift_time_weight) * reduced["shift_time_loss"]
    )
    return {"loss": total, **reduced}


def _nearest_timing_mae_ms(
    predicted_events: torch.Tensor,
    target_events: torch.Tensor,
    sample_rate: int,
) -> float:
    errors = []
    for predicted, target in zip(predicted_events, target_events):
        predicted_positions = predicted.nonzero(as_tuple=False).flatten().float()
        target_positions = target.nonzero(as_tuple=False).flatten().float()
        if len(predicted_positions) == 0 or len(target_positions) == 0:
            continue
        distances = (predicted_positions[:, None] - target_positions[None, :]).abs()
        errors.append(
            0.5
            * (
                distances.min(dim=1).values.mean()
                + distances.min(dim=0).values.mean()
            )
        )
    if not errors:
        return float("inf")
    return float(torch.stack(errors).mean() * 1000.0 / int(sample_rate))


def ecg_event_alignment_metrics(
    prediction: Dict[str, torch.Tensor],
    target: torch.Tensor,
    valid: torch.Tensor,
    modalities: Sequence[str],
    sample_rate: int,
    horizons_seconds: Sequence[int],
    patch_samples: int,
    recent_waveform: Optional[torch.Tensor] = None,
    recursive_events: bool = False,
) -> Dict[str, object]:
    probability = {name: value.detach().cpu().float() for name, value in prediction.items()}
    targets = target.detach().cpu().float()
    mask = valid.detach().cpu().bool()
    ecg_index = tuple(modalities).index("ECG")
    selected = mask[:, ecg_index]
    by_horizon = {}
    for seconds in horizons_seconds:
        samples = int(seconds) * int(sample_rate)
        patch_count = samples // int(patch_samples)
        events = _qrs_events(targets[selected, ecg_index, :samples], sample_rate)
        horizon_prediction = {
            name: value[selected, :patch_count]
            if name != "qrs_logits"
            else value[selected, :samples]
            for name, value in probability.items()
        }
        if recursive_events:
            if recent_waveform is None:
                raise ValueError("recursive ECG metrics require recent waveform")
            recent_ecg = recent_waveform.detach().cpu().float()[selected, ecg_index]
            centers, weights, base_centers, effective_rr = _recursive_event_geometry(
                horizon_prediction, recent_ecg, sample_rate
            )
        else:
            centers, weights = _predicted_event_geometry(
                horizon_prediction, patch_samples
            )
            base_centers = centers
        predicted_events = torch.zeros_like(events)
        in_range = (base_centers >= 0) & (base_centers < samples)
        active = (
            in_range
            if recursive_events and "rr_residual_gate_logits" in horizon_prediction
            else (weights >= 0.5) & in_range
        )
        rounded_centers = centers.round().long().clamp(0, samples - 1)
        predicted_events.scatter_(1, rounded_centers, active)
        if recursive_events:
            recent_events = _qrs_events(recent_ecg, sample_rate)
            presence, target_centers, rr_target, rr_valid = _event_slot_targets(
                events, recent_events, patch_count, sample_rate
            )
        else:
            rr_target, rr_valid = _rr_sequence_targets(
                events, patch_samples, sample_rate
            )
        rr_logits = horizon_prediction["rr_logits"]
        rr_centers = torch.linspace(
            0.25, 2.0, rr_logits.shape[-1], dtype=rr_logits.dtype
        )
        rr_mean = (
            effective_rr
            if recursive_events and "rr_residual_gate_logits" in horizon_prediction
            else (rr_logits.softmax(dim=-1) * rr_centers).sum(dim=-1)
        )
        if recursive_events:
            timing_error = (centers[presence] - target_centers[presence]).abs()
        else:
            presence, target_offset = _patch_event_targets(events, patch_samples)
            predicted_offset = horizon_prediction["event_offset_samples"]
            timing_error = (predicted_offset[presence] - target_offset[presence]).abs()
        by_horizon[str(seconds)] = {
            "head_qrs_event_f1": _event_f1(
                predicted_events, events, sample_rate
            ),
            "head_qrs_timing_mae_ms": _nearest_timing_mae_ms(
                predicted_events, events, sample_rate
            ),
            "head_rr_sequence_mae_ms": float(
                (rr_mean[rr_valid] - rr_target[rr_valid]).abs().mean() * 1000.0
            )
            if rr_valid.any()
            else float("inf"),
            "head_offset_mae_ms": float(
                timing_error.mean() * 1000.0 / int(sample_rate)
            )
            if presence.any()
            else float("inf"),
            "predicted_events_per_record": float(active.sum(dim=-1).float().mean()),
            "target_events_per_record": float(events.sum(dim=-1).float().mean()),
        }
    return {"by_horizon_seconds": by_horizon}


def event_aligned_gate_result(
    validation: Dict[str, object],
    w5_validation: Dict[str, object],
    event_metrics: Dict[str, object],
    maximum_w5_waveform_ratio: float = 1.10,
) -> Dict[str, object]:
    current_waveform = validation["model_waveform"]
    baseline_waveform = w5_validation["model_waveform"]
    qrs_improved = []
    rr_improved = []
    active_event_heads = []
    for horizon, values in current_waveform["by_horizon_seconds"].items():
        current_ecg = values["by_modality"]["ECG"]
        baseline_ecg = baseline_waveform["by_horizon_seconds"][horizon][
            "by_modality"
        ]["ECG"]
        if float(current_ecg["qrs_event_f1"]) > float(
            baseline_ecg["qrs_event_f1"]
        ):
            qrs_improved.append(horizon)
        if float(current_ecg["rr_interval_mae_ms"]) < float(
            baseline_ecg["rr_interval_mae_ms"]
        ):
            rr_improved.append(horizon)
        if float(
            event_metrics["by_horizon_seconds"][horizon]["head_qrs_event_f1"]
        ) > 0.0:
            active_event_heads.append(horizon)
    current_probability = validation["probability"]["by_horizon_seconds"]
    baseline_probability = w5_validation["probability"]["by_horizon_seconds"]
    eeg_retained = []
    emg_retained = []
    for horizon, values in current_probability.items():
        baseline = baseline_probability[horizon]
        if float(values["EEG"]["spectral_mean_mae"]) <= 1.02 * float(
            baseline["EEG"]["spectral_mean_mae"]
        ):
            eeg_retained.append(horizon)
        if float(values["EMG"]["envelope_mean_mae"]) <= 1.02 * float(
            baseline["EMG"]["envelope_mean_mae"]
        ):
            emg_retained.append(horizon)
    current_mae = float(current_waveform["all"]["mean_standardized_mae"])
    baseline_mae = float(baseline_waveform["all"]["mean_standardized_mae"])
    amplitude_ratios = {
        modality: float(values["generated_to_target_ratio"])
        for modality, values in validation["amplitude"].items()
    }
    result = {
        "qrs_waveform_f1_improved_horizons": qrs_improved,
        "rr_waveform_mae_improved_horizons": rr_improved,
        "active_event_head_horizons": active_event_heads,
        "eeg_probability_retained_horizons": eeg_retained,
        "emg_probability_retained_horizons": emg_retained,
        "waveform_mae": current_mae,
        "w5_waveform_mae": baseline_mae,
        "waveform_ratio_to_w5": current_mae / max(baseline_mae, 1e-12),
        "maximum_w5_waveform_ratio": float(maximum_w5_waveform_ratio),
        "amplitude_ratios": amplitude_ratios,
        "noncollapsed_amplitude": all(
            0.25 <= value <= 2.5 for value in amplitude_ratios.values()
        ),
    }
    result["passed"] = bool(
        len(qrs_improved) >= 2
        and len(rr_improved) >= 2
        and len(active_event_heads) >= 2
        and len(eeg_retained) >= 2
        and len(emg_retained) >= 2
        and result["waveform_ratio_to_w5"] <= float(maximum_w5_waveform_ratio)
        and result["noncollapsed_amplitude"]
    )
    return result
