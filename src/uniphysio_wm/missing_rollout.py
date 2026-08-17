from __future__ import annotations

from typing import Dict, Sequence

import torch

from .masking import random_full_biased_natural_modality_subset


def forced_history_view(
    history_signals: torch.Tensor,
    history_present: torch.Tensor,
    subset: Sequence[str],
    modalities: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    if history_signals.ndim != 4 or history_present.shape != history_signals.shape[:3]:
        raise ValueError("history tensors must be [batch, epochs, modalities, samples]")
    selected = torch.tensor(
        [modality in subset for modality in modalities],
        dtype=torch.bool,
        device=history_present.device,
    )
    if not selected.any():
        raise ValueError("forced modality subset must be nonempty")
    forced_present = history_present.bool() & selected.reshape(1, 1, -1)
    forced_signals = history_signals.masked_fill(
        ~forced_present.unsqueeze(-1), 0.0
    )
    return forced_signals, forced_present


def sampled_history_view(
    history_signals: torch.Tensor,
    history_present: torch.Tensor,
    full_modality_probability: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if history_signals.ndim != 4 or history_present.shape != history_signals.shape[:3]:
        raise ValueError("history tensors must be [batch, epochs, modalities, samples]")
    naturally_stable = history_present.bool().all(dim=1)
    selected = random_full_biased_natural_modality_subset(
        naturally_stable,
        full_modality_probability=float(full_modality_probability),
    )
    forced_present = history_present.bool() & selected.unsqueeze(1)
    forced_signals = history_signals.masked_fill(
        ~forced_present.unsqueeze(-1), 0.0
    )
    return forced_signals, forced_present, selected


def missing_rollout_summary(
    results: Dict[str, Dict[str, object]], modalities: Sequence[str]
) -> Dict[str, object]:
    full_key = "+".join(modalities)
    if full_key not in results:
        raise ValueError("missing-rollout results require the full-modality reference")
    full = results[full_key]
    full_stage = float(full["stage"]["all_horizons"]["macro_f1"])
    full_physiology = float(
        full["future_physiology"]["all_features"]["mean_normalized_mae"]
    )
    full_waveform = float(full["waveform"]["all"]["mean_standardized_mae"])
    nonfull = [key for key in results if key != full_key]
    if not nonfull:
        raise ValueError("missing-rollout results require nonfull modality subsets")
    degradation = {}
    for key in nonfull:
        values = results[key]
        stage = float(values["stage"]["all_horizons"]["macro_f1"])
        physiology = float(
            values["future_physiology"]["all_features"]["mean_normalized_mae"]
        )
        waveform = float(values["waveform"]["all"]["mean_standardized_mae"])
        degradation[key] = {
            "stage_macro_f1_drop": full_stage - stage,
            "future_physiology_mae_ratio": physiology / max(full_physiology, 1e-12),
            "waveform_mae_ratio": waveform / max(full_waveform, 1e-12),
        }
    uncertainty_response = {}
    for modality in modalities:
        absent = [
            float(results[key]["uncertainty"][modality]["mean_scale"])
            for key in nonfull
            if modality not in key.split("+")
        ]
        full_uncertainty = float(full["uncertainty"][modality]["mean_scale"])
        absent_mean = sum(absent) / len(absent)
        uncertainty_response[modality] = {
            "full_mean_scale": full_uncertainty,
            "absent_mean_scale": absent_mean,
            "ratio": absent_mean / max(full_uncertainty, 1e-12),
            "increased": absent_mean > full_uncertainty,
        }
    stage_drops = [values["stage_macro_f1_drop"] for values in degradation.values()]
    physiology_ratios = [
        values["future_physiology_mae_ratio"] for values in degradation.values()
    ]
    waveform_ratios = [values["waveform_mae_ratio"] for values in degradation.values()]
    return {
        "full_modality": {
            "stage_macro_f1": full_stage,
            "future_physiology_mae": full_physiology,
            "waveform_mae": full_waveform,
        },
        "by_subset_degradation": degradation,
        "nonfull_mean_stage_macro_f1_drop": sum(stage_drops) / len(stage_drops),
        "nonfull_worst_stage_macro_f1_drop": max(stage_drops),
        "nonfull_mean_future_physiology_mae_ratio": sum(physiology_ratios)
        / len(physiology_ratios),
        "nonfull_mean_waveform_mae_ratio": sum(waveform_ratios) / len(waveform_ratios),
        "uncertainty_response": uncertainty_response,
    }


def missing_rollout_gate_result(
    summary: Dict[str, object],
    mean_stage_drop_limit: float = 0.10,
    worst_stage_drop_limit: float = 0.20,
    physiology_mae_ratio_limit: float = 1.25,
    waveform_mae_ratio_limit: float = 1.25,
    uncertainty_modalities_required: int = 2,
) -> Dict[str, object]:
    uncertainty_increased = [
        modality
        for modality, values in summary["uncertainty_response"].items()
        if bool(values["increased"])
    ]
    result = {
        "mean_stage_retained": float(summary["nonfull_mean_stage_macro_f1_drop"])
        <= float(mean_stage_drop_limit),
        "worst_stage_retained": float(summary["nonfull_worst_stage_macro_f1_drop"])
        <= float(worst_stage_drop_limit),
        "future_physiology_retained": float(
            summary["nonfull_mean_future_physiology_mae_ratio"]
        )
        <= float(physiology_mae_ratio_limit),
        "waveform_retained": float(summary["nonfull_mean_waveform_mae_ratio"])
        <= float(waveform_mae_ratio_limit),
        "uncertainty_increased_modalities": uncertainty_increased,
        "uncertainty_response_passed": len(uncertainty_increased)
        >= int(uncertainty_modalities_required),
    }
    result["passed"] = bool(
        result["mean_stage_retained"]
        and result["worst_stage_retained"]
        and result["future_physiology_retained"]
        and result["waveform_retained"]
        and result["uncertainty_response_passed"]
    )
    return result
