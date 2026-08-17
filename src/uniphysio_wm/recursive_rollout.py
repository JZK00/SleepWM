from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn.functional as F


def _latent_metrics(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    return {
        "smooth_l1": float(F.smooth_l1_loss(prediction, target)),
        "cosine_similarity": float(
            F.cosine_similarity(prediction, target, dim=-1).mean()
        ),
    }


def recursive_latent_metrics(
    prediction: torch.Tensor,
    direct_prediction: torch.Tensor,
    target: torch.Tensor,
    correction: torch.Tensor,
    horizons: Sequence[int],
) -> Dict[str, object]:
    if not (
        prediction.shape
        == direct_prediction.shape
        == target.shape
        == correction.shape
    ):
        raise ValueError("recursive latent tensors must have identical shapes")
    direct = _latent_metrics(direct_prediction, target)
    hybrid = _latent_metrics(prediction, target)
    direct_loss = float(direct["smooth_l1"])
    hybrid_loss = float(hybrid["smooth_l1"])
    return {
        "direct": direct,
        "hybrid": hybrid,
        "relative_smooth_l1_improvement": (direct_loss - hybrid_loss)
        / max(direct_loss, 1e-12),
        "correction_rms": float(correction.square().mean().sqrt()),
        "by_horizon": {
            str(horizon): {
                "direct": _latent_metrics(
                    direct_prediction[:, index], target[:, index]
                ),
                "hybrid": _latent_metrics(prediction[:, index], target[:, index]),
                "correction_rms": float(
                    correction[:, index].square().mean().sqrt()
                ),
            }
            for index, horizon in enumerate(horizons)
        },
    }


def state_alignment_metrics(
    direct_state: torch.Tensor,
    corrected_state: torch.Tensor,
    teacher_state: torch.Tensor,
) -> Dict[str, object]:
    if not direct_state.shape == corrected_state.shape == teacher_state.shape:
        raise ValueError("observation alignment tensors must have identical shapes")
    direct = _latent_metrics(direct_state, teacher_state)
    corrected = _latent_metrics(corrected_state, teacher_state)
    direct_loss = float(direct["smooth_l1"])
    corrected_loss = float(corrected["smooth_l1"])
    return {
        "direct": direct,
        "corrected": corrected,
        "relative_smooth_l1_improvement": (direct_loss - corrected_loss)
        / max(direct_loss, 1e-12),
    }


def observation_repair_gate_result(
    results: Dict[str, object],
    modalities: Sequence[str],
    *,
    minimum_alignment_improvement: float = 0.05,
    maximum_full_stage_drop: float = 0.005,
    minimum_nonfull_stage_delta: float = 0.0,
    minimum_single_modality_stage_delta: float = -0.005,
    maximum_waveform_ratio: float = 1.02,
) -> Dict[str, object]:
    full_name = "+".join(modalities)
    nonfull_names = [name for name in results if name != full_name]
    full = results[full_name]
    full_comparison = full["recursive_comparison"]
    hybrid_full_stage = float(full["stage"]["all_horizons"]["macro_f1"])
    direct_full_stage = float(
        full_comparison["direct_stage"]["all_horizons"]["macro_f1"]
    )
    hybrid_nonfull_stage = [
        float(results[name]["stage"]["all_horizons"]["macro_f1"])
        for name in nonfull_names
    ]
    direct_nonfull_stage = [
        float(
            results[name]["recursive_comparison"]["direct_stage"]["all_horizons"][
                "macro_f1"
            ]
        )
        for name in nonfull_names
    ]
    hybrid_nonfull_physiology = [
        float(
            results[name]["future_physiology"]["all_features"][
                "mean_normalized_mae"
            ]
        )
        for name in nonfull_names
    ]
    direct_nonfull_physiology = [
        float(
            results[name]["recursive_comparison"]["direct_future_physiology"][
                "all_features"
            ]["mean_normalized_mae"]
        )
        for name in nonfull_names
    ]
    alignment_improvements = [
        float(
            results[name]["recursive_comparison"]["observation_state"][
                "relative_smooth_l1_improvement"
            ]
        )
        for name in nonfull_names
    ]
    single_modality_deltas = {
        modality: float(results[modality]["stage"]["all_horizons"]["macro_f1"])
        - float(
            results[modality]["recursive_comparison"]["direct_stage"][
                "all_horizons"
            ]["macro_f1"]
        )
        for modality in modalities
    }
    mean_hybrid_stage = sum(hybrid_nonfull_stage) / len(hybrid_nonfull_stage)
    mean_direct_stage = sum(direct_nonfull_stage) / len(direct_nonfull_stage)
    mean_hybrid_physiology = sum(hybrid_nonfull_physiology) / len(
        hybrid_nonfull_physiology
    )
    mean_direct_physiology = sum(direct_nonfull_physiology) / len(
        direct_nonfull_physiology
    )
    mean_alignment = sum(alignment_improvements) / len(alignment_improvements)
    hybrid_waveform = float(full["waveform"]["all"]["mean_standardized_mae"])
    direct_waveform = float(
        full_comparison["direct_waveform"]["all"]["mean_standardized_mae"]
    )
    result = {
        "observation_alignment_improved": mean_alignment
        >= float(minimum_alignment_improvement),
        "full_stage_retained": direct_full_stage - hybrid_full_stage
        <= float(maximum_full_stage_drop),
        "nonfull_stage_improved": mean_hybrid_stage - mean_direct_stage
        >= float(minimum_nonfull_stage_delta),
        "single_modality_stage_retained": all(
            delta >= float(minimum_single_modality_stage_delta)
            for delta in single_modality_deltas.values()
        ),
        "nonfull_physiology_improved": mean_hybrid_physiology
        < mean_direct_physiology,
        "waveform_retained": hybrid_waveform
        <= float(maximum_waveform_ratio) * direct_waveform,
        "mean_observation_alignment_improvement": mean_alignment,
        "hybrid_nonfull_stage_macro_f1": mean_hybrid_stage,
        "direct_nonfull_stage_macro_f1": mean_direct_stage,
        "single_modality_stage_deltas": single_modality_deltas,
        "hybrid_nonfull_physiology_mae": mean_hybrid_physiology,
        "direct_nonfull_physiology_mae": mean_direct_physiology,
        "hybrid_full_stage_macro_f1": hybrid_full_stage,
        "direct_full_stage_macro_f1": direct_full_stage,
        "hybrid_waveform_mae": hybrid_waveform,
        "direct_waveform_mae": direct_waveform,
    }
    result["passed"] = all(
        bool(result[name])
        for name in (
            "observation_alignment_improved",
            "full_stage_retained",
            "nonfull_stage_improved",
            "single_modality_stage_retained",
            "nonfull_physiology_improved",
            "waveform_retained",
        )
    )
    return result


def recursive_rollout_gate_result(
    full_result: Dict[str, object],
    missing_summary: Dict[str, object],
    *,
    minimum_latent_improvement: float = 0.02,
    maximum_stage_drop: float = 0.01,
    minimum_missing_mean_stage: float = 0.4646,
    physiology_horizons_required: int = 2,
    maximum_waveform_ratio: float = 1.02,
    minimum_correction_rms: float = 1e-4,
) -> Dict[str, object]:
    comparison = full_result["recursive_comparison"]
    latent = comparison["latent"]
    hybrid_stage = float(full_result["stage"]["all_horizons"]["macro_f1"])
    direct_stage = float(comparison["direct_stage"]["all_horizons"]["macro_f1"])
    full_stage = float(missing_summary["full_modality"]["stage_macro_f1"])
    missing_mean_stage = full_stage - float(
        missing_summary["nonfull_mean_stage_macro_f1_drop"]
    )
    hybrid_physiology = full_result["future_physiology"]
    direct_physiology = comparison["direct_future_physiology"]
    improved_horizons = [
        str(horizon)
        for horizon, values in hybrid_physiology["by_horizon"].items()
        if float(values["mean_normalized_mae"])
        < float(direct_physiology["by_horizon"][str(horizon)]["mean_normalized_mae"])
    ]
    hybrid_waveform = float(full_result["waveform"]["all"]["mean_standardized_mae"])
    direct_waveform = float(
        comparison["direct_waveform"]["all"]["mean_standardized_mae"]
    )
    result = {
        "latent_improved": float(latent["relative_smooth_l1_improvement"])
        >= float(minimum_latent_improvement),
        "stage_retained": direct_stage - hybrid_stage <= float(maximum_stage_drop),
        "missing_stage_retained": missing_mean_stage
        >= float(minimum_missing_mean_stage),
        "physiology_improved_horizons": improved_horizons,
        "physiology_improved": len(improved_horizons)
        >= int(physiology_horizons_required),
        "waveform_retained": hybrid_waveform
        <= float(maximum_waveform_ratio) * direct_waveform,
        "recursive_path_active": float(latent["correction_rms"])
        >= float(minimum_correction_rms),
        "hybrid_stage_macro_f1": hybrid_stage,
        "direct_stage_macro_f1": direct_stage,
        "missing_mean_stage_macro_f1": missing_mean_stage,
        "hybrid_waveform_mae": hybrid_waveform,
        "direct_waveform_mae": direct_waveform,
    }
    result["passed"] = all(
        bool(result[name])
        for name in (
            "latent_improved",
            "stage_retained",
            "missing_stage_retained",
            "physiology_improved",
            "waveform_retained",
            "recursive_path_active",
        )
    )
    return result
