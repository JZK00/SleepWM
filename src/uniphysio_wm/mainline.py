from __future__ import annotations

from typing import Dict, Optional, Type

import torch

from .models import (
    GatedRecursiveTaskAwareWaveformWorldModel,
    PhysiologyFrontendEncoder,
    observation_config,
)


WAVEFORM_PREFIX = "waveform_decoder."
OBSERVATION_PREFIXES = (
    "observation_state_adapters.",
    "observation_reliability_heads.",
    "task_stage_residual_heads.",
    "task_physiology_residual_heads.",
)
MISSING_CONDITION_PREFIXES = (
    "waveform_decoder.missing_memory_tokens.",
    "waveform_decoder.missing_context_adapters.",
    "waveform_decoder.missing_physiology_calibration_heads.",
    "waveform_decoder.missing_emg_log_rms_bias",
)


def build_mainline_model(
    config: dict,
    model_class: Type[GatedRecursiveTaskAwareWaveformWorldModel] = (
        GatedRecursiveTaskAwareWaveformWorldModel
    ),
    extra_model_kwargs: Optional[dict] = None,
) -> GatedRecursiveTaskAwareWaveformWorldModel:
    physiology = config["physiology"]
    waveform = config["waveform"]
    observation = config["observation_repair"]
    recursive = config["recursive"]
    group_sizes = {
        group: len(names) for group, names in physiology["feature_groups"].items()
    }
    model_kwargs = dict(extra_model_kwargs or {})
    return model_class(
        PhysiologyFrontendEncoder(
            observation_config(config["data"], config["model"])
        ),
        recursive["direct_horizons"],
        rollout_horizons=config["data"]["future_horizons"],
        num_classes=int(config["data"].get("num_classes", 5)),
        transition_layers=int(config["model"].get("transition_layers", 2)),
        transition_heads=int(config["model"].get("transition_heads", 4)),
        dropout=float(config["model"].get("dropout", 0.1)),
        freeze_observation_encoder=False,
        stage_residual_from_current=True,
        use_frozen_target_encoder=True,
        physiology_group_sizes=group_sizes,
        physiology_hidden_dim=int(physiology.get("hidden_dim", 128)),
        physiology_dynamics_dim=int(
            physiology.get("physiology_dynamics_dim", 64)
        ),
        physiology_dynamics_layers=int(
            physiology.get("physiology_dynamics_layers", 1)
        ),
        physiology_dynamics_heads=int(
            physiology.get("physiology_dynamics_heads", 4)
        ),
        waveform_seconds=int(waveform.get("max_seconds", 10)),
        waveform_patch_samples=int(waveform.get("patch_samples", 64)),
        waveform_decoder_dim=int(waveform.get("decoder_dim", 64)),
        waveform_decoder_layers=int(waveform.get("decoder_layers", 1)),
        waveform_decoder_heads=int(waveform.get("decoder_heads", 4)),
        waveform_structured_event_heads=bool(
            waveform.get("structured_event_heads", False)
        ),
        waveform_probabilistic_event_heads=bool(
            waveform.get("probabilistic_event_heads", False)
        ),
        waveform_ecg_refractory_event_head=bool(
            waveform.get("ecg_refractory_event_head", False)
        ),
        waveform_ecg_rr_bins=int(waveform.get("ecg_rr_bins", 48)),
        waveform_output_baseline=str(waveform.get("output_baseline", "repeat")),
        waveform_physiological_event_renderer=bool(
            waveform.get("physiological_event_renderer", False)
        ),
        waveform_ecg_time_aligned_renderer=bool(
            waveform.get("ecg_time_aligned_renderer", False)
        ),
        waveform_ecg_recursive_event_renderer=bool(
            waveform.get("ecg_recursive_event_renderer", False)
        ),
        waveform_ecg_recent_rr_residual=bool(
            waveform.get("ecg_recent_rr_residual", False)
        ),
        waveform_ecg_recent_amplitude_calibration=bool(
            waveform.get("ecg_recent_amplitude_calibration", False)
        ),
        waveform_safe_modality_residual_refinement=bool(
            waveform.get("safe_modality_residual_refinement", False)
        ),
        waveform_missing_modality_conditioning=bool(
            waveform.get("missing_modality_conditioning", False)
        ),
        waveform_missing_physiology_calibration=bool(
            waveform.get("missing_physiology_calibration", False)
        ),
        waveform_missing_emg_teacher_calibration=bool(
            waveform.get("missing_emg_teacher_calibration", False)
        ),
        waveform_ecg_event_sigma_seconds=float(
            waveform.get("ecg_event_sigma_seconds", 0.02)
        ),
        observation_adapter_hidden_dim=int(observation.get("hidden_dim", 128)),
        task_adapter_hidden_dim=int(observation.get("task_hidden_dim", 128)),
        recursive_hidden_dim=int(recursive.get("hidden_dim", 128)),
        recursive_anchor_direct=bool(recursive.get("anchor_direct", False)),
        **model_kwargs,
    )


def _checkpoint_state(checkpoint: dict, name: str) -> Dict[str, torch.Tensor]:
    state = checkpoint.get("model_state")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{name} checkpoint does not contain model_state")
    return state


def assemble_mainline_components(
    model: GatedRecursiveTaskAwareWaveformWorldModel,
    observation_checkpoint: dict,
    recursive_checkpoint: dict,
    waveform_checkpoint: dict,
) -> dict:
    observation_state = _checkpoint_state(observation_checkpoint, "O2")
    recursive_state = _checkpoint_state(recursive_checkpoint, "R3")
    waveform_state = _checkpoint_state(waveform_checkpoint, "W5")
    target_state = model.state_dict()
    assembled = {}
    source_counts = {
        "R3_non_waveform": 0,
        "W5_waveform": 0,
        "zero_initialized_missing_condition": 0,
    }
    for key, target in target_state.items():
        if key.startswith(WAVEFORM_PREFIX):
            source_name = "W5"
            source = waveform_state
            if key.startswith(MISSING_CONDITION_PREFIXES) and key not in source:
                assembled[key] = target
                source_counts["zero_initialized_missing_condition"] += 1
                continue
            source_counts["W5_waveform"] += 1
        else:
            source_name = "R3"
            source = recursive_state
            source_counts["R3_non_waveform"] += 1
        if key not in source:
            raise ValueError(f"{source_name} checkpoint is missing {key}")
        if source[key].shape != target.shape:
            raise ValueError(
                f"{source_name} shape mismatch for {key}: "
                f"source={tuple(source[key].shape)} target={tuple(target.shape)}"
            )
        assembled[key] = source[key]

    repair_keys = [
        key
        for key in target_state
        if key.startswith(OBSERVATION_PREFIXES)
    ]
    if not repair_keys:
        raise ValueError("mainline model has no O2 observation-repair parameters")
    for key in repair_keys:
        if key not in observation_state or key not in recursive_state:
            raise ValueError(f"O2/R3 lineage is missing {key}")
        if not torch.equal(observation_state[key], recursive_state[key]):
            raise ValueError(f"R3 does not preserve the selected O2 tensor {key}")

    model.load_state_dict(assembled, strict=True)
    loaded_state = model.state_dict()
    for key in target_state:
        if key.startswith(MISSING_CONDITION_PREFIXES) and key not in waveform_state:
            expected = target_state[key]
        else:
            expected = waveform_state[key] if key.startswith(WAVEFORM_PREFIX) else recursive_state[key]
        if not torch.equal(loaded_state[key], expected):
            raise RuntimeError(f"assembled tensor identity failed for {key}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    return {
        "o2_preserved_in_r3": True,
        "o2_repair_tensor_count": len(repair_keys),
        "r3_non_waveform_tensor_count": source_counts["R3_non_waveform"],
        "w5_waveform_tensor_count": source_counts["W5_waveform"],
        "zero_initialized_missing_condition_tensor_count": source_counts[
            "zero_initialized_missing_condition"
        ],
        "all_parameters_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "passed": True,
    }
