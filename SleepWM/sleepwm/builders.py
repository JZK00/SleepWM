from __future__ import annotations
from typing import Optional, Type
from sleepwm.models import GatedRecursiveTaskAwareWaveformWorldModel, PhysiologyFrontendEncoder, observation_config

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

