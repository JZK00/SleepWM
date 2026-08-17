import torch

from uniphysio_wm.models import (
    HybridRecursiveWaveformWorldModel,
    GatedRecursiveTaskAwareWaveformWorldModel,
    MaskedMultiModalModel,
    MultiModalEncoder,
    ObservationConfig,
    PrivateTemporalPhysiologyWorldModel,
    ReliabilityGatedObservationWaveformWorldModel,
    TaskAwareReliabilityObservationWaveformWorldModel,
    PhysiologyFrontendEncoder,
    PhysiologyAwareWorldModel,
    PhysiologyStateSpaceWorldModel,
    ShortHorizonWaveformWorldModel,
    SleepStageClassifier,
    TCNSleepClassifier,
    TrajectoryPhysiologyWorldModel,
)


def tiny_config() -> ObservationConfig:
    return ObservationConfig(
        sample_rate=16,
        epoch_seconds=2,
        patch_samples=8,
        d_model=16,
        layers=1,
        heads=4,
        dropout=0.0,
    )


def test_tcn_and_transformer_classifier_shapes() -> None:
    signals = torch.randn(4, 3, 32)
    present = torch.ones(4, 3, dtype=torch.bool)
    tcn = TCNSleepClassifier(3, channels=8, levels=2, num_classes=5)
    transformer = SleepStageClassifier(MultiModalEncoder(tiny_config()), num_classes=5)
    assert tcn(signals, present).shape == (4, 5)
    assert transformer(signals, present).shape == (4, 5)


def test_state_token_pooling_keeps_patch_contract() -> None:
    config = ObservationConfig(
        sample_rate=16,
        epoch_seconds=2,
        patch_samples=8,
        d_model=16,
        layers=1,
        heads=4,
        dropout=0.0,
        pooling="state",
    )
    encoder = MultiModalEncoder(config)
    signals = torch.randn(2, 3, config.samples_per_epoch)
    representation, tokens = encoder.encode_with_representation(signals)
    assert representation.shape == (2, config.d_model)
    assert tokens.shape == (2, 3, config.num_patches, config.d_model)


def test_masked_model_reconstructs_target_patches() -> None:
    config = tiny_config()
    model = MaskedMultiModalModel(MultiModalEncoder(config))
    signals = torch.randn(2, 3, config.samples_per_epoch)
    mask = torch.zeros(2, config.num_patches, dtype=torch.bool)
    mask[:, 1:3] = True
    output = model(signals, target_modality=1, patch_mask=mask)
    assert output["prediction"].shape == (2, config.num_patches, config.patch_samples)
    assert torch.isfinite(output["reconstruction_loss"])
    output["reconstruction_loss"].backward()


def test_masked_model_predicts_ema_teacher_latents() -> None:
    config = tiny_config()
    model = MaskedMultiModalModel(MultiModalEncoder(config), latent_prediction=True)
    signals = torch.randn(2, 3, config.samples_per_epoch)
    present = torch.ones(2, 3, dtype=torch.bool)
    mask = torch.zeros(2, config.num_patches, dtype=torch.bool)
    mask[:, 1:3] = True
    teacher = model.teacher_tokens(signals, present)
    output = model(signals, target_modality=0, patch_mask=mask, target_latent=teacher[:, 0])
    assert torch.isfinite(output["latent_reconstruction_loss"])
    output["latent_reconstruction_loss"].backward()
    before = next(model.teacher_encoder.parameters()).clone()
    with torch.no_grad():
        next(model.encoder.parameters()).add_(0.1)
    model.update_teacher(0.9)
    assert not torch.equal(before, next(model.teacher_encoder.parameters()))


def test_contextual_teacher_targets_depend_on_other_modalities() -> None:
    config = tiny_config()
    model = MaskedMultiModalModel(
        MultiModalEncoder(config),
        latent_prediction=True,
        teacher_target="contextual",
    )
    present = torch.ones(2, 3, dtype=torch.bool)
    signals = torch.randn(2, 3, config.samples_per_epoch)
    changed = signals.clone()
    changed[:, 1] = changed[:, 1] + 3.0 * torch.randn_like(changed[:, 1])

    original_target = model.teacher_tokens(signals, present)[:, 0]
    changed_target = model.teacher_tokens(changed, present)[:, 0]

    assert not torch.allclose(original_target, changed_target)


def test_frozen_probe_keeps_encoder_in_evaluation_mode() -> None:
    model = SleepStageClassifier(MultiModalEncoder(tiny_config()), freeze_encoder=True)
    model.train()
    assert model.training
    assert not model.encoder.training


def test_physiology_world_model_outputs_current_and_residual_future_features() -> None:
    model = PhysiologyAwareWorldModel(
        MultiModalEncoder(tiny_config()),
        horizons=(1, 2),
        transition_layers=1,
        transition_heads=4,
        dropout=0.0,
        stage_residual_from_current=True,
        physiology_group_sizes={"EEG": 5, "ECG": 3, "EMG": 3},
        physiology_hidden_dim=16,
    ).eval()
    history = torch.randn(2, 3, 3, tiny_config().samples_per_epoch)
    with torch.no_grad():
        output = model.rollout(history)
    assert output["current_physiology"].shape == (2, 11)
    assert output["future_physiology"].shape == (2, 2, 11)
    expected = output["current_physiology"].unsqueeze(1).expand(-1, 2, -1)
    assert torch.equal(output["future_physiology"], expected)


def test_physiology_losses_respect_validity_masks() -> None:
    model = PhysiologyAwareWorldModel(
        MultiModalEncoder(tiny_config()),
        horizons=(1,),
        transition_layers=1,
        transition_heads=4,
        dropout=0.0,
        physiology_group_sizes={"EEG": 5, "ECG": 3, "EMG": 3},
        physiology_hidden_dim=16,
    )
    history = torch.randn(2, 3, 3, tiny_config().samples_per_epoch)
    current_target = torch.zeros(2, 11)
    future_target = torch.zeros(2, 1, 11)
    current_valid = torch.ones_like(current_target, dtype=torch.bool)
    future_valid = torch.ones_like(future_target, dtype=torch.bool)
    future_target[:, :, 6] = 1e6
    future_valid[:, :, 6] = False
    output = model(
        history,
        current_physiology_targets=current_target,
        current_physiology_valid=current_valid,
        future_physiology_targets=future_target,
        future_physiology_valid=future_valid,
    )
    assert torch.isfinite(output["current_physiology_loss"])
    assert torch.isfinite(output["future_physiology_loss"])
    assert float(output["future_physiology_loss"].detach()) < 10.0
    output["loss"].backward()


def test_trajectory_physiology_model_uses_full_modality_history() -> None:
    model = TrajectoryPhysiologyWorldModel(
        MultiModalEncoder(tiny_config()),
        horizons=(1, 2),
        transition_layers=1,
        transition_heads=4,
        dropout=0.0,
        stage_residual_from_current=True,
        physiology_group_sizes={"EEG": 5, "ECG": 3, "EMG": 3},
        physiology_hidden_dim=16,
    ).eval()
    history = torch.randn(2, 3, 3, tiny_config().samples_per_epoch)
    with torch.no_grad():
        output = model.rollout(history)
    assert output["history_physiology"].shape == (2, 3, 11)
    assert output["current_physiology"].shape == (2, 11)
    assert output["future_physiology"].shape == (2, 2, 11)
    assert torch.count_nonzero(output["physiology_adapter_delta"]) == 0
    expected = output["current_physiology"].unsqueeze(1).expand(-1, 2, -1)
    assert torch.equal(output["future_physiology"], expected)


def test_trajectory_physiology_loss_updates_zero_initialized_adapter() -> None:
    model = TrajectoryPhysiologyWorldModel(
        MultiModalEncoder(tiny_config()),
        horizons=(1,),
        transition_layers=1,
        transition_heads=4,
        dropout=0.0,
        physiology_group_sizes={"EEG": 5, "ECG": 3, "EMG": 3},
        physiology_hidden_dim=16,
    )
    history = torch.randn(2, 3, 3, tiny_config().samples_per_epoch)
    future = torch.randn(2, 1, 3, tiny_config().samples_per_epoch)
    history_target = torch.zeros(2, 3, 11)
    history_valid = torch.ones_like(history_target, dtype=torch.bool)
    future_target = torch.zeros(2, 1, 11)
    future_valid = torch.ones_like(future_target, dtype=torch.bool)
    history_target[:, 1, 6] = 1e6
    history_valid[:, 1, 6] = False
    output = model(
        history,
        future_signals=future,
        history_physiology_targets=history_target,
        history_physiology_valid=history_valid,
        future_physiology_targets=future_target,
        future_physiology_valid=future_valid,
    )
    assert torch.isfinite(output["history_physiology_loss"])
    assert torch.isfinite(output["future_physiology_loss"])
    output["loss"].backward()
    adapter_gradient = model.physiology_state_adapters["EEG"].weight.grad
    assert adapter_gradient is not None
    assert torch.count_nonzero(adapter_gradient) > 0


def test_private_temporal_model_starts_from_zero_late_fusion() -> None:
    model = PrivateTemporalPhysiologyWorldModel(
        MultiModalEncoder(tiny_config()),
        horizons=(1, 2),
        transition_layers=1,
        transition_heads=4,
        private_transition_layers=1,
        private_transition_heads=4,
        dropout=0.0,
        stage_residual_from_current=True,
        physiology_group_sizes={"EEG": 5, "ECG": 3, "EMG": 3},
        physiology_hidden_dim=16,
    ).eval()
    history = torch.randn(2, 3, 3, tiny_config().samples_per_epoch)
    with torch.no_grad():
        output = model.rollout(history)
    assert not hasattr(model, "physiology_state_adapters")
    assert output["private_predicted_states"].shape == (2, 2, 3, 16)
    assert torch.count_nonzero(output["private_fusion_delta"]) == 0
    expected = output["current_physiology"].unsqueeze(1).expand(-1, 2, -1)
    assert torch.equal(output["future_physiology"], expected)


def test_private_temporal_latent_loss_trains_private_dynamics() -> None:
    model = PrivateTemporalPhysiologyWorldModel(
        MultiModalEncoder(tiny_config()),
        horizons=(1,),
        transition_layers=1,
        transition_heads=4,
        private_transition_layers=1,
        private_transition_heads=4,
        dropout=0.0,
        use_frozen_target_encoder=True,
        physiology_group_sizes={"EEG": 5, "ECG": 3, "EMG": 3},
        physiology_hidden_dim=16,
    )
    history = torch.randn(2, 3, 3, tiny_config().samples_per_epoch)
    future = torch.randn(2, 1, 3, tiny_config().samples_per_epoch)
    history_target = torch.zeros(2, 3, 11)
    future_target = torch.zeros(2, 1, 11)
    output = model(
        history,
        future_signals=future,
        history_physiology_targets=history_target,
        history_physiology_valid=torch.ones_like(history_target, dtype=torch.bool),
        future_physiology_targets=future_target,
        future_physiology_valid=torch.ones_like(future_target, dtype=torch.bool),
    )
    assert torch.isfinite(output["private_latent_loss"])
    output["loss"].backward()
    private_gradient = next(model.private_transitions["ECG"].parameters()).grad
    fusion_gradient = model.private_future_state_adapters["ECG"].weight.grad
    assert private_gradient is not None
    assert fusion_gradient is not None
    assert torch.count_nonzero(private_gradient) > 0
    assert torch.count_nonzero(fusion_gradient) > 0


def test_physiology_frontend_is_zero_initialized_and_finite() -> None:
    encoder = PhysiologyFrontendEncoder(tiny_config()).eval()
    signals = torch.randn(2, 3, tiny_config().samples_per_epoch)
    with torch.no_grad():
        residuals = encoder.modality_token_residuals(signals)
        eeg_features = encoder._eeg_features(signals[:, 0])
        ecg_global, ecg_local = encoder._ecg_features(signals[:, 1])
        emg_features = encoder._emg_features(signals[:, 2])
    assert residuals is not None
    assert residuals.shape == (2, 3, tiny_config().num_patches, tiny_config().d_model)
    assert torch.count_nonzero(residuals) == 0
    assert eeg_features.shape == (2, 5)
    assert ecg_global.shape == (2, 27)
    assert ecg_local.shape == (2, tiny_config().num_patches, 10)
    assert emg_features.shape == (2, 3)
    assert torch.isfinite(eeg_features).all()
    assert torch.isfinite(ecg_global).all()
    assert torch.isfinite(ecg_local).all()
    assert torch.isfinite(emg_features).all()


def test_physiology_frontend_preserves_initial_encoder_and_receives_gradient() -> None:
    frontend = PhysiologyFrontendEncoder(tiny_config()).eval()
    plain = MultiModalEncoder(tiny_config()).eval()
    shared_state = {
        key: value
        for key, value in frontend.state_dict().items()
        if not key.startswith("physiology_frontends.")
    }
    plain.load_state_dict(shared_state)
    signals = torch.randn(2, 3, tiny_config().samples_per_epoch)
    with torch.no_grad():
        assert torch.equal(frontend(signals), plain(signals))
    frontend.train()
    frontend(signals)[:, 0].sum().backward()
    gradient = frontend.physiology_frontends["ECG_local"].projection.weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient) > 0


def test_frozen_target_encoder_disables_physiology_frontend() -> None:
    model = TrajectoryPhysiologyWorldModel(
        PhysiologyFrontendEncoder(tiny_config()),
        horizons=(1,),
        transition_layers=1,
        transition_heads=4,
        dropout=0.0,
        use_frozen_target_encoder=True,
        physiology_group_sizes={"EEG": 5, "ECG": 3, "EMG": 3},
        physiology_hidden_dim=16,
    )
    assert model.encoder.frontend_enabled
    assert model.target_encoder is not None
    assert not model.target_encoder.frontend_enabled


def test_physiology_state_space_starts_from_f4_outputs() -> None:
    arguments = {
        "horizons": (1, 2),
        "transition_layers": 1,
        "transition_heads": 4,
        "dropout": 0.0,
        "stage_residual_from_current": True,
        "physiology_group_sizes": {"EEG": 5, "ECG": 3, "EMG": 3},
        "physiology_hidden_dim": 16,
    }
    f5 = PhysiologyStateSpaceWorldModel(
        PhysiologyFrontendEncoder(tiny_config()),
        physiology_dynamics_dim=16,
        physiology_dynamics_layers=1,
        physiology_dynamics_heads=4,
        **arguments,
    ).eval()
    f4 = TrajectoryPhysiologyWorldModel(
        PhysiologyFrontendEncoder(tiny_config()),
        **arguments,
    ).eval()
    f4_state = {
        key: value
        for key, value in f5.state_dict().items()
        if not key.startswith("physiology_dynamics_")
    }
    f4.load_state_dict(f4_state)
    history = torch.randn(2, 3, 3, tiny_config().samples_per_epoch)
    with torch.no_grad():
        f4_output = f4.rollout(history)
        f5_output = f5.rollout(history)
    assert torch.count_nonzero(f5_output["physiology_trajectory_delta"]) == 0
    assert torch.count_nonzero(f5_output["physiology_dynamics_state_delta"]) == 0
    assert torch.equal(f5_output["predicted_states"], f4_output["predicted_states"])
    assert torch.equal(f5_output["stage_logits"], f4_output["stage_logits"])
    assert torch.equal(f5_output["future_physiology"], f4_output["future_physiology"])


def test_physiology_state_space_new_outputs_receive_gradient() -> None:
    model = PhysiologyStateSpaceWorldModel(
        PhysiologyFrontendEncoder(tiny_config()),
        horizons=(1,),
        transition_layers=1,
        transition_heads=4,
        physiology_dynamics_dim=16,
        physiology_dynamics_layers=1,
        physiology_dynamics_heads=4,
        dropout=0.0,
        physiology_group_sizes={"EEG": 5, "ECG": 3, "EMG": 3},
        physiology_hidden_dim=16,
    )
    history = torch.randn(2, 3, 3, tiny_config().samples_per_epoch)
    future = torch.randn(2, 1, 3, tiny_config().samples_per_epoch)
    history_target = torch.zeros(2, 3, 11)
    future_target = torch.ones(2, 1, 11)
    output = model(
        history,
        future_signals=future,
        history_physiology_targets=history_target,
        history_physiology_valid=torch.ones_like(history_target, dtype=torch.bool),
        future_physiology_targets=future_target,
        future_physiology_valid=torch.ones_like(future_target, dtype=torch.bool),
    )
    output["loss"].backward()
    delta_gradient = model.physiology_dynamics_delta_heads["ECG"][-1].weight.grad
    state_gradient = model.physiology_dynamics_state_adapters["ECG"].weight.grad
    assert delta_gradient is not None
    assert state_gradient is not None
    assert torch.count_nonzero(delta_gradient) > 0
    assert torch.count_nonzero(state_gradient) > 0


def test_short_waveform_model_preserves_f5_and_starts_from_repeat_window() -> None:
    arguments = {
        "horizons": (1, 2),
        "transition_layers": 1,
        "transition_heads": 4,
        "physiology_dynamics_dim": 16,
        "physiology_dynamics_layers": 1,
        "physiology_dynamics_heads": 4,
        "dropout": 0.0,
        "stage_residual_from_current": True,
        "physiology_group_sizes": {"EEG": 5, "ECG": 3, "EMG": 3},
        "physiology_hidden_dim": 16,
    }
    waveform_model = ShortHorizonWaveformWorldModel(
        PhysiologyFrontendEncoder(tiny_config()),
        waveform_seconds=1,
        waveform_patch_samples=8,
        waveform_decoder_dim=16,
        waveform_decoder_layers=1,
        waveform_decoder_heads=4,
        **arguments,
    ).eval()
    f5 = PhysiologyStateSpaceWorldModel(
        PhysiologyFrontendEncoder(tiny_config()), **arguments
    ).eval()
    f5.load_state_dict(
        {
            key: value
            for key, value in waveform_model.state_dict().items()
            if not key.startswith("waveform_decoder.")
        }
    )
    history = torch.randn(2, 3, 3, tiny_config().samples_per_epoch)
    with torch.no_grad():
        f5_output = f5.rollout(history)
        waveform_output = waveform_model.rollout(history)
    assert torch.equal(waveform_output["predicted_states"], f5_output["predicted_states"])
    assert torch.equal(waveform_output["stage_logits"], f5_output["stage_logits"])
    assert torch.equal(
        waveform_output["future_physiology"], f5_output["future_physiology"]
    )
    assert torch.equal(
        waveform_output["future_waveforms"], history[:, -1, :, -16:]
    )


def test_short_waveform_loss_trains_zero_initialized_decoder() -> None:
    model = ShortHorizonWaveformWorldModel(
        PhysiologyFrontendEncoder(tiny_config()),
        horizons=(1,),
        transition_layers=1,
        transition_heads=4,
        physiology_dynamics_dim=16,
        physiology_dynamics_layers=1,
        physiology_dynamics_heads=4,
        dropout=0.0,
        physiology_group_sizes={"EEG": 5, "ECG": 3, "EMG": 3},
        physiology_hidden_dim=16,
        waveform_seconds=1,
        waveform_patch_samples=8,
        waveform_decoder_dim=16,
        waveform_decoder_layers=1,
        waveform_decoder_heads=4,
    )
    history = torch.randn(2, 3, 3, tiny_config().samples_per_epoch)
    output = model.rollout(history)
    losses = model.waveform_decoder.waveform_loss(
        output["future_waveforms"],
        torch.zeros_like(output["future_waveforms"]),
        torch.ones(2, 3, dtype=torch.bool),
        horizons_seconds=(1,),
    )
    losses["loss"].backward()
    gradient = model.waveform_decoder.output_heads["ECG"][-1].weight.grad
    assert torch.isfinite(losses["loss"])
    assert gradient is not None
    assert torch.count_nonzero(gradient) > 0


def test_hybrid_recursive_rollout_starts_exactly_from_direct_outputs() -> None:
    model = HybridRecursiveWaveformWorldModel(
        PhysiologyFrontendEncoder(tiny_config()),
        horizons=(1, 2, 4),
        transition_layers=1,
        transition_heads=4,
        physiology_dynamics_dim=16,
        physiology_dynamics_layers=1,
        physiology_dynamics_heads=4,
        dropout=0.0,
        stage_residual_from_current=True,
        physiology_group_sizes={"EEG": 5, "ECG": 3, "EMG": 3},
        physiology_hidden_dim=16,
        waveform_seconds=1,
        waveform_patch_samples=8,
        waveform_decoder_dim=16,
        waveform_decoder_layers=1,
        waveform_decoder_heads=4,
        recursive_hidden_dim=16,
    ).eval()
    history = torch.randn(2, 3, 3, tiny_config().samples_per_epoch)
    with torch.no_grad():
        output = model.rollout_context(history)
    assert torch.count_nonzero(output["recursive_state_correction"]) == 0
    assert torch.equal(output["predicted_states"], output["direct_predicted_states"])
    assert torch.equal(output["stage_logits"], output["direct_stage_logits"])
    assert torch.equal(
        output["future_physiology"], output["direct_future_physiology"]
    )


def test_hybrid_recursive_rollout_accumulates_shared_step_residuals() -> None:
    model = HybridRecursiveWaveformWorldModel(
        PhysiologyFrontendEncoder(tiny_config()),
        horizons=(1, 2, 4),
        transition_layers=1,
        transition_heads=4,
        physiology_dynamics_dim=16,
        physiology_dynamics_layers=1,
        physiology_dynamics_heads=4,
        dropout=0.0,
        physiology_group_sizes={"EEG": 5, "ECG": 3, "EMG": 3},
        physiology_hidden_dim=16,
        waveform_seconds=1,
        waveform_patch_samples=8,
        waveform_decoder_dim=16,
        waveform_decoder_layers=1,
        waveform_decoder_heads=4,
        recursive_hidden_dim=16,
    ).eval()
    with torch.no_grad():
        model.recursive_delta_head[-1].bias.fill_(0.05)
    history = torch.randn(2, 3, 3, tiny_config().samples_per_epoch)
    with torch.no_grad():
        correction = model.rollout_context(history)["recursive_state_correction"]
    assert torch.allclose(correction[:, 1], 2.0 * correction[:, 0], atol=1e-6)
    assert torch.allclose(correction[:, 2], 4.0 * correction[:, 0], atol=1e-6)


def test_reliability_observation_repair_starts_from_direct_and_masks_absent() -> None:
    model = ReliabilityGatedObservationWaveformWorldModel(
        PhysiologyFrontendEncoder(tiny_config()),
        horizons=(1, 2),
        transition_layers=1,
        transition_heads=4,
        physiology_dynamics_dim=16,
        physiology_dynamics_layers=1,
        physiology_dynamics_heads=4,
        dropout=0.0,
        stage_residual_from_current=True,
        physiology_group_sizes={"EEG": 5, "ECG": 3, "EMG": 3},
        physiology_hidden_dim=16,
        waveform_seconds=1,
        waveform_patch_samples=8,
        waveform_decoder_dim=16,
        waveform_decoder_layers=1,
        waveform_decoder_heads=4,
        observation_adapter_hidden_dim=16,
    ).eval()
    history = torch.randn(2, 3, 3, tiny_config().samples_per_epoch)
    present = torch.ones(2, 3, 3, dtype=torch.bool)
    present[:, :, 2] = False
    history[:, :, 2] = 0.0
    with torch.no_grad():
        initial = model.rollout_context(history, present)
    assert torch.count_nonzero(initial["observation_state_correction"]) == 0
    assert torch.equal(initial["predicted_states"], initial["direct_predicted_states"])
    assert torch.count_nonzero(initial["observation_reliability"][:, 2]) == 0

    with torch.no_grad():
        model.observation_state_adapters["ECG"][-1].bias.fill_(0.1)
        repaired = model.rollout_context(history, present)
    assert torch.count_nonzero(repaired["observation_state_correction"]) > 0
    assert not torch.equal(repaired["predicted_states"], repaired["direct_predicted_states"])


def test_task_aware_observation_repair_starts_from_observation_branch() -> None:
    model = TaskAwareReliabilityObservationWaveformWorldModel(
        PhysiologyFrontendEncoder(tiny_config()),
        horizons=(1, 2),
        transition_layers=1,
        transition_heads=4,
        physiology_dynamics_dim=16,
        physiology_dynamics_layers=1,
        physiology_dynamics_heads=4,
        dropout=0.0,
        stage_residual_from_current=True,
        physiology_group_sizes={"EEG": 5, "ECG": 3, "EMG": 3},
        physiology_hidden_dim=16,
        waveform_seconds=1,
        waveform_patch_samples=8,
        waveform_decoder_dim=16,
        waveform_decoder_layers=1,
        waveform_decoder_heads=4,
        observation_adapter_hidden_dim=16,
        task_adapter_hidden_dim=16,
    ).eval()
    history = torch.randn(2, 3, 3, tiny_config().samples_per_epoch)
    present = torch.ones(2, 3, 3, dtype=torch.bool)
    present[:, :, 2] = False
    history[:, :, 2] = 0.0
    with torch.no_grad():
        initial = model.rollout_context(history, present)
    assert torch.equal(initial["stage_logits"], initial["observation_base_stage_logits"])
    assert torch.equal(
        initial["future_physiology"],
        initial["observation_base_future_physiology"],
    )
    assert torch.count_nonzero(initial["task_stage_residual"]) == 0
    assert torch.count_nonzero(initial["task_physiology_residual"]) == 0


def test_task_aware_observation_repair_uses_only_available_modalities() -> None:
    model = TaskAwareReliabilityObservationWaveformWorldModel(
        PhysiologyFrontendEncoder(tiny_config()),
        horizons=(1,),
        transition_layers=1,
        transition_heads=4,
        physiology_dynamics_dim=16,
        physiology_dynamics_layers=1,
        physiology_dynamics_heads=4,
        dropout=0.0,
        physiology_group_sizes={"EEG": 5, "ECG": 3, "EMG": 3},
        physiology_hidden_dim=16,
        waveform_seconds=1,
        waveform_patch_samples=8,
        waveform_decoder_dim=16,
        waveform_decoder_layers=1,
        waveform_decoder_heads=4,
        observation_adapter_hidden_dim=16,
        task_adapter_hidden_dim=16,
    ).eval()
    with torch.no_grad():
        model.task_stage_residual_heads["ECG"][-1].bias.fill_(0.25)
        model.task_stage_residual_heads["EMG"][-1].bias.fill_(2.0)
        model.task_physiology_residual_heads["ECG"][-1].bias.fill_(0.5)
        model.task_physiology_residual_heads["EMG"][-1].bias.fill_(3.0)
    history = torch.randn(2, 3, 3, tiny_config().samples_per_epoch)
    present = torch.zeros(2, 3, 3, dtype=torch.bool)
    present[:, :, 1] = True
    history[:, :, 0] = 0.0
    history[:, :, 2] = 0.0
    with torch.no_grad():
        output = model.rollout_context(history, present)
    assert torch.allclose(
        output["task_stage_residual"],
        torch.full_like(output["task_stage_residual"], 0.25),
        atol=1e-6,
    )
    assert torch.allclose(
        output["task_physiology_residual"],
        torch.full_like(output["task_physiology_residual"], 0.5),
        atol=1e-6,
    )


def test_gated_recursive_rollout_supports_true_long_horizons() -> None:
    model = GatedRecursiveTaskAwareWaveformWorldModel(
        PhysiologyFrontendEncoder(tiny_config()),
        horizons=(1, 2, 4),
        rollout_horizons=(1, 2, 4, 10, 14),
        transition_layers=1,
        transition_heads=4,
        physiology_dynamics_dim=16,
        physiology_dynamics_layers=1,
        physiology_dynamics_heads=4,
        dropout=0.0,
        physiology_group_sizes={"EEG": 5, "ECG": 3, "EMG": 3},
        physiology_hidden_dim=16,
        waveform_seconds=1,
        waveform_patch_samples=8,
        waveform_decoder_dim=16,
        waveform_decoder_layers=1,
        waveform_decoder_heads=4,
        observation_adapter_hidden_dim=16,
        task_adapter_hidden_dim=16,
        recursive_hidden_dim=16,
    ).eval()
    history = torch.randn(2, 3, 3, tiny_config().samples_per_epoch)
    present = torch.ones(2, 3, 3, dtype=torch.bool)
    with torch.no_grad():
        output = model.rollout_context_horizons(history, present)
    assert output["predicted_states"].shape == (2, 5, 16)
    assert output["stage_logits"].shape == (2, 5, 5)
    assert output["future_physiology"].shape == (2, 5, 11)
    assert output["recursive_update_gate"].shape == (2, 5, 16)
    assert output["recursive_log_variance"].shape == (2, 5)
    assert output["recursive_horizons"].tolist() == [1, 2, 4, 10, 14]
    expected = output["corrected_history_state"].unsqueeze(1).expand(-1, 5, -1)
    assert torch.equal(output["predicted_states"], expected)


def test_gated_recursive_rollout_accumulates_state_updates() -> None:
    model = GatedRecursiveTaskAwareWaveformWorldModel(
        PhysiologyFrontendEncoder(tiny_config()),
        horizons=(1, 2, 4),
        rollout_horizons=(1, 2, 4),
        transition_layers=1,
        transition_heads=4,
        physiology_dynamics_dim=16,
        physiology_dynamics_layers=1,
        physiology_dynamics_heads=4,
        dropout=0.0,
        physiology_group_sizes={"EEG": 5, "ECG": 3, "EMG": 3},
        physiology_hidden_dim=16,
        waveform_seconds=1,
        waveform_patch_samples=8,
        waveform_decoder_dim=16,
        waveform_decoder_layers=1,
        waveform_decoder_heads=4,
        observation_adapter_hidden_dim=16,
        task_adapter_hidden_dim=16,
        recursive_hidden_dim=16,
    ).eval()
    with torch.no_grad():
        model.recursive_delta_head[-1].bias.fill_(0.1)
        model.recursive_update_gate[-2].weight.zero_()
        model.recursive_update_gate[-2].bias.fill_(20.0)
    history = torch.randn(2, 3, 3, tiny_config().samples_per_epoch)
    with torch.no_grad():
        correction = model.rollout_context_horizons(history)[
            "recursive_state_correction"
        ]
    assert torch.allclose(correction[:, 1], 2.0 * correction[:, 0], atol=1e-5)
    assert torch.allclose(correction[:, 2], 4.0 * correction[:, 0], atol=1e-5)


def test_anchored_gated_rollout_retains_direct_short_states() -> None:
    model = GatedRecursiveTaskAwareWaveformWorldModel(
        PhysiologyFrontendEncoder(tiny_config()),
        horizons=(1, 2, 4),
        rollout_horizons=(1, 2, 4, 10, 14),
        recursive_anchor_direct=True,
        transition_layers=1,
        transition_heads=4,
        physiology_dynamics_dim=16,
        physiology_dynamics_layers=1,
        physiology_dynamics_heads=4,
        dropout=0.0,
        physiology_group_sizes={"EEG": 5, "ECG": 3, "EMG": 3},
        physiology_hidden_dim=16,
        waveform_seconds=1,
        waveform_patch_samples=8,
        waveform_decoder_dim=16,
        waveform_decoder_layers=1,
        waveform_decoder_heads=4,
        observation_adapter_hidden_dim=16,
        task_adapter_hidden_dim=16,
        recursive_hidden_dim=16,
    ).eval()
    history = torch.randn(2, 3, 3, tiny_config().samples_per_epoch)
    with torch.no_grad():
        output = model.rollout_context_horizons(history)
    assert torch.equal(
        output["predicted_states"][:, :3], output["observation_predicted_states"]
    )
    assert torch.equal(
        output["stage_logits"][:, :3], output["observation_stage_logits"]
    )
    assert torch.equal(
        output["future_physiology"][:, :3],
        output["observation_future_physiology"],
    )
    anchor = output["observation_predicted_states"][:, 2].unsqueeze(1)
    assert torch.equal(output["predicted_states"][:, 3:], anchor.expand(-1, 2, -1))
    assert torch.count_nonzero(output["recursive_state_correction"]) == 0


def test_structured_waveform_heads_condition_decoder_and_receive_gradients() -> None:
    model = ShortHorizonWaveformWorldModel(
        PhysiologyFrontendEncoder(tiny_config()),
        horizons=(1,),
        transition_layers=1,
        transition_heads=4,
        physiology_dynamics_dim=16,
        physiology_dynamics_layers=1,
        physiology_dynamics_heads=4,
        dropout=0.0,
        physiology_group_sizes={"EEG": 5, "ECG": 3, "EMG": 3},
        physiology_hidden_dim=16,
        waveform_seconds=1,
        waveform_patch_samples=8,
        waveform_decoder_dim=16,
        waveform_decoder_layers=1,
        waveform_decoder_heads=4,
        waveform_structured_event_heads=True,
    )
    history = torch.randn(2, 3, 3, tiny_config().samples_per_epoch)
    output = model.rollout(history)
    assert torch.equal(
        output["future_waveforms"], history[:, -1, :, -tiny_config().sample_rate :]
    )
    assert output["future_waveform_structure"]["ECG"]["qrs_logits"].shape == (
        2,
        tiny_config().sample_rate,
    )
    target = torch.randn_like(output["future_waveforms"])
    losses = model.waveform_decoder.waveform_loss(
        output["future_waveforms"],
        target,
        torch.ones(2, 3, dtype=torch.bool),
        horizons_seconds=(1,),
        structure_predictions=output["future_waveform_structure"],
        multi_resolution_fft_sizes=(4, 8, 16),
        auxiliary_structure_weight=0.25,
    )
    losses["loss"].backward()
    structure_gradient = model.waveform_decoder.structure_heads["ECG"][-1].weight.grad
    waveform_gradient = model.waveform_decoder.output_heads["ECG"][-1].weight.grad
    assert torch.isfinite(losses["loss"])
    assert structure_gradient is not None
    assert torch.count_nonzero(structure_gradient) > 0
    assert waveform_gradient is not None
    assert torch.count_nonzero(waveform_gradient) > 0


def test_no_repeat_waveform_decoder_has_no_recent_signal_skip() -> None:
    model = ShortHorizonWaveformWorldModel(
        PhysiologyFrontendEncoder(tiny_config()),
        horizons=(1,),
        transition_layers=1,
        transition_heads=4,
        physiology_dynamics_dim=16,
        physiology_dynamics_layers=1,
        physiology_dynamics_heads=4,
        dropout=0.0,
        physiology_group_sizes={"EEG": 5, "ECG": 3, "EMG": 3},
        physiology_hidden_dim=16,
        waveform_seconds=1,
        waveform_patch_samples=8,
        waveform_decoder_dim=16,
        waveform_decoder_layers=1,
        waveform_decoder_heads=4,
        waveform_output_baseline="none",
    ).eval()
    history = torch.randn(2, 3, 3, tiny_config().samples_per_epoch)
    with torch.no_grad():
        output = model.rollout(history)
    assert torch.count_nonzero(output["future_waveforms"]) == 0
    assert not torch.equal(
        output["future_waveforms"],
        history[:, -1, :, -tiny_config().sample_rate :],
    )
