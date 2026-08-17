import torch

from uniphysio_wm.models import (
    ObservationConfig,
    PhysiologyFrontendEncoder,
    ShortHorizonWaveformWorldModel,
)
from uniphysio_wm.event_aligned_waveform import ecg_event_alignment_loss
from uniphysio_wm.probabilistic_waveform import (
    probabilistic_waveform_gate_result,
    probabilistic_waveform_loss,
    probabilistic_waveform_metrics,
    refractory_ecg_gate_result,
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


def build_model(
    probabilistic: bool,
    refractory_ecg: bool = False,
    physiological_renderer: bool = False,
    time_aligned_ecg: bool = False,
    recursive_ecg: bool = False,
    recent_rr_residual: bool = False,
    amplitude_calibration: bool = False,
    safe_refinement: bool = False,
) -> ShortHorizonWaveformWorldModel:
    return ShortHorizonWaveformWorldModel(
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
        waveform_probabilistic_event_heads=probabilistic,
        waveform_ecg_refractory_event_head=refractory_ecg,
        waveform_ecg_rr_bins=8,
        waveform_output_baseline="none" if physiological_renderer else "repeat",
        waveform_physiological_event_renderer=physiological_renderer,
        waveform_ecg_time_aligned_renderer=time_aligned_ecg,
        waveform_ecg_recursive_event_renderer=recursive_ecg,
        waveform_ecg_recent_rr_residual=recent_rr_residual,
        waveform_ecg_recent_amplitude_calibration=amplitude_calibration,
        waveform_safe_modality_residual_refinement=safe_refinement,
    )


def test_probability_residual_starts_exactly_from_f5w2_waveform() -> None:
    base = build_model(False).eval()
    probabilistic = build_model(True).eval()
    incompatible = probabilistic.load_state_dict(base.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(
        key.startswith(
            (
                "waveform_decoder.probability_heads.",
                "waveform_decoder.probability_adapters.",
                "waveform_decoder.event_output_heads.",
            )
        )
        for key in incompatible.missing_keys
    )
    history = torch.randn(2, 3, 3, tiny_config().samples_per_epoch)
    with torch.no_grad():
        base_output = base.rollout(history)
        probability_output = probabilistic.rollout(history)
    assert torch.equal(
        probability_output["future_waveforms"], base_output["future_waveforms"]
    )
    assert probability_output["future_waveform_probabilities"]["ECG"][
        "qrs_logits"
    ].shape == (2, tiny_config().sample_rate)


def test_probability_and_conditioned_waveform_losses_train_new_modules() -> None:
    model = build_model(True)
    history = torch.randn(3, 3, 3, tiny_config().samples_per_epoch)
    output = model.rollout(history)
    target = torch.randn_like(output["future_waveforms"])
    valid = torch.ones(3, 3, dtype=torch.bool)
    probability_losses = probabilistic_waveform_loss(
        output["future_waveform_probabilities"],
        target,
        valid,
        modalities=("EEG", "ECG", "EMG"),
        sample_rate=tiny_config().sample_rate,
        horizons_seconds=(1,),
        patch_samples=8,
    )
    waveform_losses = model.waveform_decoder.waveform_loss(
        output["future_waveforms"],
        target,
        valid,
        horizons_seconds=(1,),
        multi_resolution_fft_sizes=(4, 8, 16),
    )
    loss = waveform_losses["loss"] + 0.25 * probability_losses["loss"]
    loss.backward()
    probability_gradient = model.waveform_decoder.probability_heads["ECG"][-1].weight.grad
    waveform_gradient = model.waveform_decoder.event_output_heads["ECG"][-1].weight.grad
    assert torch.isfinite(loss)
    assert probability_gradient is not None
    assert torch.count_nonzero(probability_gradient) > 0
    assert waveform_gradient is not None
    assert torch.count_nonzero(waveform_gradient) > 0


def test_refractory_ecg_starts_from_s6p0_and_trains_only_new_path() -> None:
    s6p0 = build_model(True).eval()
    refractory = build_model(True, refractory_ecg=True).eval()
    incompatible = refractory.load_state_dict(s6p0.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(
        key.startswith(
            (
                "waveform_decoder.ecg_point_process_head.",
                "waveform_decoder.ecg_point_process_adapter.",
                "waveform_decoder.ecg_point_process_output_head.",
            )
        )
        for key in incompatible.missing_keys
    )
    history = torch.randn(3, 3, 3, tiny_config().samples_per_epoch)
    with torch.no_grad():
        s6p0_output = s6p0.rollout(history)
        refractory_output = refractory.rollout(history)
    assert torch.equal(
        refractory_output["future_waveforms"], s6p0_output["future_waveforms"]
    )
    ecg = refractory_output["future_waveform_probabilities"]["ECG"]
    assert ecg["qrs_logits"].shape == (3, tiny_config().sample_rate)
    assert ecg["rr_logits"].shape == (3, 2, 8)


def test_refractory_ecg_probability_and_waveform_paths_receive_gradients() -> None:
    model = build_model(True, refractory_ecg=True)
    history = torch.randn(3, 3, 3, tiny_config().samples_per_epoch)
    output = model.rollout(history)
    target = torch.randn_like(output["future_waveforms"])
    valid = torch.ones(3, 3, dtype=torch.bool)
    probability_losses = probabilistic_waveform_loss(
        output["future_waveform_probabilities"],
        target,
        valid,
        modalities=("EEG", "ECG", "EMG"),
        sample_rate=tiny_config().sample_rate,
        horizons_seconds=(1,),
        patch_samples=8,
    )
    waveform_losses = model.waveform_decoder.waveform_loss(
        output["future_waveforms"],
        target,
        valid,
        horizons_seconds=(1,),
        multi_resolution_fft_sizes=(4, 8, 16),
    )
    loss = waveform_losses["loss"] + 0.25 * probability_losses["loss"]
    loss.backward()
    event_gradient = model.waveform_decoder.ecg_point_process_head[-1].weight.grad
    output_gradient = model.waveform_decoder.ecg_point_process_output_head[-1].weight.grad
    assert torch.isfinite(loss)
    assert event_gradient is not None
    assert torch.count_nonzero(event_gradient) > 0
    assert output_gradient is not None
    assert torch.count_nonzero(output_gradient) > 0


def test_physiological_renderer_generates_all_modalities_without_repeat_skip() -> None:
    model = build_model(
        True, refractory_ecg=True, physiological_renderer=True
    ).eval()
    with torch.no_grad():
        model.waveform_decoder.output_heads["EMG"][-1].bias.copy_(
            torch.linspace(-0.5, 0.5, 8)
        )
        eeg_probability_bias = model.waveform_decoder.probability_heads["EEG"][-1].bias
        eeg_probability_bias[:5].fill_(0.5)
    history = torch.randn(2, 3, 3, tiny_config().samples_per_epoch) * 0.1
    history[:, :, 1] = 0.0
    history[:, :, 1, 4] = 4.0
    history[:, :, 1, 20] = 4.0
    with torch.no_grad():
        output = model.rollout(history)
    generated = output["future_waveforms"]
    assert model.waveform_decoder.output_baseline == "none"
    assert torch.count_nonzero(generated[:, 0]) > 0
    assert torch.count_nonzero(generated[:, 1]) > 0
    assert torch.count_nonzero(generated[:, 2]) > 0
    assert not torch.equal(
        generated,
        history[:, -1, :, -tiny_config().sample_rate :],
    )


def test_time_aligned_ecg_renderer_is_differentiable_and_samples_non_ecg() -> None:
    w5 = build_model(
        True, refractory_ecg=True, physiological_renderer=True
    ).eval()
    model = build_model(
        True,
        refractory_ecg=True,
        physiological_renderer=True,
        time_aligned_ecg=True,
    )
    incompatible = model.load_state_dict(w5.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(
        key.startswith("waveform_decoder.ecg_event_timing_head.")
        for key in incompatible.missing_keys
    )
    history = torch.randn(3, 3, 3, tiny_config().samples_per_epoch)
    history[:, :, 1] = 0.0
    history[:, :, 1, 4] = 4.0
    history[:, :, 1, 20] = 4.0
    output = model.rollout(history)
    ecg = output["future_waveform_probabilities"]["ECG"]
    assert ecg["event_hazard_logits"].shape == (3, 2)
    assert ecg["event_offset_samples"].shape == (3, 2)
    assert ecg["event_amplitude"].shape == (3, 2)
    target = torch.randn_like(output["future_waveforms"])
    loss = ecg_event_alignment_loss(
        ecg,
        output["future_waveforms"],
        target,
        torch.ones(3, 3, dtype=torch.bool),
        modalities=("EEG", "ECG", "EMG"),
        sample_rate=tiny_config().sample_rate,
        horizons_seconds=(1,),
        patch_samples=8,
    )["loss"]
    loss.backward()
    timing_gradient = model.waveform_decoder.ecg_event_timing_head[-1].weight.grad
    point_gradient = model.waveform_decoder.ecg_point_process_head[-1].weight.grad
    assert torch.isfinite(loss)
    assert timing_gradient is not None
    assert torch.count_nonzero(timing_gradient) > 0
    assert point_gradient is not None
    assert torch.count_nonzero(point_gradient) > 0

    with torch.no_grad():
        context = model.rollout_context(history)
        draws = model.waveform_decoder.sample_waveform_distribution(
            history[:, -1],
            context["predicted_states"][:, 0],
            context["physiology_dynamics_states"][:, 0],
            num_samples=3,
        )
    assert draws.shape == (3, 3, 3, tiny_config().sample_rate)
    assert not torch.equal(draws[:, 0, 2], draws[:, 1, 2])


def test_recursive_ecg_renderer_uses_rr_sequence_and_retains_amplitude() -> None:
    model = build_model(
        True,
        refractory_ecg=True,
        physiological_renderer=True,
        time_aligned_ecg=True,
        recursive_ecg=True,
    ).eval()
    history = torch.randn(2, 3, 3, tiny_config().samples_per_epoch) * 0.1
    history[:, :, 1] = 0.0
    history[:, :, 1, 4] = 4.0
    history[:, :, 1, 20] = 4.0
    with torch.no_grad():
        model.waveform_decoder.ecg_event_timing_head[-1].bias[0] = 4.0
        output = model.rollout(history)
    generated_ecg = output["future_waveforms"][:, 1]
    assert torch.count_nonzero(generated_ecg) > 0
    assert generated_ecg.square().mean().sqrt() > 0.05


def test_recent_rr_residual_renderer_does_not_gate_beat_amplitude() -> None:
    model = build_model(
        True,
        refractory_ecg=True,
        physiological_renderer=True,
        time_aligned_ecg=True,
        recursive_ecg=True,
        recent_rr_residual=True,
    ).eval()
    history = torch.randn(2, 3, 3, tiny_config().samples_per_epoch) * 0.1
    history[:, :, 1] = 0.0
    history[:, :, 1, 4] = 4.0
    history[:, :, 1, 20] = 4.0
    with torch.no_grad():
        model.waveform_decoder.ecg_event_timing_head[-1].bias[0] = -10.0
        output = model.rollout(history)
    ecg_probability = output["future_waveform_probabilities"]["ECG"]
    assert "rr_residual_gate_logits" in ecg_probability
    assert torch.count_nonzero(output["future_waveforms"][:, 1]) > 0


def test_recent_ecg_rms_calibrates_recursive_template_energy() -> None:
    model = build_model(
        True,
        refractory_ecg=True,
        physiological_renderer=True,
        time_aligned_ecg=True,
        recursive_ecg=True,
        recent_rr_residual=True,
        amplitude_calibration=True,
    ).eval()
    history = torch.randn(2, 3, 3, tiny_config().samples_per_epoch) * 0.1
    history[:, :, 1] = 0.0
    history[:, :, 1, 4] = 4.0
    history[:, :, 1, 20] = 4.0
    with torch.no_grad():
        output = model.rollout(history)
    recent_rms = history[:, -1, 1, -tiny_config().sample_rate :].square().mean(-1).sqrt()
    generated_rms = output["future_waveforms"][:, 1].square().mean(-1).sqrt()
    assert torch.allclose(generated_rms, recent_rms, atol=1e-5, rtol=1e-4)


def test_safe_modality_refinement_starts_exactly_at_w5_and_receives_gradients() -> None:
    w5 = build_model(
        True, refractory_ecg=True, physiological_renderer=True
    ).eval()
    refined = build_model(
        True,
        refractory_ecg=True,
        physiological_renderer=True,
        safe_refinement=True,
    ).eval()
    incompatible = refined.load_state_dict(w5.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(
        key.startswith("waveform_decoder.safe_refinement_heads.")
        for key in incompatible.missing_keys
    )
    history = torch.randn(3, 3, 3, tiny_config().samples_per_epoch) * 0.1
    history[:, :, 1] = 0.0
    history[:, :, 1, 4] = 4.0
    history[:, :, 1, 20] = 4.0
    with torch.no_grad():
        w5_output = w5.rollout(history)
        refined_output = refined.rollout(history)
    assert torch.equal(
        refined_output["future_waveforms"], w5_output["future_waveforms"]
    )
    for modality in ("EEG", "ECG", "EMG"):
        base_probability = w5_output["future_waveform_probabilities"][modality]
        safe_probability = refined_output["future_waveform_probabilities"][modality]
        for name, value in base_probability.items():
            assert torch.equal(safe_probability[name], value)

    refined.train()
    output = refined.rollout(history)
    target = torch.randn_like(output["future_waveforms"])
    valid = torch.ones(3, 3, dtype=torch.bool)
    waveform_loss = refined.waveform_decoder.waveform_loss(
        output["future_waveforms"],
        target,
        valid,
        horizons_seconds=(1,),
        multi_resolution_fft_sizes=(4, 8, 16),
    )["loss"]
    probability_loss = probabilistic_waveform_loss(
        output["future_waveform_probabilities"],
        target,
        valid,
        modalities=("EEG", "ECG", "EMG"),
        sample_rate=tiny_config().sample_rate,
        horizons_seconds=(1,),
        patch_samples=8,
    )["loss"]
    (waveform_loss + probability_loss).backward()
    for modality in ("EEG", "ECG", "EMG"):
        gradient = refined.waveform_decoder.safe_refinement_heads[modality][
            -1
        ].weight.grad
        assert gradient is not None
        assert torch.count_nonzero(gradient) > 0


def test_probability_metrics_and_gate_report_calibration_contract() -> None:
    sample_rate = 16
    patch_samples = 8
    samples = 16
    predictions = {
        "EEG": {
            "spectral_mean": torch.randn(4, 2, 9),
            "spectral_scale": torch.ones(4, 2, 9),
        },
        "ECG": {
            "qrs_logits": torch.randn(4, samples),
            "rr_logits": torch.randn(4, 2, 8),
            "rr_mean_seconds": torch.ones(4, 2),
            "rr_scale_seconds": torch.ones(4, 2) * 0.2,
        },
        "EMG": {
            "envelope_mean": torch.rand(4, 2),
            "envelope_scale": torch.ones(4, 2) * 0.2,
            "rms_mean": torch.rand(4, 2),
            "rms_scale": torch.ones(4, 2) * 0.2,
            "burst_logits": torch.randn(4, 2),
        },
    }
    metrics = probabilistic_waveform_metrics(
        predictions,
        torch.randn(4, 3, samples),
        torch.randn(4, 3, samples),
        torch.ones(4, 3, dtype=torch.bool),
        modalities=("EEG", "ECG", "EMG"),
        sample_rate=sample_rate,
        horizons_seconds=(1,),
        patch_samples=patch_samples,
    )
    assert "qrs_brier" in metrics["by_horizon_seconds"]["1"]["ECG"]
    assert "rr_coverage_80" in metrics["by_horizon_seconds"]["1"]["ECG"]
    assert "spectral_crps" in metrics["by_horizon_seconds"]["1"]["EEG"]
    assert "envelope_coverage_80" in metrics["by_horizon_seconds"]["1"]["EMG"]

    waveform = {"all": {"mean_standardized_mae": 0.8}}
    baseline = {"all": {"mean_standardized_mae": 1.0}}
    probability = {
        "by_horizon_seconds": {
            str(horizon): {
                "EEG": {"spectral_mean_mae": 0.8, "baseline_spectral_mae": 1.0},
                "ECG": {
                    "qrs_brier": 0.1,
                    "baseline_qrs_brier": 0.2,
                    "qrs_event_f1": 0.6,
                    "baseline_qrs_event_f1": 0.5,
                    "rr_mean_mae_ms": 60.0,
                    "baseline_rr_mae_ms": 80.0,
                    "rr_coverage_80": 0.8,
                },
                "EMG": {"envelope_mean_mae": 0.8, "baseline_envelope_mae": 1.0},
            }
            for horizon in (1, 5, 10)
        }
    }
    gate = probabilistic_waveform_gate_result(waveform, baseline, probability)
    assert gate["passed"]

    strict_gate = refractory_ecg_gate_result(waveform, baseline, probability)
    assert strict_gate["passed"]
    assert strict_gate["qrs_event_f1_improved_horizons"] == ["1", "5", "10"]
