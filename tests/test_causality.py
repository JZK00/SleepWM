import torch
import pytest

from train_forecast import build_optimizer
from uniphysio_wm.models import CausalPhysioWorldModel, MultiModalEncoder, ObservationConfig


def test_future_targets_cannot_change_rollout() -> None:
    torch.manual_seed(7)
    config = ObservationConfig(
        sample_rate=8,
        epoch_seconds=2,
        patch_samples=4,
        d_model=16,
        layers=1,
        heads=4,
        dropout=0.0,
    )
    model = CausalPhysioWorldModel(
        MultiModalEncoder(config),
        horizons=(1, 2),
        transition_layers=1,
        transition_heads=4,
        dropout=0.0,
    ).eval()
    history = torch.randn(2, 3, 3, config.samples_per_epoch)
    future_a = torch.randn(2, 2, 3, config.samples_per_epoch)
    future_b = future_a + 100.0
    labels = torch.zeros(2, 2, dtype=torch.long)
    with torch.no_grad():
        output_a = model(history, future_signals=future_a, future_labels=labels)
        output_b = model(history, future_signals=future_b, future_labels=labels)
    assert torch.equal(output_a["predicted_states"], output_b["predicted_states"])
    assert not torch.equal(output_a["target_states"], output_b["target_states"])


def test_frozen_forecast_encoder_stays_in_evaluation_mode() -> None:
    config = ObservationConfig(
        sample_rate=8,
        epoch_seconds=2,
        patch_samples=4,
        d_model=16,
        layers=1,
        heads=4,
        dropout=0.2,
    )
    model = CausalPhysioWorldModel(
        MultiModalEncoder(config),
        horizons=(1,),
        transition_layers=1,
        transition_heads=4,
        freeze_observation_encoder=True,
    )
    model.train()
    assert not model.encoder.training
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())


def test_stage_residual_starts_from_current_stage_prediction() -> None:
    config = ObservationConfig(
        sample_rate=8,
        epoch_seconds=2,
        patch_samples=4,
        d_model=16,
        layers=1,
        heads=4,
        dropout=0.0,
    )
    model = CausalPhysioWorldModel(
        MultiModalEncoder(config),
        horizons=(1, 2),
        transition_layers=1,
        transition_heads=4,
        dropout=0.0,
        stage_residual_from_current=True,
    ).eval()
    history = torch.randn(2, 3, 3, config.samples_per_epoch)
    with torch.no_grad():
        output = model.rollout(history)
    expected = output["current_stage_logits"].unsqueeze(1).expand_as(output["stage_logits"])
    assert torch.equal(output["stage_logits"], expected)


def test_student_teacher_target_encoder_is_frozen_and_stable() -> None:
    torch.manual_seed(11)
    config = ObservationConfig(
        sample_rate=8,
        epoch_seconds=2,
        patch_samples=4,
        d_model=16,
        layers=1,
        heads=4,
        dropout=0.0,
    )
    model = CausalPhysioWorldModel(
        MultiModalEncoder(config),
        horizons=(1,),
        transition_layers=1,
        transition_heads=4,
        dropout=0.0,
        use_frozen_target_encoder=True,
    )
    model.train()
    assert model.encoder.training
    assert model.target_encoder is not None
    assert not model.target_encoder.training
    assert any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.target_encoder.parameters())

    target_before = {key: value.detach().clone() for key, value in model.target_encoder.state_dict().items()}
    history = torch.randn(2, 3, 3, config.samples_per_epoch)
    future = torch.randn(2, 1, 3, config.samples_per_epoch)
    output = model(history, future_signals=future)
    optimizer = torch.optim.AdamW(parameter for parameter in model.parameters() if parameter.requires_grad)
    optimizer.zero_grad(set_to_none=True)
    output["loss"].backward()
    optimizer.step()

    assert not output["target_states"].requires_grad
    for key, value in model.target_encoder.state_dict().items():
        assert torch.equal(value, target_before[key])


def test_student_encoder_uses_lower_learning_rate() -> None:
    config = ObservationConfig(
        sample_rate=8,
        epoch_seconds=2,
        patch_samples=4,
        d_model=16,
        layers=1,
        heads=4,
    )
    model = CausalPhysioWorldModel(
        MultiModalEncoder(config),
        horizons=(1,),
        transition_layers=1,
        transition_heads=4,
        use_frozen_target_encoder=True,
    )
    optimizer = build_optimizer(
        model,
        {
            "train": {"learning_rate": 1e-4, "weight_decay": 0.01},
            "forecast": {"encoder_learning_rate": 1e-5},
        },
    )
    assert sorted(group["lr"] for group in optimizer.param_groups) == [1e-5, 1e-4]


def test_transition_stage_weight_matches_manual_weighted_loss() -> None:
    torch.manual_seed(13)
    config = ObservationConfig(
        sample_rate=8,
        epoch_seconds=2,
        patch_samples=4,
        d_model=16,
        layers=1,
        heads=4,
        dropout=0.0,
    )
    model = CausalPhysioWorldModel(
        MultiModalEncoder(config),
        horizons=(1, 2),
        transition_layers=1,
        transition_heads=4,
        dropout=0.0,
    ).eval()
    history = torch.randn(2, 3, 3, config.samples_per_epoch)
    history_labels = torch.tensor([[0, 0, 0], [1, 1, 1]])
    future_labels = torch.tensor([[0, 2], [3, 1]])
    output = model(
        history,
        future_labels=future_labels,
        history_labels=history_labels,
        latent_weight=0.0,
        transition_stage_weight=2.0,
    )
    element_loss = torch.nn.functional.cross_entropy(
        output["stage_logits"].reshape(-1, 5),
        future_labels.reshape(-1),
        reduction="none",
    ).reshape_as(future_labels)
    weights = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    expected = (element_loss * weights).sum() / weights.sum()
    assert torch.allclose(output["stage_loss"], expected)


def test_factorized_transition_outputs_normalized_future_distribution() -> None:
    config = ObservationConfig(
        sample_rate=8,
        epoch_seconds=2,
        patch_samples=4,
        d_model=16,
        layers=1,
        heads=4,
        dropout=0.0,
    )
    model = CausalPhysioWorldModel(
        MultiModalEncoder(config),
        horizons=(1, 2),
        transition_layers=1,
        transition_heads=4,
        dropout=0.0,
        stage_residual_from_current=True,
        factorized_transition_head=True,
        change_prior_probabilities=(0.1, 0.3),
    ).eval()
    history = torch.randn(2, 3, 3, config.samples_per_epoch)
    with torch.no_grad():
        output = model.rollout(history)
    assert output["change_logits"].shape == (2, 2)
    assert output["destination_logits"].shape == (2, 2, 5)
    assert torch.allclose(output["stage_logits"].exp().sum(dim=-1), torch.ones(2, 2), atol=1e-6)
    assert torch.allclose(model.change_horizon_bias.sigmoid(), torch.tensor([0.1, 0.3]))


def test_factorized_transition_losses_are_finite() -> None:
    config = ObservationConfig(
        sample_rate=8,
        epoch_seconds=2,
        patch_samples=4,
        d_model=16,
        layers=1,
        heads=4,
        dropout=0.0,
    )
    model = CausalPhysioWorldModel(
        MultiModalEncoder(config),
        horizons=(1, 2),
        transition_layers=1,
        transition_heads=4,
        dropout=0.0,
        stage_residual_from_current=True,
        factorized_transition_head=True,
    ).eval()
    history = torch.randn(2, 3, 3, config.samples_per_epoch)
    history_labels = torch.tensor([[0, 0, 0], [1, 1, 1]])
    future_labels = torch.tensor([[0, 2], [3, 1]])
    output = model(
        history,
        future_labels=future_labels,
        history_labels=history_labels,
        change_loss_weight=0.5,
        destination_loss_weight=0.5,
    )
    assert torch.isfinite(output["change_loss"])
    assert torch.isfinite(output["destination_loss"])
    assert torch.isfinite(output["loss"])

    with pytest.raises(ValueError, match="requires stage_residual_from_current"):
        CausalPhysioWorldModel(
            MultiModalEncoder(config),
            horizons=(1,),
            factorized_transition_head=True,
        )
