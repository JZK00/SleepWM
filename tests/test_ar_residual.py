import torch

from uniphysio_wm.ar_residual import (
    ARAnchoredCrossModalResidual,
    fit_feature_ar,
)


def test_feature_ar_recovers_horizon_specific_linear_dynamics() -> None:
    generator = torch.Generator().manual_seed(7)
    history = torch.randn(256, 4, 2, generator=generator)
    target = torch.stack(
        (
            0.5 * history[:, -1] - 0.2 * history[:, -2],
            -0.3 * history[:, -1] + 0.4 * history[:, 0],
        ),
        dim=1,
    )
    valid = torch.ones_like(target, dtype=torch.bool)
    model = fit_feature_ar(history, target, valid, alpha=1e-6)
    prediction = model(history)
    assert torch.max(torch.abs(prediction - target)) < 1e-4


def test_zero_residual_uses_ar_and_missing_input_falls_back_exactly() -> None:
    history = torch.randn(32, 4, 2)
    target = torch.randn(32, 2, 2)
    valid = torch.ones_like(target, dtype=torch.bool)
    ar_model = fit_feature_ar(history, target, valid)
    model = ARAnchoredCrossModalResidual(
        ar_model,
        state_dim=8,
        modality_count=3,
        hidden_dim=16,
        dropout=0.0,
    ).eval()
    states = torch.randn(32, 2, 8)
    reliability = torch.rand(32, 3)
    availability = torch.ones(32, 3)
    base = torch.randn(32, 2, 2)
    output = model(history, torch.ones_like(history, dtype=torch.bool), states, reliability, availability, base)
    assert torch.equal(output["future_physiology"], output["ar_future_physiology"])
    assert torch.count_nonzero(output["cross_modal_residual"]) == 0

    availability[:, 1] = 0.0
    fallback = model(history, torch.ones_like(history, dtype=torch.bool), states, reliability, availability, base)
    assert torch.equal(fallback["future_physiology"], base)
