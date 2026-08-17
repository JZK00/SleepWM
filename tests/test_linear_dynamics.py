import numpy as np

from uniphysio_wm.linear_dynamics import (
    fit_ridge_map,
    fit_state_score_ar,
    fit_state_score_var,
    predict_state_score_ar,
    predict_state_score_var,
    rollout_linear_state,
)


def test_ridge_map_recovers_linear_multioutput_relation() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(500, 3))
    weights = np.asarray([[1.2, -0.4], [0.3, 0.8], [-0.7, 0.2]])
    targets = features @ weights + np.asarray([0.5, -1.0])
    model = fit_ridge_map(features, targets, alpha=1e-6)
    assert np.allclose(model.predict(features), targets, atol=1e-5)


def test_ar_and_var_predictions_keep_forecast_shape() -> None:
    rng = np.random.default_rng(11)
    history = rng.normal(size=(200, 4, 3))
    future = np.stack(
        (0.5 * history[:, -1] + 0.2 * history[:, -2], history[:, -1] - history[:, 0]),
        axis=1,
    )
    ar = fit_state_score_ar(history, future, alpha=1e-4)
    var = fit_state_score_var(history, future, alpha=1e-4)
    assert predict_state_score_ar(ar, history).shape == future.shape
    assert predict_state_score_var(var, history).shape == future.shape
    assert np.allclose(predict_state_score_var(var, history), future, atol=1e-4)


def test_linear_state_rollout_uses_requested_horizons() -> None:
    rng = np.random.default_rng(17)
    current = rng.normal(size=(300, 2))
    next_state = current * 0.5 + 1.0
    transition = fit_ridge_map(current, next_state, alpha=1e-6)
    rollout = rollout_linear_state(transition, current[:5], (1, 2, 4))
    assert rollout.shape == (5, 3, 2)
    assert np.allclose(rollout[:, 0], current[:5] * 0.5 + 1.0, atol=1e-5)
    assert np.allclose(rollout[:, 1], current[:5] * 0.25 + 1.5, atol=1e-5)
