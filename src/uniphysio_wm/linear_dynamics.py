from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class RidgeMap:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    coefficients: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        normalized = (values - self.x_mean) / self.x_scale
        return normalized @ self.coefficients + self.y_mean


def fit_ridge_map(features: np.ndarray, targets: np.ndarray, alpha: float = 1.0) -> RidgeMap:
    """Fit a centered multi-output ridge map without validation-set selection."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("features and targets must be aligned two-dimensional arrays")
    if x.shape[0] < 2 or x.shape[1] < 1 or y.shape[1] < 1:
        raise ValueError("ridge fitting requires at least two samples and one feature")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    x_mean = x.mean(axis=0)
    x_scale = x.std(axis=0)
    x_scale[x_scale < 1e-8] = 1.0
    y_mean = y.mean(axis=0)
    normalized = (x - x_mean) / x_scale
    centered_targets = y - y_mean
    gram = normalized.T @ normalized
    gram.flat[:: gram.shape[0] + 1] += float(alpha)
    coefficients = np.linalg.solve(gram, normalized.T @ centered_targets)
    return RidgeMap(x_mean, x_scale, y_mean, coefficients)


def fit_state_score_ar(
    history_scores: np.ndarray,
    future_scores: np.ndarray,
    alpha: float = 1.0,
) -> Tuple[Tuple[RidgeMap, ...], ...]:
    """Fit direct horizon-specific AR models independently for each class score."""

    history = np.asarray(history_scores)
    future = np.asarray(future_scores)
    if history.ndim != 3 or future.ndim != 3:
        raise ValueError("history and future scores must be [samples, time, classes]")
    if history.shape[0] != future.shape[0] or history.shape[2] != future.shape[2]:
        raise ValueError("history and future score shapes are incompatible")
    return tuple(
        tuple(
            fit_ridge_map(history[:, :, class_index], future[:, horizon_index, class_index : class_index + 1], alpha)
            for class_index in range(history.shape[2])
        )
        for horizon_index in range(future.shape[1])
    )


def predict_state_score_ar(
    models: Sequence[Sequence[RidgeMap]],
    history_scores: np.ndarray,
) -> np.ndarray:
    history = np.asarray(history_scores)
    if history.ndim != 3:
        raise ValueError("history scores must be [samples, time, classes]")
    prediction = np.empty((history.shape[0], len(models), history.shape[2]), dtype=np.float64)
    for horizon_index, horizon_models in enumerate(models):
        if len(horizon_models) != history.shape[2]:
            raise ValueError("AR model class count does not match history scores")
        for class_index, model in enumerate(horizon_models):
            prediction[:, horizon_index, class_index] = model.predict(
                history[:, :, class_index]
            )[:, 0]
    return prediction


def fit_state_score_var(
    history_scores: np.ndarray,
    future_scores: np.ndarray,
    alpha: float = 1.0,
) -> Tuple[RidgeMap, ...]:
    """Fit direct horizon-specific VAR models over all lagged class scores."""

    history = np.asarray(history_scores)
    future = np.asarray(future_scores)
    if history.ndim != 3 or future.ndim != 3:
        raise ValueError("history and future scores must be [samples, time, classes]")
    if history.shape[0] != future.shape[0] or history.shape[2] != future.shape[2]:
        raise ValueError("history and future score shapes are incompatible")
    features = history.reshape(history.shape[0], -1)
    return tuple(
        fit_ridge_map(features, future[:, horizon_index], alpha)
        for horizon_index in range(future.shape[1])
    )


def predict_state_score_var(
    models: Sequence[RidgeMap],
    history_scores: np.ndarray,
) -> np.ndarray:
    history = np.asarray(history_scores)
    if history.ndim != 3:
        raise ValueError("history scores must be [samples, time, classes]")
    features = history.reshape(history.shape[0], -1)
    return np.stack([model.predict(features) for model in models], axis=1)


def rollout_linear_state(
    transition: RidgeMap,
    current_states: np.ndarray,
    horizons: Sequence[int],
) -> np.ndarray:
    requested = tuple(int(value) for value in horizons)
    if not requested or min(requested) < 1 or tuple(sorted(set(requested))) != requested:
        raise ValueError("horizons must be sorted unique positive integers")
    state = np.asarray(current_states, dtype=np.float64)
    if state.ndim != 2:
        raise ValueError("current states must be [samples, features]")
    predictions = []
    requested_set = set(requested)
    for step in range(1, max(requested) + 1):
        state = transition.predict(state)
        if step in requested_set:
            predictions.append(state.copy())
    return np.stack(predictions, axis=1)
