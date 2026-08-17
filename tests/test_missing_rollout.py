import torch

from uniphysio_wm.missing_rollout import (
    forced_history_view,
    missing_rollout_gate_result,
    missing_rollout_summary,
    sampled_history_view,
)
from uniphysio_wm.recursive_rollout import recursive_rollout_gate_result


def test_forced_history_view_physically_removes_unavailable_modalities() -> None:
    signals = torch.arange(2 * 3 * 3 * 4, dtype=torch.float32).reshape(2, 3, 3, 4)
    present = torch.ones(2, 3, 3, dtype=torch.bool)
    present[0, 1, 0] = False
    forced, forced_present = forced_history_view(
        signals, present, ("EEG", "EMG"), ("EEG", "ECG", "EMG")
    )
    assert not forced_present[..., 1].any()
    assert torch.count_nonzero(forced[..., 1, :]) == 0
    assert torch.count_nonzero(forced[0, 1, 0]) == 0
    assert torch.equal(forced[1, :, 0], signals[1, :, 0])
    assert torch.equal(forced[..., 2, :], signals[..., 2, :])


def _result(stage: float, physiology: float, waveform: float, scale: float) -> dict:
    return {
        "stage": {"all_horizons": {"macro_f1": stage}},
        "future_physiology": {
            "all_features": {"mean_normalized_mae": physiology}
        },
        "waveform": {"all": {"mean_standardized_mae": waveform}},
        "uncertainty": {
            modality: {"mean_scale": scale} for modality in ("EEG", "ECG", "EMG")
        },
    }


def test_missing_rollout_summary_and_gate_use_full_reference() -> None:
    modalities = ("EEG", "ECG", "EMG")
    results = {
        "EEG": _result(0.56, 0.50, 0.50, 1.2),
        "ECG": _result(0.55, 0.51, 0.51, 1.2),
        "EMG": _result(0.54, 0.52, 0.52, 1.2),
        "EEG+ECG": _result(0.58, 0.48, 0.48, 1.1),
        "EEG+EMG": _result(0.58, 0.48, 0.48, 1.1),
        "ECG+EMG": _result(0.57, 0.49, 0.49, 1.1),
        "EEG+ECG+EMG": _result(0.60, 0.45, 0.45, 1.0),
    }
    summary = missing_rollout_summary(results, modalities)
    gate = missing_rollout_gate_result(summary)
    assert summary["full_modality"]["stage_macro_f1"] == 0.60
    assert set(gate["uncertainty_increased_modalities"]) == set(modalities)
    assert gate["passed"]


def test_sampled_history_view_full_batches_preserve_natural_availability() -> None:
    signals = torch.randn(4, 3, 3, 8)
    present = torch.ones(4, 3, 3, dtype=torch.bool)
    present[0, :, 2] = False
    sampled, sampled_present, selected = sampled_history_view(signals, present, 1.0)
    assert torch.equal(sampled_present, present)
    assert torch.equal(selected, present.all(dim=1))
    assert torch.equal(sampled[present], signals[present])
    assert torch.count_nonzero(sampled[~present]) == 0


def test_recursive_rollout_gate_requires_latent_and_downstream_retention() -> None:
    full_result = {
        "stage": {"all_horizons": {"macro_f1": 0.60}},
        "future_physiology": {
            "by_horizon": {
                "1": {"mean_normalized_mae": 0.40},
                "2": {"mean_normalized_mae": 0.41},
                "4": {"mean_normalized_mae": 0.42},
            }
        },
        "waveform": {"all": {"mean_standardized_mae": 0.44}},
        "recursive_comparison": {
            "latent": {
                "relative_smooth_l1_improvement": 0.03,
                "correction_rms": 0.01,
            },
            "direct_stage": {"all_horizons": {"macro_f1": 0.605}},
            "direct_future_physiology": {
                "by_horizon": {
                    "1": {"mean_normalized_mae": 0.42},
                    "2": {"mean_normalized_mae": 0.43},
                    "4": {"mean_normalized_mae": 0.44},
                }
            },
            "direct_waveform": {"all": {"mean_standardized_mae": 0.44}},
        },
    }
    missing_summary = {
        "full_modality": {"stage_macro_f1": 0.60},
        "nonfull_mean_stage_macro_f1_drop": 0.13,
    }
    assert recursive_rollout_gate_result(full_result, missing_summary)["passed"]
