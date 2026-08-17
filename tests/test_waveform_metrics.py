import torch

from uniphysio_wm.waveform_metrics import (
    structured_head_metrics,
    structured_waveform_gate_result,
    waveform_forecast_metrics,
    waveform_gate_result,
)


def test_waveform_metrics_report_modalities_and_horizons() -> None:
    target = torch.randn(4, 3, 20)
    prediction = target + 0.1 * torch.randn_like(target)
    metrics = waveform_forecast_metrics(
        prediction,
        target,
        torch.ones(4, 3, dtype=torch.bool),
        modalities=("EEG", "ECG", "EMG"),
        sample_rate=10,
        horizons_seconds=(1, 2),
    )
    assert set(metrics["by_horizon_seconds"]) == {"1", "2"}
    assert set(metrics["by_modality"]) == {"EEG", "ECG", "EMG"}
    assert metrics["all"]["mean_standardized_mae"] > 0.0
    assert "qrs_event_f1" in metrics["by_horizon_seconds"]["1"]["by_modality"]["ECG"]
    assert "envelope_correlation" in metrics["by_horizon_seconds"]["1"]["by_modality"]["EMG"]


def test_waveform_gate_requires_overall_and_two_modality_improvements() -> None:
    def result(mae, eeg, ecg, emg):
        return {
            "all": {"mean_standardized_mae": mae},
            "by_horizon_seconds": {
                str(horizon): {
                    "by_modality": {
                        "EEG": {"log_spectral_mae": eeg},
                        "ECG": {"qrs_event_f1": ecg},
                        "EMG": {"envelope_correlation": emg},
                    }
                }
                for horizon in (1, 5, 10)
            },
        }

    gate = waveform_gate_result(
        result(0.90, 0.8, 0.7, 0.6),
        result(1.00, 1.0, 0.5, 0.4),
        required_relative_mae_improvement=0.02,
    )
    assert gate["passed"]
    assert set(gate["modalities_improved_on_at_least_two_horizons"]) == {
        "EEG",
        "ECG",
        "EMG",
    }


def test_structured_waveform_gate_requires_ecg_and_another_modality() -> None:
    def result(mae, eeg, qrs, rr, emg):
        return {
            "all": {"mean_standardized_mae": mae},
            "by_horizon_seconds": {
                str(horizon): {
                    "by_modality": {
                        "EEG": {"log_spectral_mae": eeg},
                        "ECG": {
                            "qrs_event_f1": qrs,
                            "rr_interval_mae_ms": rr,
                        },
                        "EMG": {"envelope_correlation": emg},
                    }
                }
                for horizon in (1, 5, 10)
            },
        }

    gate = structured_waveform_gate_result(
        result(0.90, 0.8, 0.4, 80.0, 0.3),
        result(1.00, 1.0, 0.5, 100.0, 0.2),
    )
    assert gate["passed"]
    assert gate["modalities_improved_on_at_least_two_horizons"] == [
        "EEG",
        "ECG",
        "EMG",
    ]


def test_structured_head_metrics_report_explicit_physiology_outputs() -> None:
    sample_rate = 16
    patch_samples = 8
    samples = 16
    prediction = {
        "EEG": {
            "log_spectrum": torch.randn(4, 2, 5),
            "band_log_power": torch.randn(4, 2, 4),
        },
        "ECG": {
            "qrs_logits": torch.randn(4, samples),
            "rr_seconds": torch.rand(4, 2) + 0.25,
        },
        "EMG": {
            "envelope": torch.rand(4, 2),
            "rms": torch.rand(4, 2),
            "burst_logits": torch.randn(4, 2),
        },
    }
    metrics = structured_head_metrics(
        prediction,
        torch.randn(4, 3, samples),
        torch.ones(4, 3, dtype=torch.bool),
        sample_rate,
        horizons_seconds=(1,),
        patch_samples=patch_samples,
    )
    horizon = metrics["by_horizon_seconds"]["1"]
    assert "head_log_spectral_mae" in horizon["EEG"]
    assert "head_qrs_event_f1" in horizon["ECG"]
    assert "head_envelope_correlation" in horizon["EMG"]
