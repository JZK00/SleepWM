import torch

from uniphysio_wm.physiology_realism import (
    comprehensive_physiology_metrics,
    physiology_report_summary,
)
from uniphysio_wm.probabilistic_waveform import _eeg_targets, _emg_targets
from uniphysio_wm.renderer_probability_calibration import (
    evaluate_renderer_probability_calibration,
    fit_renderer_probability_calibration,
)


def _synthetic_bundle():
    torch.manual_seed(17)
    records = 32
    sample_rate = 16
    samples = 32
    patch_samples = 8
    target = 0.05 * torch.randn(records, 3, samples)
    for row in range(records):
        for position in (3, 11, 19, 27):
            target[row, 1, position] += 3.0
        amplitudes = torch.tensor((0.15, 0.35, 1.10, 0.25)).repeat_interleave(patch_samples)
        target[row, 2] += amplitudes * torch.randn(samples)
    generated = target + 0.04 * torch.randn_like(target)
    generated[:, 1] = torch.roll(target[:, 1], shifts=1, dims=-1)
    recent = torch.roll(target, shifts=3, dims=-1)
    valid = torch.ones(records, 3, dtype=torch.bool)
    eeg_mean = _eeg_targets(target[:, 0], patch_samples, sample_rate)
    envelope, rms, burst = _emg_targets(target[:, 2], patch_samples, sample_rate)
    probabilities = {
        "EEG": {
            "spectral_mean": eeg_mean + 0.02 * torch.randn_like(eeg_mean),
            "spectral_scale": torch.full_like(eeg_mean, 0.1),
        },
        "ECG": {
            "qrs_logits": torch.randn(records, samples),
            "rr_logits": torch.randn(records, samples // patch_samples, 12),
            "rr_mean_seconds": torch.full((records, samples // patch_samples), 0.5),
            "rr_scale_seconds": torch.full((records, samples // patch_samples), 0.1),
        },
        "EMG": {
            "envelope_mean": envelope + 0.05 * torch.randn_like(envelope),
            "envelope_scale": torch.full_like(envelope, 0.1),
            "rms_mean": rms + 0.05 * torch.randn_like(rms),
            "rms_scale": torch.full_like(rms, 0.1),
            "burst_logits": torch.where(burst, torch.tensor(1.0), torch.tensor(-1.0)),
        },
    }
    return generated, probabilities, recent, target, valid, sample_rate, patch_samples


def test_renderer_probability_calibration_reports_all_modalities() -> None:
    generated, probabilities, recent, target, valid, sample_rate, patch_samples = (
        _synthetic_bundle()
    )
    calibration = fit_renderer_probability_calibration(
        generated,
        probabilities,
        recent,
        target,
        valid,
        ("EEG", "ECG", "EMG"),
        sample_rate,
        (2,),
        patch_samples,
    )
    metrics = evaluate_renderer_probability_calibration(
        calibration,
        generated,
        probabilities,
        recent,
        target,
        valid,
        ("EEG", "ECG", "EMG"),
        sample_rate,
        (2,),
        patch_samples,
        seed=20260804,
        sample_count=64,
    )
    horizon = metrics["by_horizon_seconds"]["2"]
    assert set(horizon) == {"EEG", "ECG", "EMG"}
    assert 0.0 <= horizon["EEG"]["coverage_80"] <= 1.0
    assert 0.0 <= horizon["EEG"]["sample64_coverage_80"] <= 1.0
    assert horizon["ECG"]["rr_calibrated_mae_ms"] >= 0.0
    assert 0.0 <= horizon["EMG"]["burst_f1"] <= 1.0


def test_comprehensive_physiology_report_includes_coupling_and_hrv() -> None:
    generated, _, recent, target, valid, sample_rate, patch_samples = _synthetic_bundle()
    metrics = comprehensive_physiology_metrics(
        generated,
        target,
        recent,
        valid,
        ("EEG", "ECG", "EMG"),
        sample_rate,
        (2,),
        patch_samples,
    )
    horizon = metrics["by_horizon_seconds"]["2"]
    assert "band_log_power_mae" in horizon["EEG"]
    assert "rmssd_mae_ms" in horizon["ECG"]
    assert "beat_morphology_correlation" in horizon["ECG"]
    assert "burst_f1" in horizon["EMG"]
    assert set(horizon["cross_modal_coupling"]["by_pair"]) == {
        "EEG-ECG",
        "EEG-EMG",
        "ECG-EMG",
    }
    summary = physiology_report_summary(metrics)
    assert summary["required_metric_families_complete"]
