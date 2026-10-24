from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt
from scipy.ndimage import uniform_filter1d


FEATURE_NAMES = (
    "eeg_log_delta_power",
    "eeg_log_theta_power",
    "eeg_log_alpha_power",
    "eeg_log_beta_power",
    "eeg_spectral_centroid_hz",
    "ecg_hr_bpm",
    "ecg_median_rr_ms",
    "ecg_rmssd_ms",
    "emg_log_rms",
    "emg_log_mean_rectified",
    "emg_high_frequency_ratio",
)

FEATURE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "EEG": FEATURE_NAMES[:5],
    "ECG": FEATURE_NAMES[5:8],
    "EMG": FEATURE_NAMES[8:],
}

FEATURE_UNITS = {
    "eeg_log_delta_power": "log10(V^2)",
    "eeg_log_theta_power": "log10(V^2)",
    "eeg_log_alpha_power": "log10(V^2)",
    "eeg_log_beta_power": "log10(V^2)",
    "eeg_spectral_centroid_hz": "Hz",
    "ecg_hr_bpm": "beats/min",
    "ecg_median_rr_ms": "ms",
    "ecg_rmssd_ms": "ms",
    "emg_log_rms": "log10(V)",
    "emg_log_mean_rectified": "log10(V)",
    "emg_high_frequency_ratio": "ratio",
}


def _power_spectrum(signals: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(signals, dtype=np.float64)
    values = values - values.mean(axis=1, keepdims=True)
    window = np.hanning(values.shape[1])
    transformed = np.fft.rfft(values * window[None, :], axis=1)
    power = np.abs(transformed) ** 2 / max(float(sample_rate) * np.square(window).sum(), 1e-12)
    frequencies = np.fft.rfftfreq(values.shape[1], d=1.0 / sample_rate)
    return frequencies, power


def _band_power(frequencies: np.ndarray, power: np.ndarray, low: float, high: float) -> np.ndarray:
    selected = (frequencies >= low) & (frequencies < high)
    if selected.sum() < 2:
        raise ValueError(f"frequency band {low}-{high} Hz has fewer than two bins")
    return np.trapz(power[:, selected], frequencies[selected], axis=1)


def eeg_features(signals: np.ndarray, sample_rate: int) -> np.ndarray:
    frequencies, power = _power_spectrum(signals, sample_rate)
    bands = tuple(
        _band_power(frequencies, power, low, high)
        for low, high in ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0))
    )
    selected = (frequencies >= 0.5) & (frequencies < 30.0)
    selected_power = power[:, selected]
    centroid = (selected_power * frequencies[selected][None, :]).sum(axis=1)
    centroid /= selected_power.sum(axis=1).clip(min=1e-24)
    return np.column_stack([*(np.log10(values.clip(min=1e-24)) for values in bands), centroid])


def ecg_interval_features(signal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(signal, dtype=np.float64)
    values = values - np.median(values)
    sos = butter(2, (5.0, 20.0), btype="bandpass", fs=sample_rate, output="sos")
    filtered = sosfiltfilt(sos, values)
    derivative_energy = np.square(np.diff(filtered, prepend=filtered[0]))
    integrated = uniform_filter1d(
        derivative_energy,
        size=max(3, int(round(0.12 * sample_rate))),
        mode="nearest",
    )
    prominence = max(float(np.std(integrated)) * 0.5, np.finfo(np.float64).eps)
    peaks, _ = find_peaks(
        integrated,
        distance=max(1, int(round(0.30 * sample_rate))),
        prominence=prominence,
    )
    intervals = np.diff(peaks) / float(sample_rate)
    intervals = intervals[(intervals >= 0.30) & (intervals <= 2.0)]
    features = np.full(3, np.nan, dtype=np.float64)
    valid = np.zeros(3, dtype=bool)
    if len(intervals) >= 2:
        median_rr = float(np.median(intervals))
        heart_rate = 60.0 / median_rr
        if 30.0 <= heart_rate <= 200.0:
            features[:2] = (heart_rate, 1000.0 * median_rr)
            valid[:2] = True
    if valid[0] and len(intervals) >= 3:
        features[2] = 1000.0 * float(np.sqrt(np.mean(np.square(np.diff(intervals)))))
        valid[2] = True
    return features, valid


def emg_features(signals: np.ndarray, sample_rate: int) -> np.ndarray:
    values = np.asarray(signals, dtype=np.float64)
    centered = values - np.median(values, axis=1, keepdims=True)
    rms = np.sqrt(np.mean(np.square(centered), axis=1))
    mean_rectified = np.mean(np.abs(centered), axis=1)
    frequencies, power = _power_spectrum(centered, sample_rate)
    high = _band_power(frequencies, power, 20.0, min(45.0, 0.49 * sample_rate))
    total = _band_power(frequencies, power, 5.0, min(45.0, 0.49 * sample_rate))
    return np.column_stack(
        (
            np.log10(rms.clip(min=1e-24)),
            np.log10(mean_rectified.clip(min=1e-24)),
            high / total.clip(min=1e-24),
        )
    )


def extract_record_features(
    signals: np.ndarray,
    sample_rate: int,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract fixed EEG/ECG/EMG epoch features from [epochs, 3, samples]."""

    values = np.asarray(signals)
    if values.ndim != 3 or values.shape[1] != 3:
        raise ValueError("signals must be [epochs, 3, samples] in EEG/ECG/EMG order")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    features = np.full((values.shape[0], len(FEATURE_NAMES)), np.nan, dtype=np.float64)
    valid = np.zeros_like(features, dtype=bool)
    for start in range(0, values.shape[0], chunk_size):
        stop = min(values.shape[0], start + chunk_size)
        features[start:stop, :5] = eeg_features(values[start:stop, 0], sample_rate)
        features[start:stop, 8:] = emg_features(values[start:stop, 2], sample_rate)
        valid[start:stop, :5] = np.isfinite(features[start:stop, :5])
        valid[start:stop, 8:] = np.isfinite(features[start:stop, 8:])
    for epoch_index in range(values.shape[0]):
        ecg_values, ecg_valid = ecg_interval_features(values[epoch_index, 1], sample_rate)
        features[epoch_index, 5:8] = ecg_values
        valid[epoch_index, 5:8] = ecg_valid
    return features.astype(np.float32), valid
