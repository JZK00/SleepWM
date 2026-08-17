import numpy as np

from uniphysio_wm.physio_features import FEATURE_NAMES, extract_record_features


def test_fixed_physio_features_are_finite_for_synthetic_record() -> None:
    sample_rate = 128
    seconds = 30
    time = np.arange(sample_rate * seconds) / sample_rate
    eeg = 20e-6 * np.sin(2.0 * np.pi * time) + 5e-6 * np.sin(2.0 * np.pi * 10.0 * time)
    ecg = np.zeros_like(time)
    for second in range(1, seconds):
        center = int(second * sample_rate)
        offsets = np.arange(-5, 6)
        ecg[center + offsets] += 1e-3 * np.exp(-0.5 * np.square(offsets / 1.5))
    rng = np.random.default_rng(9)
    emg = 5e-6 * rng.normal(size=len(time))
    signals = np.stack((eeg, ecg, emg), axis=0)[None, ...]

    features, valid = extract_record_features(signals, sample_rate)

    assert features.shape == valid.shape == (1, len(FEATURE_NAMES))
    assert valid.all()
    assert 55.0 <= features[0, FEATURE_NAMES.index("ecg_hr_bpm")] <= 65.0
    assert 900.0 <= features[0, FEATURE_NAMES.index("ecg_median_rr_ms")] <= 1100.0


def test_physio_feature_extractor_rejects_wrong_modality_count() -> None:
    signals = np.zeros((2, 2, 128), dtype=np.float32)
    try:
        extract_record_features(signals, sample_rate=128)
    except ValueError as error:
        assert "EEG/ECG/EMG" in str(error)
    else:
        raise AssertionError("expected a modality-count error")
