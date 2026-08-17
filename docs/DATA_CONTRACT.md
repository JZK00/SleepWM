# Data Contract

## Manifest

Each row represents one continuous PSG recording.

Required columns:

| Field | Meaning |
| --- | --- |
| `record_id` | Stable recording identifier |
| `subject` | Stable subject identifier used for leakage checks |
| `split` | One of `train`, `val`, or `test` |
| `npz_path` | Absolute path or path relative to the manifest |

Recommended publication fields include `dataset`, `scene`, `device`,
`available_modalities`, `eeg_location`, `ecg_location`, and `emg_location`.
Sensor location must be explicit because sleep chin EMG and driving facial,
neck, or forearm EMG do not have identical semantics.

## Processed recording

Each NPZ contains:

| Key | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `signals` | `[epochs, modalities, samples]` | `float32` | Synchronized 30-second epochs |
| `labels` | `[epochs]` | `int64` | Sleep-stage labels |
| `modality_present` | `[epochs, modalities]` | `bool` | Optional natural availability mask |

The stored modality order is `EEG`, `ECG`, `EMG`. Experiments may select a
subset but must not silently reorder the arrays.

Sleep labels use `W=0`, `N1=1`, `N2=2`, `N3=3`, and `REM=4`. Unknown,
movement, and unusable epochs must be excluded during preprocessing rather than
mapped to a valid stage.

## Split and leakage rules

1. A subject may occur in exactly one split.
2. Split assignment is fixed before epoch sampling or class balancing.
3. Validation and test subjects do not contribute normalization statistics.
4. Forecasting normalization must not use future samples from the same record.
5. ISRUC Cohort I is a sealed external test set and cannot be used for early
   stopping, threshold selection, normalization, or hyperparameter tuning.
6. Naturally missing modalities and modalities hidden for training must be
   stored as separate masks.

## Dataset roles

| Dataset | Initial role |
| --- | --- |
| HMC Sleep | Main training and in-domain evaluation |
| CAP Sleep | Additional pretraining and cross-center evaluation after channel audit |
| Sleep-EDF Expanded | Auxiliary naturally-incomplete-modality experiments |
| ISRUC Cohort I | Sealed external evaluation only |

