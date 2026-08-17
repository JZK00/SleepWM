# Experiment Protocol

## Scientific questions

Q1. Does EEG/ECG/EMG fusion improve sleep staging over EEG alone?

Q2. Does masked multimodal pretraining improve a frozen linear probe and
end-to-end fine-tuning over an architecture-matched random initialization?

Q3. Does full-modality dropout improve robustness for every non-empty modality
combination?

Q4. Can a history-only transition model predict future physiological states and
sleep-stage trajectories without future leakage?

## Fixed experiment matrix

| ID | Configuration | Primary comparison |
| --- | --- | --- |
| B0 | `configs/baseline/eeg_tcn.yaml` | EEG-only supervised baseline |
| B1 | `configs/baseline/multimodal_tcn.yaml` | Ordinary early fusion |
| B2 | `configs/baseline/multimodal_transformer.yaml` | Random Transformer encoder |
| P0 | `configs/pretrain/masked_multimodal.yaml` | Local span masking |
| P1 | `configs/pretrain/missing_modality.yaml` | Span plus complete modality masking |
| T0 | `configs/train/linear_probe.yaml` | Frozen P1 representation |
| T1 | `configs/train/finetune.yaml` | End-to-end P1 fine-tuning |
| T1-full35 | `configs/train/finetune_full_biased.yaml` | Locked full/subset sampling balance |
| F0 | `configs/forecast/causal_rollout.yaml` | Causal future-state prediction |

## Evaluation

Sleep staging reports accuracy, balanced accuracy, macro-F1, Cohen's kappa,
per-class F1, and the confusion matrix. The primary model-selection metric is
validation macro-F1; the test split is evaluated after model selection.

Missing-modality evaluation covers all seven non-empty combinations of EEG,
ECG, and EMG. Results are reported as absolute scores and degradation relative
to complete input.

Masked reconstruction initially reports MSE and correlation. Paper-level
physiological evaluation must additionally include EEG band-power error, ECG
RR/HRV error, EMG envelope or tone error, and cross-modal coupling retention.

Forecasting reports horizon-specific latent cosine error and future sleep-stage
metrics at 30, 60, and 120 seconds. Future work may add probabilistic coverage
and longer risk horizons.

## Ablation discipline

Only one factor changes in each primary comparison. B2, P0, and P1 use the same
Transformer size. T0 and T1 use the same P1 checkpoint. All runs use identical
subject splits, preprocessing, label mapping, and model-selection rules.

At least three fixed seeds should be used for paper tables. Report mean,
standard deviation, and the per-seed results rather than selecting the best seed.
The Stage 3 single-seed sampling sweep selected full-observation probability
`0.35` before confirmatory seeds; this value must not be retuned per seed.
