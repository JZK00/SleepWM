# SleepWM

Code for **SleepWM: an observational world model for multimodal sleep state completion and forecasting**.

SleepWM maintains a latent sleep state as EEG, ECG and EMG disappear and return. The final model predicts future sleep stages, 11 physiological features and stage-transition risk. Cross-modal pretraining learns correspondence between modalities within individual epochs; temporal prediction and partial-view state alignment are trained in subsequent stages.

This compact release focuses on the final model and its required training lineage. Historical configuration variants, auxiliary waveform-generation studies, external-transfer experiments, figure-generation scripts and benchmark suites are omitted. It contains source code, configurations and tests; recordings, participant manifests, weights and results must be obtained or produced separately.

中文：这是主线精简版。训练、评估、数据处理各保留一个入口；全部配置集中在两个文件，模型和训练实现放在 `sleepwm/` 内。运行时需要另行准备数据和兼容权重。

## Layout

```text
train.py                 Run any required training stage
evaluate.py              Final-model evaluation or synthetic inference check
prepare_data.py          Normalization, physiological features and toy data
sleepwm/                 Encoders, state updates, readouts and shared utilities
  training/              Required staged training implementations
configs/training.yaml    Selected-run settings and training dependencies
configs/checkpoints.json Six final checkpoint paths
tests/                   State/masking and readout regression tests
```

The other root files contain installation metadata, the MIT license, citation metadata and ignore rules. There are no separate reporting or documentation directories.

## Install

Use Python 3.10 or later. Run all commands from this repository root:

```bash
pip install -e '.[dev]'
CUDA_VISIBLE_DEVICES='' python -m pytest -q
```

For GPU execution, set `CUDA_VISIBLE_DEVICES` explicitly and address the visible device as `cuda:0`. For example, `CUDA_VISIBLE_DEVICES=4` assigns physical GPU 4. CPU checks use `CUDA_VISIBLE_DEVICES=''`.

## Data

Prepare provider recordings as 30-second epochs at 128 Hz, in modality order EEG, ECG, EMG. Each record is an NPZ with `signals` shaped `[epochs, 3, 3840]`, integer `labels` (Wake, N1, N2, N3, REM = 0–4), and Boolean `modality_present` shaped `[epochs, 3]`. Signals retain physical units before train-fitted normalization. Unavailable channels are represented by the mask.

A CSV manifest needs `record_id,subject,split,npz_path`. Paths may be absolute or relative to the manifest. Prefix subject identifiers with the dataset name when combining cohorts, and keep each subject in exactly one split. Exact paper replication requires the original split assignment, channel choices and preprocessing; the NPZ schema alone does not establish equivalence. Provider-specific EDF conversion and frozen participant manifests are not included in this compact package.

```bash
python prepare_data.py normalize \
  --manifest data/manifests/hmc_cap_processed_manifest.csv \
  --output data/manifests/hmc_cap_train_normalization.json
python prepare_data.py features \
  --manifest data/manifests/hmc_cap_processed_manifest.csv \
  --output-dir data/features
```

Normalization and feature statistics are fitted on training subjects only. Feature extraction creates `feature_manifest.csv` and `feature_statistics.json` and defaults to train/validation records. Held-out evaluation requires separately prepared test feature records using the frozen training statistics. `python prepare_data.py toy --output-dir .toy_data` creates synthetic records for software checks; it does not reproduce study results. Each subcommand has `--help`.

## Train

`configs/training.yaml` contains 27 stage records, ordered by checkpoint dependency. Identical blocks use YAML aliases. Export a stage to get a standalone editable YAML:

```bash
python train.py --list
python train.py --stage pretrain --export-config pretrain.local.yaml
python train.py --stage pretrain --stage-help
CUDA_VISIBLE_DEVICES=4 python train.py --stage pretrain --device cuda:0
```

To use edited paths/settings, add `--config pretrain.local.yaml`. Trainer-specific options such as `--epochs`, `--seed` and `--output-dir` are forwarded to the selected stage; consult `--stage-help`. An exported config for a checkpoint-driven event stage specifies its input checkpoints and optimizer-loop budget; data settings are inherited from those checkpoints. `SLEEPWM_DATA_ROOT` can relocate their embedded data paths.

The dependency sequence is:

1. **Cross-modal pretraining** (`pretrain`): within-epoch masked/contextual latent supervision, five training epochs, selected epoch 2 in the archived run.
2. **Task-trained reference**: supervised sleep tasks, forecasting, physiology/waveform modules and observation repair. These modules remain because the actual final checkpoint inherits them.
3. **Partial observation**: the two PO2 stages initialize the PO3 carry/correct state model (`po3_recursive_belief`).
4. **Frozen-state readouts**: PO4 outcome adapter, the required direct-observation branch and hazard head, PO17 physiology/hazard adaptation, then PO18 event correction. Their `requires` entries identify the checkpoints to train first.

Training writes to the configured `outputs/` paths. Checkpoint loading also finds the corresponding `outputs/` file when a `checkpoints/` path is absent. Use fresh output directories for new experiments and update dependent paths together. The supplied configuration lineage is for seed 20260804. The paper's downstream runs share a pretrained initialization; this package does not imply three independent pretraining runs or include every run's weights.

## Evaluate

The final system needs **six compatible checkpoint files**, listed in `configs/checkpoints.json`: PO3 state model, PO4 outcome adapter, direct-observation branch, PO17 latent adapter, direct hazard head and PO18 event correction. A single older checkpoint is insufficient. Use trusted checkpoints containing their configuration and model/adapter state dictionaries.

```bash
# Four synthetic histories; does not open a dataset.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 python evaluate.py \
  --checkpoints configs/checkpoints.json --device cpu --synthetic

# Complete observation plus the eight primary partial-observation conditions.
CUDA_VISIBLE_DEVICES=4 python evaluate.py \
  --checkpoints configs/checkpoints.json --device cuda:0 \
  --split val --protocol primary --output outputs/validation.json
```

Use `--data-config local_data.yaml` to override the `data` and `physiology` path fields. `--protocol multiview` evaluates retained single/pair modality views at six interruption durations. The default split is validation; `--split test` explicitly selects held-out evaluation.

The primary partial-observation summary averages **eight fixed conditions** over the 30/60/120-second anchors, not newly sampled random test masks. Those short-horizon predictions use corrected future anchors. History-state propagation and the longer recursive extension are different operations. Earlier PO4-only diagnostics and external-transfer experiments use different protocols and should not be pooled with these final-readout results.

For direct Python inference:

```python
import json
from sleepwm import SleepWM

with open("configs/checkpoints.json") as handle:
    model = SleepWM.from_checkpoints(json.load(handle), device="cpu")
# signals: [batch, history, 3, samples]; present: matching Boolean [batch, history, 3]
# Set unavailable signal values to zero before calling the model.
outputs = model(signals, present)
```

## Verification and attribution

The compact release was checked on CPU with Python 3.10 and PyTorch 2.6.0: 21 targeted tests, argument parsing for all 27 stages, and synthetic end-to-end data preparation/evaluation. Loading the six archived checkpoints yielded 4,168,163 parameters. Every output was exactly equal to the previous release for complete observation, all-sensor loss, EEG loss and recovery after a gap. These checks establish software compatibility, not a new measurement of paper accuracy. Full training was not rerun.

The direct-observation branch implements GRU-D, which is needed internally by the final readout. Attribution: Che et al., *Recurrent Neural Networks for Multivariate Time Series with Missing Values*, Scientific Reports 8, 6085 (2018), https://doi.org/10.1038/s41598-018-24271-9. GRU-D is not claimed as a new SleepWM algorithm. Other comparator implementations are not bundled. Dependencies retain their own licenses.

The original MIT notice is retained in `LICENSE`. Cite the accompanying article and `CITATION.cff`; update the article metadata and archival software identifier when finalized. Repository: https://github.com/JZK00/SleepWM.
