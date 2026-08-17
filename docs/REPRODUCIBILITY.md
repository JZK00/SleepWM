# Reproducibility

## Environment

```bash
conda create -n sleepwm python=3.10
conda activate sleepwm
pip install -e ".[dev,data]"
bash scripts/run_smoke_test.sh
```

Record the Python, PyTorch, CUDA, GPU, and operating-system versions used for a
scientific run. The reported efficiency profile was measured on an RTX 3090.

## Data invariants

1. Split participants before window extraction.
2. Keep train, validation, calibration, and test participants disjoint.
3. Compute normalization and physiological feature statistics from training
   participants only.
4. Retain channel names, sampling metadata, label provenance, and recording IDs
   in each manifest.
5. Never use future signals, labels, or physiology features as causal inputs.

## Run artifacts

Each training directory must contain:

```text
resolved_config.yaml
best.pt
metrics.json
```

Each checkpoint records model parameters, construction metadata, epoch,
validation metric, and experiment identifier. Keep the resolved config and code
revision with every result.

## Frozen evaluation protocol

1. Select checkpoints using validation participants only.
2. Keep all model settings fixed across the reported independent training runs.
3. Evaluate every dynamic condition and sensor subset from the same checkpoint.
4. Store per-run and per-participant predictions before aggregation.
5. Average conditions and 30/60/120-second horizons according to the manuscript
   estimand; do not mix complete observation into the dynamic aggregate.
6. Use participant-level paired bootstrap intervals and paired rank tests for the
   prespecified strongest-baseline comparisons.
7. Apply multiplicity correction exactly as documented in the manuscript.
8. Preserve unfavorable prespecified stress strata in the supplement.

## Output locations

- Training runs: `outputs/`
- Evaluation summaries: the output directory passed to `test_sleepwm.py`
- Checkpoint hashes: `checkpoints/MANIFEST.json`

Generated predictions and participant-level statistics are local run artifacts.
They are not committed to the public source repository.

## Toy verification

`scripts/run_smoke_test.sh` generates synthetic data and validates software,
configuration, causality, and the public package namespace. Toy outputs are not
scientific evidence and must never be mixed with real evaluation outputs.

## Public-release safeguards

- Do not commit raw PSG, credentials, local server paths, or participant data.
- Do not commit checkpoint binaries to Git; publish them in a versioned archive.
- Verify every public weight against the SHA-256 manifest.
- Replace repository, DOI, and archive placeholders before tagging v1.0.0.
- Any numerical change after the frozen release requires a new release ID.
