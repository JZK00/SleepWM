# Contributing

SleepWM is a frozen research release. Contributions that improve documentation,
portability, tests, preprocessing clarity, or reproducibility are welcome.

Changes that alter a reported split, checkpoint, endpoint, missingness condition,
aggregation rule, or numerical result must use a new release identifier and must
not overwrite the frozen v1 evidence.

## Development setup

```bash
conda create -n sleepwm python=3.10
conda activate sleepwm
pip install -e ".[dev,data]"
bash scripts/run_smoke_test.sh
```

Please keep raw data, credentials, participant information, machine-specific
paths, and checkpoint binaries out of commits. Add focused tests for behavioral
changes and describe whether a change affects scientific results.
