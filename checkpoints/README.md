# Checkpoints

The lightweight repository includes one complete matched SleepWM v1 inference
set. Every file is smaller than GitHub's 100 MB per-file limit and is verified by
`tools/check_release_integrity.py`.

```text
checkpoints/
  sleepwm_backbone/run_01.pt
  sleepwm_outcome_heads/run_01.pt
  recovery_gate/run_01.pt
  reliability_v2/run_01.pt
```

## Roles

- `sleepwm_backbone`: carry-and-correct belief filter and latent dynamics.
- `sleepwm_outcome_heads`: future state, physiology, and transition readouts.
- `recovery_gate`: target-domain correction weight when observations return.
- `reliability_v2`: physiological error magnitude and high-error ranking head.

All four `run_01.pt` files belong to the same independently trained model and
should be used together. The expected SHA-256 values are stored in
`MANIFEST.json`.

```bash
python tools/check_release_integrity.py
```

The complete-observation teacher, carry-and-correct initializer, and the two
additional independent runs used for paper statistics are not required for
deployment and are not bundled. They can be archived separately with the
experimental record.
