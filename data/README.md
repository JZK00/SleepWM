# Data

SleepWM does not redistribute raw polysomnography. Download each dataset under
its own license and citation requirements.

| Dataset | Use in SleepWM | Official source |
| --- | --- | --- |
| HMC Sleep Staging Database v1.1 | Primary cohort | https://physionet.org/content/hmc-sleep-staging/1.1/ |
| CAP Sleep Database v1.0.0 | Primary multimodal cohort | https://physionet.org/content/capslpdb/1.0.0/ |
| ISRUC-SLEEP | External source-only evaluation | https://sleeptight.isr.uc.pt/ |
| Sleep-EDF Expanded v1.0.0 | External transfer and recovery | https://physionet.org/content/sleep-edfx/1.0.0/ |

## Expected local layout

```text
data/
  manifests/
    hmc_cap_processed_manifest.csv
    hmc_cap_train_normalization.json
    isruc_processed_manifest.csv
    sleep_edfx_processed_manifest.csv
  derived/
    physio_features/
      feature_manifest.csv
      feature_statistics.json
```

Raw and processed signal files may live outside the repository. Manifest paths
can be absolute locally but must not expose private server paths in public logs.

## Required manifest properties

Each row represents one recording and records:

- participant and recording identifiers;
- train, validation, calibration, or test split;
- processed signal path and available modalities;
- channel mapping and original/resampled sampling rate;
- epoch labels and their provenance; and
- dataset and preprocessing version.

Participant sets must be disjoint. Normalization, channel selection, and the 11
physiological feature statistics must be determined without validation,
calibration, or test participants.

See [`../docs/DATA_CONTRACT.md`](../docs/DATA_CONTRACT.md) for the complete schema,
channel mapping, validity masks, and leakage rules.
