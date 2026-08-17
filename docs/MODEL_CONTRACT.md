# SleepWM model contract

This document defines the public SleepWM model. Historical experiment numbers
are development provenance, not separate publication models.

## Scope

SleepWM infers a task-relevant physiological belief from a causal history of EEG,
ECG, and EMG observations whose availability, quality, and age can change over
time. It maintains and recursively evolves that belief when observations are
missing, corrects drift when reliable evidence returns, and predicts:

- future five-class sleep stage;
- 11 standardized physiological descriptors; and
- cumulative sleep-transition probability.

Primary forecast horizons are 30, 60, and 120 seconds. Longer gaps are stress
tests of state maintenance rather than new prediction endpoints.

## Inference graph

For observation set `o_<=t`, modality availability `m_t`, quality `q_t`, and
freshness/age `d_t`, SleepWM computes:

```text
b_t       = belief_filter(o_<=t, m_t, q_t, d_t)
b_(t+h)   = latent_rollout(b_t, h)
b_corr    = reliability_correct(b_(t+h), fresh_evidence, uncertainty)
y_stage   = stage_head(b_corr)
y_phys    = physiology_head(b_corr)
y_event   = cumulative_transition_head(b_corr)
```

The observation process and latent state are modeled separately. Missing sensors
do not imply a stationary body state.

## Observation frontends

- **EEG:** multiscale temporal patches and spectral structure.
- **ECG:** beat morphology, heart-rate, and RR-timing evidence.
- **EMG:** rectified energy, envelope, RMS, and burst evidence.

The released protocol uses one selected channel per modality and 128-Hz signals
segmented into 30-second epochs. A causal context contains 20 epochs (10 minutes).

## Belief maintenance

The carry-and-correct filter combines the previous belief with available modality
evidence. Observation freshness, quality, and modality availability control how
much a new observation can update the state. During a blackout, the model carries
the previous belief into recursive dynamics instead of inserting future samples
or bidirectional context.

## Latent rollout and teacher constraint

Recursive transition blocks advance the belief to each forecast horizon. During
training, a frozen complete-observation teacher supplies current and future latent
targets. Teacher observations define targets only; they never enter the causal
rollout input. Causality tests verify that modifying future targets does not alter
predictions at the cutoff.

## Reliability-aware correction

When a modality remains available or returns, SleepWM combines latent and direct
outcome evidence through bounded, reliability-conditioned corrections. The gate
uses belief statistics, modality age and freshness, observation quality,
latent/direct disagreement, horizon, and predicted uncertainty. The correction
cannot replace the latent state with an unconstrained direct predictor.

## Outcome definitions

### Future sleep stage

Five-class Macro-F1 is reported at 30, 60, and 120 seconds and then aggregated
according to the frozen participant, condition, and horizon protocol.

### Future physiology

The model predicts five EEG descriptors, three ECG descriptors, and three EMG
descriptors using training-participant normalization and an ECG-validity mask.
The primary metric is standardized feature MAE, not sample-wise waveform error.

### Sleep-transition event

The event head predicts whether a sleep-stage change occurs by each forecast
horizon. This is a sleep-transition endpoint, not a driving-risk or clinical-risk
label. Every same-protocol baseline receives matched explicit event supervision.

## Single-view and missing-modality contract

The primary setting begins with a complete causal history and then changes sensor
availability. In the supporting single-view ablation, only EEG, ECG, or EMG
remains for 30 or 120 seconds:

- direct prediction uses only the surviving view;
- explicit completion reconstructs missing modality embeddings before prediction;
- SleepWM maintains the historical multimodal belief and corrects it with the
  surviving view.

Explicit completion does not reconstruct raw future EEG, ECG, or EMG.

## Reliability output

Reliability-V2 estimates high-error likelihood and expected error magnitude. It
supports sample ranking, abstention, and fallback analysis. The release does not
claim complete person-specific probability calibration.

## Structural waveform appendix

The supplementary renderer predicts probabilistic EEG spectral structure and EMG
envelope/RMS/burst structure with conformal intervals. It is secondary evidence
and is not part of the primary endpoint claim. Exact future phase, QRS timing over
long horizons, and high-fidelity EMG burst reconstruction are not claimed.

## Public naming and compatibility

The publication model is always **SleepWM**. The public Python namespace is
`sleepwm`. The `uniphysio_wm` namespace and several historical config keys remain
only to load frozen checkpoints and preserve reproducibility.

## Out of scope

- driving fatigue, accident, or clinical deterioration prediction;
- counterfactual intervention simulation;
- universal equivalence of all sensor subsets;
- a uniquely identifiable biological latent state; and
- prospective clinical workflow benefit.
