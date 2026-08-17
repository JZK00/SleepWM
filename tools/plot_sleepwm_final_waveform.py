from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_belief_outcomes import dynamic_view, route_outputs
from train_belief_waveform_adapter import MetricAccumulator, build_adapter, selected_targets
from train_recursive_belief_filter import build_student
from train_waveform_diffusion_refiner import (
    build_emg_calibrated_conditions,
    build_structural_condition,
)
from uniphysio_wm.belief_waveform_adapter import (
    belief_waveform_context,
    last_observed_waveforms,
)
from uniphysio_wm.belief_waveform_diffusion import (
    append_emg_burst_condition,
    build_frozen_structural_refiner,
    sample_belief_conditioned_waveforms,
)
from uniphysio_wm.engine import (
    data_loader,
    load_checkpoint,
    physio_feature_sequence_dataset,
    resolve_device,
    seed_everything,
)
from uniphysio_wm.partial_observation import DynamicObservationSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the final SleepWM probabilistic structural waveform forecast."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--refiner-checkpoint")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def conditions(modalities):
    return (
        ("full_observation", None),
        (
            "hard_all_4ep",
            DynamicObservationSpec(
                "hard_all_4ep", {modality: 4 for modality in modalities}
            ),
        ),
    )


def decode_base(model, adapter, adapter_checkpoint, signals, present, horizons):
    output = model.rollout_context_horizons(signals, present, horizons)
    routes = route_outputs(model, output)
    route = routes["recursive_belief"]
    availability = present.to(signals.dtype).mean(dim=1)
    decoded, _, probabilities = model.waveform_decoder(
        signals[:, -1],
        route["states"][:, 0],
        output["physiology_dynamics_states"][:, 0],
        return_structure=True,
        return_probabilities=True,
        modality_availability=availability,
    )
    recent = last_observed_waveforms(
        signals, present, model.waveform_decoder.max_samples
    )
    context = belief_waveform_context(
        output, route, routes["static_persistence"], present
    )
    calibration = adapter_checkpoint["calibration"]
    adapted, _ = adapter(
        decoded,
        recent,
        context,
        eeg_correction_scale=float(calibration["eeg_correction_scale"]),
        emg_correction_scale=float(calibration["emg_correction_scale"]),
    )
    return adapted, probabilities, recent, output, route


def cpu_tensor_tree(values):
    return {
        key: (
            cpu_tensor_tree(value)
            if isinstance(value, dict)
            else value.detach().float().cpu()
        )
        for key, value in values.items()
    }


class DistributionAccumulator:
    def __init__(self, modalities):
        self.modalities = tuple(modalities)
        self.values = {
            modality: {
                "covered": 0,
                "count": 0,
                "width": 0.0,
                "diversity": 0.0,
            }
            for modality in modalities
        }

    def update(self, lower, upper, members, target, valid):
        for index, modality in enumerate(self.modalities):
            selected = valid[:, index]
            if not selected.any():
                continue
            truth = target[selected, index]
            low = lower[selected, index]
            high = upper[selected, index]
            count = truth.numel()
            values = self.values[modality]
            values["covered"] += int(((truth >= low) & (truth <= high)).sum().cpu())
            values["count"] += count
            values["width"] += float((high - low).sum().cpu())
            values["diversity"] += float(
                members[:, selected, index].std(dim=0).sum().cpu()
            )

    def result(self):
        return {
            modality: {
                "pointwise_80_interval_coverage": values["covered"]
                / max(1, values["count"]),
                "mean_80_interval_width": values["width"]
                / max(1, values["count"]),
                "mean_member_standard_deviation": values["diversity"]
                / max(1, values["count"]),
            }
            for modality, values in self.values.items()
        }


def gate(validation):
    primary = validation["hard_all_4ep"]
    base = primary["base"]
    candidate = primary["probabilistic"]
    distribution = primary["distribution"]
    checks = {
        "eeg_spectrum_improved": candidate["eeg_log_spectrum_mae"]
        < base["eeg_log_spectrum_mae"],
        "eeg_band_power_improved": candidate["eeg_band_log_power_mae"]
        < base["eeg_band_log_power_mae"],
        "emg_envelope_improved": candidate["emg_envelope_mae"]
        < base["emg_envelope_mae"],
        "emg_rms_improved": candidate["emg_rms_mae"] < base["emg_rms_mae"],
        "emg_burst_retained": candidate["emg_burst_f1"]
        >= 0.98 * base["emg_burst_f1"],
        "eeg_point_mae_retained": candidate["eeg_waveform_mae"]
        <= 1.15 * base["eeg_waveform_mae"],
        "emg_point_mae_retained": candidate["emg_waveform_mae"]
        <= 1.15 * base["emg_waveform_mae"],
        "eeg_distribution_noncollapsed": distribution["EEG"][
            "mean_member_standard_deviation"
        ]
        >= 0.008,
        "emg_distribution_noncollapsed": distribution["EMG"][
            "mean_member_standard_deviation"
        ]
        >= 0.008,
    }
    return {"passed": all(checks.values()), "checks": checks}


def subject_conformal_calibration(
    records,
    modalities,
    sample_rate,
    patch_samples,
    target_coverage,
    calibration_fraction,
    split_seed,
    center_scale_config=None,
):
    target = torch.cat(records["target"]).float()
    base = torch.cat(records["base"]).float()
    candidate_unscaled = torch.cat(records["candidate"]).float()
    candidate = candidate_unscaled.clone()
    median = torch.cat(records.get("median", records["candidate"])).float()
    lower = torch.cat(records["lower"]).float()
    upper = torch.cat(records["upper"]).float()
    valid = torch.cat(records["valid"])
    subjects = tuple(records["subject"])
    unique_subjects = sorted(set(subjects))
    ordered_subjects = sorted(
        unique_subjects,
        key=lambda subject: hashlib.sha256(
            f"{int(split_seed)}:{subject}".encode("utf-8")
        ).hexdigest(),
    )
    calibration_count = max(
        1,
        min(
            len(ordered_subjects) - 1,
            int(round(len(ordered_subjects) * float(calibration_fraction))),
        ),
    )
    calibration_subjects = set(ordered_subjects[:calibration_count])
    calibration_mask = torch.tensor(
        [subject in calibration_subjects for subject in subjects], dtype=torch.bool
    )
    evaluation_mask = ~calibration_mask
    center_scale_config = dict(center_scale_config or {})
    center_scales = {"EEG": 1.0, "EMG": 1.0}
    if bool(center_scale_config.get("enabled", False)):
        minimum_scale = float(center_scale_config.get("minimum_scale", 0.75))
        maximum_scale = float(center_scale_config.get("maximum_scale", 2.0))
        eeg_index = tuple(modalities).index("EEG")
        eeg_selected = calibration_mask & valid[:, eeg_index]
        eeg_truth = target[eeg_selected, eeg_index]
        eeg_candidate = candidate[eeg_selected, eeg_index]
        frequency = torch.fft.rfftfreq(
            eeg_truth.shape[-1], d=1.0 / float(sample_rate)
        )
        band = (frequency >= 0.5) & (frequency <= 30.0)
        truth_spectrum = torch.fft.rfft(
            eeg_truth - eeg_truth.mean(-1, keepdim=True), dim=-1
        ).abs()[:, band]
        candidate_spectrum = torch.fft.rfft(
            eeg_candidate - eeg_candidate.mean(-1, keepdim=True), dim=-1
        ).abs()[:, band]
        eeg_scale = torch.median(
            (truth_spectrum + 1e-4) / (candidate_spectrum + 1e-4)
        ).clamp(minimum_scale, maximum_scale)
        center_scales["EEG"] = float(eeg_scale)

        emg_index = tuple(modalities).index("EMG")
        emg_selected = calibration_mask & valid[:, emg_index]
        window = max(1, round(0.25 * float(sample_rate)))
        truth_envelope = F.avg_pool1d(
            target[emg_selected, emg_index].abs().unsqueeze(1),
            window,
            stride=window,
        ).squeeze(1)
        candidate_envelope = F.avg_pool1d(
            candidate[emg_selected, emg_index].abs().unsqueeze(1),
            window,
            stride=window,
        ).squeeze(1)
        emg_scale = torch.median(
            (truth_envelope + 1e-4) / (candidate_envelope + 1e-4)
        ).clamp(minimum_scale, maximum_scale)
        center_scales["EMG"] = float(emg_scale)

        for modality, scale in center_scales.items():
            modality_index = tuple(modalities).index(modality)
            candidate[:, modality_index] *= scale
            median[:, modality_index] *= scale
            lower[:, modality_index] *= scale
            upper[:, modality_index] *= scale

    def evaluation_center_metrics(values):
        accumulator = MetricAccumulator(modalities, sample_rate, patch_samples)
        accumulator.update(
            "frozen",
            base[evaluation_mask],
            target[evaluation_mask],
            valid[evaluation_mask],
        )
        accumulator.update(
            "adapted",
            values[evaluation_mask],
            target[evaluation_mask],
            valid[evaluation_mask],
        )
        return accumulator.result()["adapted"]

    unscaled_evaluation_metrics = evaluation_center_metrics(candidate_unscaled)
    scaled_evaluation_metrics = evaluation_center_metrics(candidate)
    alpha = 1.0 - float(target_coverage)
    calibrated_lower = lower.clone()
    calibrated_upper = upper.clone()
    modality_results = {}

    for modality in ("EEG", "EMG"):
        modality_index = tuple(modalities).index(modality)
        selected_calibration = calibration_mask & valid[:, modality_index]
        truth = target[selected_calibration, modality_index]
        low = lower[selected_calibration, modality_index]
        high = upper[selected_calibration, modality_index]
        scores = torch.maximum(low - truth, truth - high).flatten()
        rank = min(
            len(scores),
            max(1, int(np.ceil((len(scores) + 1) * (1.0 - alpha)))),
        )
        expansion = float(scores.kthvalue(rank).values.clamp_min(0.0))
        calibrated_lower[:, modality_index] -= expansion
        calibrated_upper[:, modality_index] += expansion

        selected_evaluation = evaluation_mask & valid[:, modality_index]
        eval_truth = target[selected_evaluation, modality_index]
        raw_low = lower[selected_evaluation, modality_index]
        raw_high = upper[selected_evaluation, modality_index]
        calibrated_low = calibrated_lower[selected_evaluation, modality_index]
        calibrated_high = calibrated_upper[selected_evaluation, modality_index]
        raw_coverage = ((eval_truth >= raw_low) & (eval_truth <= raw_high)).float()
        calibrated_coverage = (
            (eval_truth >= calibrated_low) & (eval_truth <= calibrated_high)
        ).float()

        subject_coverages = []
        evaluation_subject_sequence = np.asarray(subjects, dtype=object)[
            selected_evaluation.numpy()
        ]
        for subject in sorted(set(evaluation_subject_sequence.tolist())):
            subject_mask = torch.from_numpy(evaluation_subject_sequence == subject)
            subject_coverages.append(
                float(calibrated_coverage[subject_mask].mean())
            )
        modality_results[modality] = {
            "calibration_points": int(scores.numel()),
            "cqr_additive_expansion": expansion,
            "evaluation_raw_pointwise_coverage": float(raw_coverage.mean()),
            "evaluation_calibrated_pointwise_coverage": float(
                calibrated_coverage.mean()
            ),
            "evaluation_calibrated_mean_width": float(
                (calibrated_high - calibrated_low).mean()
            ),
            "evaluation_subject_mean_coverage": float(np.mean(subject_coverages)),
            "evaluation_subject_coverage_std": float(np.std(subject_coverages)),
        }

    evaluation_records = {
        "target": [target[evaluation_mask]],
        "base": [base[evaluation_mask]],
        "candidate": [candidate[evaluation_mask]],
        "median": [median[evaluation_mask]],
        "valid": [valid[evaluation_mask]],
    }
    evaluation_records["lower"] = [calibrated_lower[evaluation_mask]]
    evaluation_records["upper"] = [calibrated_upper[evaluation_mask]]
    evaluation_records["subject"] = [
        subject for subject, selected in zip(subjects, evaluation_mask.tolist()) if selected
    ]
    return {
        "method": "subject-disjoint additive conformalized quantile interval",
        "target_coverage": float(target_coverage),
        "calibration_subjects": sorted(calibration_subjects),
        "evaluation_subjects": sorted(set(ordered_subjects) - calibration_subjects),
        "calibration_sequences": int(calibration_mask.sum()),
        "evaluation_sequences": int(evaluation_mask.sum()),
        "center_amplitude_calibration": {
            "enabled": bool(center_scale_config.get("enabled", False)),
            "fitted_scales": center_scales,
            "evaluation_metrics_before": unscaled_evaluation_metrics,
            "evaluation_metrics_after": scaled_evaluation_metrics,
        },
        "modalities": modality_results,
    }, evaluation_records


def plot_representative(
    records,
    modalities,
    sample_rate,
    output,
    interval_label="SleepWM 10-90% interval",
    title="SleepWM probabilistic waveform structure under an all-sensor interruption",
    model_label="SleepWM",
    waveform_detail_seconds=2.0,
):
    target = torch.cat(records["target"]).float()
    base = torch.cat(records["base"]).float()
    candidate = torch.cat(records["candidate"]).float()
    median = torch.cat(records.get("median", records["candidate"])).float()
    lower = torch.cat(records["lower"]).float()
    upper = torch.cat(records["upper"]).float()
    valid = torch.cat(records["valid"])
    eeg_index = modalities.index("EEG")
    emg_index = modalities.index("EMG")
    frequency = torch.fft.rfftfreq(
        target.shape[-1], d=1.0 / float(sample_rate)
    )

    def eeg_band_error(values):
        power = torch.fft.rfft(values, dim=-1).abs().square()
        truth = torch.fft.rfft(target[:, eeg_index], dim=-1).abs().square()
        errors = []
        for lower_hz, upper_hz in ((0.5, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 30.0)):
            selected = (frequency >= lower_hz) & (frequency < upper_hz)
            errors.append(
                (
                    torch.log1p(power[:, selected].mean(-1))
                    - torch.log1p(truth[:, selected].mean(-1))
                ).abs()
            )
        return torch.stack(errors, dim=-1).mean(-1)

    window = max(1, round(0.25 * sample_rate))
    target_envelope = F.avg_pool1d(
        target[:, emg_index].abs().unsqueeze(1), window, stride=window
    ).squeeze(1)
    base_envelope = F.avg_pool1d(
        base[:, emg_index].abs().unsqueeze(1), window, stride=window
    ).squeeze(1)
    candidate_envelope = F.avg_pool1d(
        candidate[:, emg_index].abs().unsqueeze(1), window, stride=window
    ).squeeze(1)
    median_envelope = F.avg_pool1d(
        median[:, emg_index].abs().unsqueeze(1), window, stride=window
    ).squeeze(1)
    eeg_base = eeg_band_error(base[:, eeg_index])
    eeg_candidate = eeg_band_error(candidate[:, eeg_index])
    eeg_median = eeg_band_error(median[:, eeg_index])
    emg_base = (base_envelope - target_envelope).abs().mean(-1)
    emg_candidate = (candidate_envelope - target_envelope).abs().mean(-1)
    emg_median = (median_envelope - target_envelope).abs().mean(-1)
    # The selected waveform is the target-free member closest to the physiology
    # predicted by the belief state. Use it as the representative trajectory;
    # pointwise ensemble medians are retained only to expose phase uncertainty.
    eligible = torch.nonzero(
        valid[:, eeg_index]
        & valid[:, emg_index]
        & (eeg_candidate < eeg_base)
        & (emg_candidate < emg_base),
        as_tuple=False,
    ).flatten()
    if not len(eligible):
        eligible = torch.nonzero(
            valid[:, eeg_index] & valid[:, emg_index], as_tuple=False
        ).flatten()
    joint = (
        eeg_candidate[eligible]
        / eeg_candidate[eligible].median().clamp_min(1e-6)
        + emg_candidate[eligible]
        / emg_candidate[eligible].median().clamp_min(1e-6)
    )
    selected = int(eligible[(joint - joint.median()).abs().argmin()])
    seconds = np.arange(target.shape[-1]) / float(sample_rate)
    detail_seconds = min(float(waveform_detail_seconds), float(seconds[-1]))
    detail_samples = max(
        2,
        min(target.shape[-1], int(round(detail_seconds * float(sample_rate)))),
    )
    detail_frequency = torch.fft.rfftfreq(
        detail_samples, d=1.0 / float(sample_rate)
    )
    envelope_seconds = (
        np.arange(target_envelope.shape[-1]) + 0.5
    ) * window / float(sample_rate)
    colors = {
        "target": "#171717",
        "base": "#6F89A8",
        "candidate": "#C74452",
        "median": "#B78A6A",
    }
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 7.4))
    axes[0, 0].fill_between(
        seconds,
        lower[selected, eeg_index],
        upper[selected, eeg_index],
        color=colors["candidate"],
        alpha=0.16,
        label=interval_label,
    )
    axes[0, 0].plot(seconds, target[selected, eeg_index], color=colors["target"], linewidth=0.85, label="Target")
    axes[0, 0].plot(seconds, base[selected, eeg_index], color=colors["base"], linestyle="--", linewidth=0.8, label="Deterministic decoder")
    axes[0, 0].plot(
        seconds,
        candidate[selected, eeg_index],
        color=colors["candidate"],
        linewidth=0.9,
        label=f"{model_label} condition-consistent forecast",
    )
    axes[0, 0].plot(
        seconds,
        median[selected, eeg_index],
        color=colors["median"],
        linewidth=0.7,
        alpha=0.65,
        label=f"{model_label} pointwise ensemble median",
    )
    for values, color, style in (
        (target, colors["target"], "-"),
        (base, colors["base"], "--"),
        (candidate, colors["candidate"], "-"),
        (median, colors["median"], "-"),
    ):
        centered = values[selected, eeg_index, :detail_samples]
        centered = centered - centered.mean()
        power = torch.fft.rfft(centered).abs().square().clamp_min(1e-8)
        axes[0, 1].plot(
            detail_frequency,
            10.0 * torch.log10(power),
            color=color,
            linestyle=style,
            linewidth=0.9,
        )
    axes[1, 0].fill_between(
        seconds,
        lower[selected, emg_index],
        upper[selected, emg_index],
        color=colors["candidate"],
        alpha=0.16,
    )
    axes[1, 0].plot(seconds, target[selected, emg_index], color=colors["target"], linewidth=0.7)
    axes[1, 0].plot(seconds, base[selected, emg_index], color=colors["base"], linestyle="--", linewidth=0.75)
    axes[1, 0].plot(
        seconds,
        candidate[selected, emg_index],
        color=colors["candidate"],
        linewidth=0.85,
    )
    axes[1, 0].plot(
        seconds,
        median[selected, emg_index],
        color=colors["median"],
        linewidth=0.65,
        alpha=0.65,
    )
    axes[1, 1].plot(envelope_seconds, target_envelope[selected], color=colors["target"], linewidth=1.3)
    axes[1, 1].plot(envelope_seconds, base_envelope[selected], color=colors["base"], linestyle="--", linewidth=1.2)
    axes[1, 1].plot(
        envelope_seconds,
        candidate_envelope[selected],
        color=colors["candidate"],
        linewidth=1.3,
    )
    axes[1, 1].plot(
        envelope_seconds,
        median_envelope[selected],
        color=colors["median"],
        linewidth=1.0,
        alpha=0.65,
    )
    axes[0, 0].set_xlim(0.0, detail_seconds)
    axes[1, 0].set_xlim(0.0, detail_seconds)
    axes[1, 1].set_xlim(0.0, detail_seconds)
    axes[0, 0].set_title(f"EEG structural forecast | fixed first {detail_seconds:g} s detail")
    axes[0, 1].set_title(
        f"EEG spectrum (fixed first {detail_seconds:g} s) | "
        f"full-window error {eeg_candidate[selected]:.3f} vs deterministic {eeg_base[selected]:.3f}"
    )
    axes[1, 0].set_title(f"EMG structural forecast | fixed first {detail_seconds:g} s detail")
    axes[1, 1].set_title(
        f"EMG envelope (fixed first {detail_seconds:g} s) | "
        f"full-window error {emg_candidate[selected]:.3f} vs deterministic {emg_base[selected]:.3f}"
    )
    axes[0, 1].set_xlim(0.5, 30.0)
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_ylabel("Power (dB)")
    for axis in (axes[1, 0], axes[1, 1]):
        axis.set_xlabel("Seconds after the causal cutoff")
    for axis in (axes[0, 0], axes[1, 0]):
        axis.set_ylabel("Standardized amplitude")
    for axis in axes.flat:
        axis.grid(color="#D8D8D8", linewidth=0.5, alpha=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.95), frameon=False)
    figure.suptitle(title, fontsize=14, y=0.995)
    figure.text(
        0.5,
        0.012,
        "Red is selected without the future target by agreement with the belief-predicted EEG spectrum and EMG envelope/RMS/burst structure; brown is the pointwise ensemble median. Exact phase agreement is not expected after the interruption. The case is the median joint structural case among improved validation samples; no test sample is used.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    figure.tight_layout(rect=(0.02, 0.05, 0.99, 0.91))
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)
    return {
        "selected_dataset_index": selected,
        "jointly_improved_candidate_count": len(eligible),
        "eeg_band_error": {
            "condition_consistent_forecast": float(eeg_candidate[selected]),
            "ensemble_median": float(eeg_median[selected]),
            "po5": float(eeg_base[selected]),
        },
        "emg_envelope_error": {
            "condition_consistent_forecast": float(emg_candidate[selected]),
            "ensemble_median": float(emg_median[selected]),
            "po5": float(emg_base[selected]),
        },
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    with config_path.open(encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    po6 = protocol["probabilistic_waveform"]
    model_label = str(protocol.get("paper_display_label", "SleepWM"))
    po3_path = Path(po6["po3_checkpoint"])
    po5_path = Path(po6["po5_checkpoint"])
    refiner_path = Path(
        args.refiner_checkpoint or po6["frozen_refiner_checkpoint"]
    )
    calibration_path = Path(po6["frozen_calibration_checkpoint"])
    po3_checkpoint = load_checkpoint(po3_path)
    po5_checkpoint = load_checkpoint(po5_path)
    refiner_checkpoint = load_checkpoint(refiner_path)
    calibration_checkpoint = load_checkpoint(calibration_path)
    config = copy.deepcopy(po3_checkpoint["config"])
    for section in ("experiment", "data", "physiology", "train"):
        config.setdefault(section, {}).update(protocol.get(section, {}))
    if args.device:
        config["train"]["device"] = args.device
    if args.seed is not None:
        config["experiment"]["seed"] = int(args.seed)
    config["train"]["batch_size"] = int(po6.get("batch_size", 16))
    if args.smoke:
        config["train"]["num_workers"] = 0
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    device = resolve_device(str(config["train"].get("device", "cuda")))
    output_dir = Path(args.output_dir or po6["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_student(config).to(device)
    model.load_state_dict(po3_checkpoint["model_state"], strict=True)
    model.requires_grad_(False)
    model.eval()
    adapter_config = copy.deepcopy(config)
    adapter_config["waveform_adapter"] = po5_checkpoint["adapter_config"]
    adapter = build_adapter(model, adapter_config).to(device)
    adapter.load_state_dict(po5_checkpoint["adapter_state"], strict=True)
    adapter.requires_grad_(False)
    adapter.eval()
    refiner = build_frozen_structural_refiner(
        refiner_checkpoint,
        int(model.encoder.config.d_model),
        int(config["physiology"]["physiology_dynamics_dim"]),
    ).to(device)

    dataset = physio_feature_sequence_dataset(config, "val")
    if args.smoke:
        dataset = Subset(dataset, range(min(64, len(dataset))))
    modalities = tuple(config["data"]["modalities"])
    horizons = tuple(int(value) for value in config["data"]["future_horizons"])
    maximum_samples = int(model.waveform_decoder.max_samples)
    sample_rate = int(config["data"]["sample_rate"])
    patch_samples = int(model.waveform_decoder.patch_samples)
    ensemble_samples = 2 if args.smoke else int(po6["ensemble_samples"])
    sampling_steps = 2 if args.smoke else int(po6["sampling_steps"])
    validation = {}
    hard_records = {
        name: []
        for name in ("target", "base", "candidate", "median", "lower", "upper", "valid")
    }
    hard_records["subject"] = []

    with torch.inference_mode():
        for condition_name, spec in conditions(modalities):
            metrics = MetricAccumulator(modalities, sample_rate, patch_samples)
            distribution = DistributionAccumulator(modalities)
            for batch_index, batch in enumerate(data_loader(dataset, config, shuffle=False)):
                natural_signals = batch["history_signals"].to(device=device, dtype=torch.float32)
                natural_present = batch["history_present"].to(device=device, dtype=torch.bool)
                signals, present, _ = dynamic_view(natural_signals, natural_present, modalities, spec)
                target, valid = selected_targets(batch, device, maximum_samples)
                base, probabilities, recent, output, route = decode_base(
                    model, adapter, po5_checkpoint, signals, present, horizons
                )
                cpu_probabilities = cpu_tensor_tree(probabilities)
                cpu_base = base.detach().float().cpu()
                cpu_recent = recent.detach().float().cpu()
                structural_condition = build_structural_condition(
                    calibration_checkpoint["calibration_state"],
                    cpu_base,
                    cpu_probabilities,
                    cpu_recent,
                    modalities,
                    sample_rate,
                    patch_samples,
                ).to(device=device, dtype=torch.float32)
                _, _, calibrated_burst_probability, calibrated_burst_threshold = build_emg_calibrated_conditions(
                    calibration_checkpoint["calibration_state"],
                    cpu_base,
                    cpu_probabilities,
                    cpu_recent,
                    modalities,
                    sample_rate,
                    patch_samples,
                )
                structural_condition = append_emg_burst_condition(
                    structural_condition,
                    calibrated_burst_probability.to(
                        device=device, dtype=torch.float32
                    ),
                    modalities,
                    patch_samples,
                    refiner.structural_condition_channels,
                )
                sampled = sample_belief_conditioned_waveforms(
                    refiner,
                    base,
                    recent,
                    structural_condition,
                    probabilities,
                    route["states"][:, 0],
                    output["physiology_dynamics_states"][:, 0],
                    modalities,
                    patch_samples,
                    sample_rate,
                    ensemble_samples,
                    sampling_steps,
                    seed + 1009 * batch_index + (0 if condition_name == "full_observation" else 500000),
                    {name: float(value) for name, value in po6["residual_strengths"].items()},
                    float(po6["eeg_spectral_blend"]),
                    float(po6["emg_peak_exponent"]),
                    float(po6["emg_maximum_gain"]),
                    float(po6["emg_phase_randomization"]),
                    calibrated_burst_probability.to(device=device, dtype=torch.float32),
                    float(calibrated_burst_threshold),
                    float(po6.get("emg_burst_anchor", 0.85)),
                    float(po6.get("emg_burst_redistribution", 0.0)),
                    bool(po6.get("preserve_base_emg_bursts", False)),
                    float(po6.get("emg_rms_projection_blend", 1.0)),
                    {
                        name: float(value)
                        for name, value in po6.get(
                            "residual_output_gains", {}
                        ).items()
                    },
                )
                metrics.update("frozen", base, target, valid)
                metrics.update("adapted", sampled["selected"], target, valid)
                distribution.update(
                    sampled["lower"], sampled["upper"], sampled["members"], target, valid
                )
                if condition_name == "hard_all_4ep":
                    hard_records["subject"].extend(str(value) for value in batch["subject"])
                    for name, values in (
                        ("target", target),
                        ("base", base),
                        ("candidate", sampled["selected"]),
                        ("median", sampled["median"]),
                        ("lower", sampled["lower"]),
                        ("upper", sampled["upper"]),
                        ("valid", valid),
                    ):
                        hard_records[name].append(values.half().cpu() if values.dtype != torch.bool else values.cpu())
                print(f"condition={condition_name} batches={batch_index + 1}", flush=True)
            routes = metrics.result()
            validation[condition_name] = {
                "base": routes["frozen"],
                "probabilistic": routes["adapted"],
                "distribution": distribution.result(),
            }

    figure_path = output_dir / "sleepwm_probabilistic_waveforms_validation.png"
    representative = plot_representative(
        hard_records,
        modalities,
        sample_rate,
        figure_path,
        interval_label=f"{model_label} 10-90% interval",
        title=(
            f"{model_label} probabilistic waveform structure under a "
            "120-s all-sensor interruption"
        ),
        model_label=model_label,
    )
    conformal_config = po6.get("conformal", {})
    conformal, conformal_records = subject_conformal_calibration(
        hard_records,
        modalities,
        sample_rate,
        patch_samples,
        float(conformal_config.get("target_coverage", 0.80)),
        float(conformal_config.get("calibration_subject_fraction", 0.50)),
        int(conformal_config.get("split_seed", seed + 6001)),
        conformal_config.get("center_amplitude_calibration", {}),
    )
    conformal_figure_path = (
        output_dir / "sleepwm_conformal_waveforms_evaluation.png"
    )
    conformal_representative = plot_representative(
        conformal_records,
        modalities,
        sample_rate,
        conformal_figure_path,
        interval_label=(
            f"{model_label} conformal "
            f"{100.0 * float(conformal['target_coverage']):.0f}% interval"
        ),
        title=(
            "SleepWM subject-disjoint conformal waveform intervals on held-out "
            "evaluation subjects"
        ),
        model_label=model_label,
    )
    result_gate = gate(validation)
    payload = {
        "validation": validation,
        "gate": result_gate,
        "representative": representative,
        "conformal": conformal,
        "conformal_representative": conformal_representative,
        "validation_sequences": len(dataset),
        "ensemble_samples": ensemble_samples,
        "sampling_steps": sampling_steps,
        "member_selection": "predicted-condition consistency only; future target not used",
        "checkpoints": {
            "po3": {"path": str(po3_path), "sha256": sha256(po3_path)},
            "po5": {"path": str(po5_path), "sha256": sha256(po5_path)},
            "frozen_refiner": {"path": str(refiner_path), "sha256": sha256(refiner_path)},
            "frozen_calibration": {"path": str(calibration_path), "sha256": sha256(calibration_path)},
        },
        "test_split_accessed": False,
        "claim_boundary": "probabilistic structural forecast, not phase-exact reconstruction",
        "literature_basis": ["CSDI", "TimeDiff", "Diffusion-TS", "FIDE", "sleep EEG latent diffusion with spectral loss"],
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "resolved_config.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps({"gate": result_gate, "representative": representative}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
