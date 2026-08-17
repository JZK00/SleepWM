from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Subset

from uniphysio_wm.engine import (
    data_loader,
    load_checkpoint,
    physio_feature_sequence_dataset,
    resolve_device,
    seed_everything,
)
from uniphysio_wm.mainline import build_mainline_model
from uniphysio_wm.metrics import classification_metrics
from uniphysio_wm.partial_observation import (
    DynamicObservationSpec,
    dynamic_observation_view,
    primary_dynamic_observation_specs,
)
from uniphysio_wm.partial_observation_model import (
    FreshnessAwareCarryCorrectWorldModel,
)


def build_evaluation_model(config: dict):
    partial = config.get("partial_observation", {})
    if not partial.get("enabled", False):
        return build_mainline_model(config)
    return build_mainline_model(
        config,
        model_class=FreshnessAwareCarryCorrectWorldModel,
        extra_model_kwargs={
            "partial_filter_hidden_dim": int(partial.get("hidden_dim", 128)),
            "initial_freshness_decay": float(
                partial.get("initial_freshness_decay", 0.35)
            ),
            "initial_uncertainty_age_scale": float(
                partial.get("initial_uncertainty_age_scale", 0.15)
            ),
            "initial_uncertainty_horizon_scale": float(
                partial.get("initial_uncertainty_horizon_scale", 0.05)
            ),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SleepWM under dynamic observation interruption."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def latent_distance(prediction: torch.Tensor, teacher: torch.Tensor) -> dict:
    return {
        "smooth_l1": float(F.smooth_l1_loss(prediction, teacher)),
        "rmse": float((prediction - teacher).square().mean().sqrt()),
        "cosine_similarity": float(
            F.cosine_similarity(prediction, teacher, dim=-1).mean()
        ),
    }


def condition_result(
    values: dict,
    labels: torch.Tensor,
    horizons: tuple[int, ...],
    num_classes: int,
) -> dict:
    logits = torch.cat(values["stage_logits"])
    history_state = torch.cat(values["history_state"])
    teacher_history_state = torch.cat(values["teacher_history_state"])
    predicted_states = torch.cat(values["predicted_states"])
    teacher_predicted_states = torch.cat(values["teacher_predicted_states"])
    reliability = torch.cat(values["reliability"])
    quality = torch.cat(values["quality"])
    availability = torch.cat(values["availability"])
    log_variance = torch.cat(values["log_variance"])
    primary_count = len(horizons)
    result = {
        "stage": {
            "primary": classification_metrics(
                logits[:, :primary_count], labels[:, :primary_count], num_classes
            ),
            "by_horizon": {
                str(horizon): classification_metrics(
                    logits[:, index], labels[:, index], num_classes
                )
                for index, horizon in enumerate(horizons)
            },
        },
        "latent": {
            "history": latent_distance(history_state, teacher_history_state),
            "future_primary": latent_distance(
                predicted_states[:, :primary_count],
                teacher_predicted_states[:, :primary_count],
            ),
            "by_horizon": {
                str(horizon): latent_distance(
                    predicted_states[:, index], teacher_predicted_states[:, index]
                )
                for index, horizon in enumerate(horizons)
            },
        },
        "observation": {
            "mean_quality": {
                modality: float(quality[:, index].mean())
                for index, modality in enumerate(values["modalities"])
            },
            "mean_availability": {
                modality: float(availability[:, index].mean())
                for index, modality in enumerate(values["modalities"])
            },
            "mean_reliability": {
                modality: float(reliability[:, index].mean())
                for index, modality in enumerate(values["modalities"])
            },
        },
        "recursive_uncertainty_scale": {
            str(horizon): float(log_variance[:, index].mul(0.5).exp().mean())
            for index, horizon in enumerate(horizons)
        },
    }
    if values["age_epochs"]:
        age_epochs = torch.cat(values["age_epochs"])
        freshness = torch.cat(values["freshness"])
        result["observation"]["mean_age_epochs"] = {
            modality: float(age_epochs[:, index].mean())
            for index, modality in enumerate(values["modalities"])
        }
        result["observation"]["mean_freshness"] = {
            modality: float(freshness[:, index].mean())
            for index, modality in enumerate(values["modalities"])
        }
    return result


def accumulator(modalities: tuple[str, ...]) -> dict:
    return {
        "stage_logits": [],
        "history_state": [],
        "teacher_history_state": [],
        "predicted_states": [],
        "teacher_predicted_states": [],
        "reliability": [],
        "quality": [],
        "availability": [],
        "log_variance": [],
        "age_epochs": [],
        "freshness": [],
        "modalities": modalities,
    }


def append_output(
    target: dict,
    output: dict,
    teacher: dict,
    quality: torch.Tensor,
    present: torch.Tensor,
) -> None:
    target["stage_logits"].append(output["stage_logits"].cpu())
    target["history_state"].append(output["corrected_history_state"].cpu())
    target["teacher_history_state"].append(
        teacher["corrected_history_state"].cpu()
    )
    target["predicted_states"].append(output["predicted_states"].cpu())
    target["teacher_predicted_states"].append(
        teacher["predicted_states"].cpu()
    )
    target["reliability"].append(output["observation_reliability"].cpu())
    target["quality"].append(quality.mean(dim=1).cpu())
    target["availability"].append(present.float().mean(dim=1).cpu())
    target["log_variance"].append(output["recursive_log_variance"].cpu())
    if "observation_age_epochs" in output:
        target["age_epochs"].append(output["observation_age_epochs"].cpu())
        target["freshness"].append(output["observation_freshness"].cpu())


def build_summary(results: dict, epoch_seconds: int) -> dict:
    reference = results["full_observation"]
    reference_f1 = float(reference["stage"]["primary"]["macro_f1"])
    for name, values in results.items():
        values["stage"]["primary_macro_f1_delta_from_full"] = (
            float(values["stage"]["primary"]["macro_f1"]) - reference_f1
        )

    duration_curves = {}
    for name, values in results.items():
        if not name.startswith("tail_"):
            continue
        parts = name.split("_")
        scope = parts[1]
        duration_epochs = int(parts[2][:-2])
        duration_curves.setdefault(scope, []).append(
            {
                "missing_seconds": duration_epochs * epoch_seconds,
                "primary_macro_f1": values["stage"]["primary"]["macro_f1"],
                "macro_f1_delta_from_full": values["stage"][
                    "primary_macro_f1_delta_from_full"
                ],
                "history_latent_smooth_l1": values["latent"]["history"][
                    "smooth_l1"
                ],
                "history_latent_cosine": values["latent"]["history"][
                    "cosine_similarity"
                ],
            }
        )
    for values in duration_curves.values():
        values.sort(key=lambda item: item["missing_seconds"])

    recovery_curve = []
    for name, values in results.items():
        if not name.startswith("recover_all_"):
            continue
        recovered_epochs = int(name.split("_after_")[1][:-2])
        recovery_curve.append(
            {
                "recovery_seconds": recovered_epochs * epoch_seconds,
                "primary_macro_f1": values["stage"]["primary"]["macro_f1"],
                "macro_f1_delta_from_full": values["stage"][
                    "primary_macro_f1_delta_from_full"
                ],
                "history_latent_smooth_l1": values["latent"]["history"][
                    "smooth_l1"
                ],
                "history_latent_cosine": values["latent"]["history"][
                    "cosine_similarity"
                ],
            }
        )
    recovery_curve.sort(key=lambda item: item["recovery_seconds"])
    return {
        "full_primary_macro_f1": reference_f1,
        "duration_curves": duration_curves,
        "recovery_curve": recovery_curve,
    }


def write_markdown(path: Path, metrics: dict) -> None:
    lines = [
        "# Dynamic observation baseline",
        "",
        "Validation-only evaluation of a frozen SleepWM checkpoint.",
        "",
        "| Condition | Primary Macro-F1 | Delta vs full | History latent Smooth-L1 | History latent cosine |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in metrics["conditions"].items():
        lines.append(
            f"| {name} | {values['stage']['primary']['macro_f1']:.4f} | "
            f"{values['stage']['primary_macro_f1_delta_from_full']:+.4f} | "
            f"{values['latent']['history']['smooth_l1']:.5f} | "
            f"{values['latent']['history']['cosine_similarity']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    checkpoint_path = Path(protocol["baseline"]["checkpoint_path"])
    checkpoint = load_checkpoint(checkpoint_path)
    config = copy.deepcopy(checkpoint["config"])
    for section in ("data", "physiology", "train"):
        config.setdefault(section, {}).update(protocol.get(section, {}))
    config["experiment"] = copy.deepcopy(protocol["experiment"])
    output_dir = Path(args.output_dir or protocol["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(config["experiment"]["seed"]))
    device = resolve_device(str(config["train"].get("device", "cuda")))
    model = build_evaluation_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    dataset = physio_feature_sequence_dataset(config, "val")
    if args.smoke:
        dataset = Subset(dataset, range(min(64, len(dataset))))
    modalities = tuple(config["data"]["modalities"])
    dynamic = protocol["dynamic_observation"]
    specs = primary_dynamic_observation_specs(
        modalities,
        tuple(int(value) for value in dynamic["duration_epochs"]),
        tuple(int(value) for value in dynamic["recovery_epochs"]),
    )
    conditions = {"full_observation": accumulator(modalities)}
    conditions.update({spec.name: accumulator(modalities) for spec in specs})
    labels = []
    subjects = set()
    with torch.inference_mode():
        for batch_index, batch in enumerate(data_loader(dataset, config, shuffle=False)):
            history_signals = batch["history_signals"].to(
                device=device, dtype=torch.float32
            )
            history_present = batch["history_present"].to(
                device=device, dtype=torch.bool
            )
            teacher = model.rollout_context(history_signals, history_present)
            natural_quality = history_present.to(dtype=history_signals.dtype)
            append_output(
                conditions["full_observation"],
                teacher,
                teacher,
                natural_quality,
                history_present,
            )
            for spec in specs:
                signals, present, quality = dynamic_observation_view(
                    history_signals, history_present, modalities, spec
                )
                output = model.rollout_context(signals, present)
                append_output(conditions[spec.name], output, teacher, quality, present)
            labels.append(batch["future_labels"].cpu())
            subjects.update(str(value) for value in batch["subject"])
            if (batch_index + 1) % 10 == 0:
                print(f"evaluated batches={batch_index + 1}", flush=True)

    labels_tensor = torch.cat(labels)
    primary_horizons = tuple(int(value) for value in dynamic["primary_horizons"])
    results = {
        name: condition_result(
            values,
            labels_tensor,
            primary_horizons,
            int(config["data"].get("num_classes", 5)),
        )
        for name, values in conditions.items()
    }
    summary = build_summary(results, int(config["data"]["epoch_seconds"]))
    metrics = {
        "protocol": {
            "experiment_id": config["experiment"]["id"],
            "split": "val",
            "test_accessed": False,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "subjects": len(subjects),
            "endpoints": len(dataset),
            "primary_horizons_epochs": primary_horizons,
            "primary_horizons_seconds": [
                value * int(config["data"]["epoch_seconds"])
                for value in primary_horizons
            ],
            "smoke": args.smoke,
        },
        "summary": summary,
        "conditions": results,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    write_markdown(output_dir / "summary.md", metrics)
    print(json.dumps({"protocol": metrics["protocol"], "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
