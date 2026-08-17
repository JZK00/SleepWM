from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from calibrate_po4_uncertainty import build_adapter
from evaluate_belief_outcomes import condition_name, dynamic_view, protocol_specs, route_outputs
from train_recursive_belief_filter import build_student
from uniphysio_wm.engine import data_loader, load_checkpoint, resolve_device, seed_everything, sequence_dataset
from uniphysio_wm.metrics import classification_metrics


CONDITIONS = (
    "full_observation",
    "hard_eeg_1ep",
    "hard_eeg_4ep",
    "hard_eeg_10ep",
    "hard_all_1ep",
    "hard_all_4ep",
    "hard_all_10ep",
)
DYNAMIC_CONDITIONS = CONDITIONS[1:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a frozen-backbone domain recovery gate for external dynamic missingness."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sequence-stride", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    return parser.parse_args()


class DomainRecoveryGate(nn.Module):
    def __init__(self, features: int, hidden: int = 24):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1).sigmoid()


def entropy(probability: torch.Tensor) -> torch.Tensor:
    return -(
        probability * probability.clamp_min(1e-8).log()
    ).sum(dim=-1) / np.log(probability.shape[-1])


@torch.inference_mode()
def collect(model, adapter, dataset, config, device):
    modalities = tuple(config["data"]["modalities"])
    horizons = tuple(int(value) for value in config["data"]["future_horizons"])
    specs = {condition_name(spec): spec for spec in protocol_specs(modalities)}
    specs["full_observation"] = None
    result = {
        name: []
        for name in (
            "features",
            "label",
            "subject",
            "condition",
            "recursive",
            "persistence",
        )
    }
    model.eval()
    adapter.eval()
    for name in CONDITIONS:
        for batch in data_loader(dataset, config, shuffle=False):
            natural = batch["history_signals"].to(device=device, dtype=torch.float32)
            natural_present = batch["history_present"].to(device=device, dtype=torch.bool)
            signals, present, quality = dynamic_view(
                natural, natural_present, modalities, specs[name]
            )
            output = model.rollout_context_horizons(signals, present, horizons)
            routes = route_outputs(model, output)
            adapted = adapter(output)
            recursive = adapted["stage_logits"].softmax(dim=-1)
            persistence = routes["static_persistence"]["stage_logits"].softmax(dim=-1)
            direct = routes["direct_incomplete"]["stage_logits"].softmax(dim=-1)
            horizon_count = recursive.shape[1]
            horizon = torch.log1p(output["recursive_horizons"].to(recursive.dtype))
            horizon = horizon / torch.log1p(
                output["recursive_horizons"].max().to(recursive.dtype)
            )
            horizon = horizon.reshape(1, -1, 1).expand(len(recursive), -1, -1)
            availability = present.to(recursive.dtype).mean(dim=1)
            observation_quality = quality.to(recursive.dtype).mean(dim=1)
            age = output["observation_age_epochs"].to(recursive.dtype)
            reliability = output["observation_reliability"]
            freshness = output["observation_freshness"]
            metadata = torch.cat(
                tuple(
                    value.unsqueeze(1).expand(-1, horizon_count, -1)
                    for value in (
                        availability,
                        observation_quality,
                        age,
                        reliability,
                        freshness,
                    )
                ),
                dim=-1,
            )
            disagreement = (recursive - persistence).abs().mean(dim=-1, keepdim=True)
            features = torch.cat(
                (
                    recursive,
                    persistence,
                    direct,
                    entropy(recursive).unsqueeze(-1),
                    entropy(persistence).unsqueeze(-1),
                    recursive.max(dim=-1).values.unsqueeze(-1),
                    persistence.max(dim=-1).values.unsqueeze(-1),
                    disagreement,
                    horizon,
                    metadata,
                ),
                dim=-1,
            )
            result["features"].append(features.cpu().reshape(-1, features.shape[-1]))
            result["label"].append(
                batch["future_labels"][:, :horizon_count].reshape(-1).cpu()
            )
            result["subject"].append(
                np.repeat(np.asarray(batch["subject"], dtype=str), horizon_count)
            )
            result["condition"].append(
                np.full(len(recursive) * horizon_count, name, dtype="U40")
            )
            result["recursive"].append(recursive.cpu().reshape(-1, recursive.shape[-1]))
            result["persistence"].append(
                persistence.cpu().reshape(-1, persistence.shape[-1])
            )
        print(f"collected split condition={name}", flush=True)
    return {
        "features": torch.cat(result["features"]),
        "label": torch.cat(result["label"]),
        "subject": np.concatenate(result["subject"]),
        "condition": np.concatenate(result["condition"]),
        "recursive": torch.cat(result["recursive"]),
        "persistence": torch.cat(result["persistence"]),
    }


def apply_gate(gate, values, mean, standard_deviation, device, batch_size=8192):
    features = (values["features"] - mean) / standard_deviation
    weights = []
    gate.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            weights.append(gate(features[start : start + batch_size].to(device)).cpu())
    weight = torch.cat(weights)
    mixed = (
        weight.unsqueeze(-1) * values["recursive"]
        + (1.0 - weight).unsqueeze(-1) * values["persistence"]
    )
    return mixed, weight


def condition_metrics(values, probability, weight):
    result = {}
    for condition in CONDITIONS:
        selected = values["condition"] == condition
        labels = values["label"][selected]
        result[condition] = {
            "gate": classification_metrics(
                probability[selected].clamp_min(1e-8).log(), labels, 5
            ),
            "recursive_belief": classification_metrics(
                values["recursive"][selected].clamp_min(1e-8).log(), labels, 5
            ),
            "static_persistence": classification_metrics(
                values["persistence"][selected].clamp_min(1e-8).log(), labels, 5
            ),
            "mean_recursive_weight": float(weight[selected].mean()),
        }
    return result


def dynamic_score(metrics, route):
    return float(
        np.mean([metrics[name][route]["macro_f1"] for name in DYNAMIC_CONDITIONS])
    )


def main() -> int:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    protocol = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config = copy.deepcopy(checkpoint["config"])
    for section in ("experiment", "data", "physiology", "train"):
        config.setdefault(section, {}).update(protocol.get(section, {}))
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    config["data"]["manifest_path"] = str(Path(args.manifest).resolve())
    config["data"]["normalization_path"] = str(
        Path(protocol["data"]["normalization_path"]).resolve()
    )
    config["data"]["future_horizons"] = [1, 2, 4]
    config["data"]["sequence_stride"] = int(args.sequence_stride)
    config["train"].update(
        batch_size=int(args.batch_size),
        num_workers=int(args.workers),
        device=args.device,
    )
    device = resolve_device(args.device)
    model = build_student(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.requires_grad_(False)
    adapter_checkpoint = load_checkpoint(args.adapter_checkpoint)
    adapter = build_adapter(model, adapter_checkpoint, config, device)
    train_values = collect(model, adapter, sequence_dataset(config, "train"), config, device)
    validation_values = collect(model, adapter, sequence_dataset(config, "val"), config, device)

    mean = train_values["features"].mean(dim=0)
    standard_deviation = train_values["features"].std(dim=0).clamp_min(1e-4)
    train_features = (train_values["features"] - mean) / standard_deviation
    gate = DomainRecoveryGate(train_features.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 3571)
    best_score = -float("inf")
    best_state = None
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        gate.train()
        order = torch.randperm(len(train_features), generator=generator)
        total = 0.0
        batches = 0
        for start in range(0, len(order), 4096):
            selected = order[start : start + 4096]
            feature = train_features[selected].to(device)
            label = train_values["label"][selected].to(device)
            recursive = train_values["recursive"][selected].to(device)
            persistence = train_values["persistence"][selected].to(device)
            weight = gate(feature)
            mixed = weight.unsqueeze(-1) * recursive + (1.0 - weight).unsqueeze(-1) * persistence
            recursive_loss = -recursive.gather(1, label.unsqueeze(1)).clamp_min(1e-8).log().squeeze(1)
            persistence_loss = -persistence.gather(1, label.unsqueeze(1)).clamp_min(1e-8).log().squeeze(1)
            oracle = (recursive_loss <= persistence_loss).to(weight.dtype)
            loss = -mixed.gather(1, label.unsqueeze(1)).clamp_min(1e-8).log().mean()
            loss = loss + 0.15 * F.binary_cross_entropy(weight, oracle)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 2.0)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        probability, weight = apply_gate(
            gate, validation_values, mean, standard_deviation, device
        )
        metrics = condition_metrics(validation_values, probability, weight)
        score = dynamic_score(metrics, "gate")
        history.append({"epoch": epoch, "loss": total / batches, "dynamic_macro_f1": score})
        print(json.dumps(history[-1]), flush=True)
        if score > best_score:
            best_score = score
            best_state = {name: value.detach().cpu() for name, value in gate.state_dict().items()}
    if best_state is None:
        raise RuntimeError("domain recovery gate produced no checkpoint")
    gate.load_state_dict(best_state, strict=True)
    validation_probability, validation_weight = apply_gate(
        gate, validation_values, mean, standard_deviation, device
    )
    validation_metrics = condition_metrics(
        validation_values, validation_probability, validation_weight
    )
    validation_gate = {
        "dynamic_average_improved": dynamic_score(validation_metrics, "gate")
        > max(
            dynamic_score(validation_metrics, "recursive_belief"),
            dynamic_score(validation_metrics, "static_persistence"),
        ),
        "long_eeg_interruption_improved": validation_metrics["hard_eeg_10ep"]["gate"]["macro_f1"]
        >= max(
            validation_metrics["hard_eeg_10ep"]["recursive_belief"]["macro_f1"],
            validation_metrics["hard_eeg_10ep"]["static_persistence"]["macro_f1"],
        ),
        "full_observation_retained": validation_metrics["full_observation"]["gate"]["macro_f1"]
        >= max(
            validation_metrics["full_observation"]["recursive_belief"]["macro_f1"],
            validation_metrics["full_observation"]["static_persistence"]["macro_f1"],
        ) - 0.005,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol": "PROJECT3_EXTERNAL_DOMAIN_RECOVERY_GATE_V1",
            "model_state": best_state,
            "feature_mean": mean,
            "feature_standard_deviation": standard_deviation,
            "feature_count": int(len(mean)),
            "validation_gate": validation_gate,
            "checkpoint": args.checkpoint,
            "adapter_checkpoint": args.adapter_checkpoint,
            "test_split_accessed": False,
        },
        output / "best.pt",
    )
    test_values = collect(model, adapter, sequence_dataset(config, "test"), config, device)
    test_probability, test_weight = apply_gate(
        gate, test_values, mean, standard_deviation, device
    )
    payload = {
        "protocol": "PROJECT3_EXTERNAL_DOMAIN_RECOVERY_GATE_V1",
        "dataset": args.dataset_id,
        "seed": seed,
        "parameter_count": sum(parameter.numel() for parameter in gate.parameters()),
        "validation_gate": {"passed": all(validation_gate.values()), "checks": validation_gate},
        "validation": validation_metrics,
        "test": condition_metrics(test_values, test_probability, test_weight),
        "history": history,
        "backbone_frozen": True,
        "target_train_labels_used": True,
        "test_split_status": "previously accessed external split; secondary confirmatory analysis",
        "claim_boundary": "domain-adapted recovery routing, not strict source-only zero-shot",
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"validation_gate": payload["validation_gate"], "test_long_eeg": payload["test"]["hard_eeg_10ep"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
