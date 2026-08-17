from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from calibrate_po4_uncertainty import build_adapter
from train_external_dynamic_gate import (
    CONDITIONS,
    DYNAMIC_CONDITIONS,
    DomainRecoveryGate,
    apply_gate,
    collect,
)
from train_recursive_belief_filter import build_student
from uniphysio_wm.engine import (
    load_checkpoint,
    resolve_device,
    seed_everything,
    sequence_dataset,
)
from uniphysio_wm.metrics import classification_metrics


METHODS = ("learned_recovery", "fixed_half", "recursive_belief", "static_persistence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen domain recovery gate against a fixed 0.5 mixture."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--gate-checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sequence-stride", type=int, default=8)
    return parser.parse_args()


def metrics(probability: torch.Tensor, label: torch.Tensor) -> dict:
    return classification_metrics(probability.clamp_min(1e-8).log(), label, 5)


def evaluate(values: dict, probabilities: dict[str, torch.Tensor]) -> tuple[dict, list[dict]]:
    condition_rows = {}
    for condition in CONDITIONS:
        selected = values["condition"] == condition
        condition_rows[condition] = {
            method: metrics(probability[selected], values["label"][selected])
            for method, probability in probabilities.items()
        }

    subject_rows = []
    scopes = {
        "dynamic_all": np.isin(values["condition"], DYNAMIC_CONDITIONS),
        "eeg_interruption_300s": values["condition"] == "hard_eeg_10ep",
    }
    for subject in sorted(np.unique(values["subject"]).tolist()):
        for scope, scope_mask in scopes.items():
            selected = (values["subject"] == subject) & scope_mask
            for method, probability in probabilities.items():
                subject_rows.append(
                    {
                        "subject": subject,
                        "scope": scope,
                        "method": method,
                        "macro_f1": metrics(
                            probability[selected], values["label"][selected]
                        )["macro_f1"],
                    }
                )
    return condition_rows, subject_rows


def main() -> int:
    args = parse_args()
    protocol = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    checkpoint = load_checkpoint(args.checkpoint)
    config = copy.deepcopy(checkpoint["config"])
    for section in ("experiment", "data", "physiology", "train"):
        config.setdefault(section, {}).update(protocol.get(section, {}))
    config["experiment"]["seed"] = int(args.run)
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
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    device = resolve_device(args.device)

    model = build_student(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.requires_grad_(False)
    adapter_checkpoint = load_checkpoint(args.adapter_checkpoint)
    adapter = build_adapter(model, adapter_checkpoint, config, device)
    values = collect(model, adapter, sequence_dataset(config, "test"), config, device)

    gate_payload = load_checkpoint(args.gate_checkpoint)
    gate = DomainRecoveryGate(int(gate_payload["feature_count"])).to(device)
    gate.load_state_dict(gate_payload["model_state"], strict=True)
    learned, weight = apply_gate(
        gate,
        values,
        gate_payload["feature_mean"],
        gate_payload["feature_standard_deviation"],
        device,
    )
    probabilities = {
        "learned_recovery": learned,
        "fixed_half": 0.5 * values["recursive"] + 0.5 * values["persistence"],
        "recursive_belief": values["recursive"],
        "static_persistence": values["persistence"],
    }
    condition_rows, subject_rows = evaluate(values, probabilities)
    dynamic_average = {
        method: float(
            np.mean(
                [condition_rows[condition][method]["macro_f1"] for condition in DYNAMIC_CONDITIONS]
            )
        )
        for method in METHODS
    }
    payload = {
        "protocol": "PROJECT3_RECOVERY_FIXED_MIXTURE_CONFIRMATION_V1",
        "dataset": args.dataset_id,
        "run": seed,
        "backbone_frozen": True,
        "gate_frozen": True,
        "fixed_mixture_weight": 0.5,
        "dynamic_average_macro_f1": dynamic_average,
        "condition_metrics": condition_rows,
        "mean_learned_recursive_weight": float(weight.mean()),
        "subject_metrics": subject_rows,
        "test_split_status": "previously accessed external split; fixed post-freeze comparison",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "run": seed,
                "dynamic_average_macro_f1": dynamic_average,
                "eeg_interruption_300s": {
                    method: condition_rows["hard_eeg_10ep"][method]["macro_f1"]
                    for method in METHODS
                },
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
