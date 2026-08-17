from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from train_recursive_belief_filter import build_student, evaluate
from uniphysio_wm.engine import (
    load_checkpoint,
    physio_feature_sequence_dataset,
    resolve_device,
    seed_everything,
)
from uniphysio_wm.mainline import build_mainline_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    config = copy.deepcopy(checkpoint["config"])
    seed_everything(int(config["experiment"]["seed"]))
    device = resolve_device(str(config["train"].get("device", "cuda")))

    teacher_checkpoint = load_checkpoint(config["baseline"]["checkpoint_path"])
    teacher = build_mainline_model(config).to(device)
    teacher.load_state_dict(teacher_checkpoint["model_state"], strict=True)
    teacher.requires_grad_(False)
    teacher.eval()

    student = build_student(config).to(device)
    student.load_state_dict(checkpoint["model_state"], strict=True)
    student.requires_grad_(False)
    student.eval()
    validation_data = physio_feature_sequence_dataset(config, "val")
    validation = evaluate(teacher, student, validation_data, config, device)
    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "validation_subject_split": "val",
        "validation_sequences": len(validation_data),
        "test_split_accessed": False,
        "validation": validation,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"test_split_accessed": False, **validation["selection"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

