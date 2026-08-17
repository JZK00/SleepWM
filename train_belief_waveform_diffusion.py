from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
import yaml

from evaluate_belief_outcomes import dynamic_view
from evaluate_probabilistic_belief_waveforms import cpu_tensor_tree, decode_base
from train_belief_waveform_adapter import build_adapter, selected_targets, training_specs
from train_recursive_belief_filter import build_student
from train_waveform_diffusion_refiner import (
    build_emg_calibrated_conditions,
    build_structural_condition,
)
from uniphysio_wm.belief_waveform_diffusion import (
    append_emg_burst_condition,
    build_frozen_structural_refiner,
    project3_joint_structure_loss,
    residual_strength_mask,
)
from uniphysio_wm.engine import (
    data_loader,
    load_checkpoint,
    physio_feature_sequence_dataset,
    resolve_device,
    save_checkpoint,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Project3 belief-conditioned waveform diffusion refiner."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    protocol_path = Path(args.config)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    diffusion = protocol["belief_diffusion"]
    po3_path = Path(diffusion["po3_checkpoint"])
    po5_path = Path(diffusion["po5_checkpoint"])
    initial_refiner_path = Path(diffusion["initial_refiner_checkpoint"])
    calibration_path = Path(diffusion["frozen_calibration_checkpoint"])
    po3_checkpoint = load_checkpoint(po3_path)
    po5_checkpoint = load_checkpoint(po5_path)
    initial_refiner_checkpoint = load_checkpoint(initial_refiner_path)
    calibration_checkpoint = load_checkpoint(calibration_path)

    config = copy.deepcopy(po3_checkpoint["config"])
    for section in ("experiment", "data", "physiology", "train"):
        config.setdefault(section, {}).update(protocol.get(section, {}))
    architecture = copy.deepcopy(initial_refiner_checkpoint["config"]["diffusion"])
    architecture.update(diffusion)
    config["diffusion"] = architecture
    if args.device:
        config["train"]["device"] = args.device
    if args.epochs is not None:
        config["train"]["epochs"] = int(args.epochs)
    if args.seed is not None:
        config["experiment"]["seed"] = int(args.seed)
    if args.output_dir:
        config["train"]["output_dir"] = args.output_dir
    if args.smoke:
        config["train"]["epochs"] = 1
        config["train"]["num_workers"] = 0

    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    device = resolve_device(str(config["train"].get("device", "cuda")))
    output_dir = Path(config["train"]["output_dir"])
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
        initial_refiner_checkpoint,
        int(model.encoder.config.d_model),
        int(config["physiology"]["physiology_dynamics_dim"]),
        structural_condition_channels=int(
            diffusion.get("structural_condition_channels", 2)
        ),
    ).to(device)
    refiner.requires_grad_(False)
    train_modalities = tuple(str(value) for value in diffusion["train_modalities"])
    for modality in train_modalities:
        refiner.denoisers[modality].requires_grad_(True)
    trainable = [parameter for parameter in refiner.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(diffusion["learning_rate"]),
        weight_decay=float(diffusion.get("weight_decay", 1e-4)),
    )

    dataset = physio_feature_sequence_dataset(config, "train")
    if args.smoke:
        dataset = torch.utils.data.Subset(dataset, range(min(96, len(dataset))))
    modalities = tuple(config["data"]["modalities"])
    active_indices = [modalities.index(modality) for modality in train_modalities]
    horizons = tuple(int(value) for value in config["data"]["future_horizons"])
    maximum_samples = int(model.waveform_decoder.max_samples)
    sample_rate = int(config["data"]["sample_rate"])
    patch_samples = int(model.waveform_decoder.patch_samples)
    specs = training_specs(modalities)
    preferred_specs = [
        next(spec for spec in specs if spec is not None and spec.name == "hard_all_4ep"),
        next(spec for spec in specs if spec is not None and spec.name == "hard_all_4ep"),
        None,
        next(spec for spec in specs if spec is not None and spec.name == "hard_eeg_4ep"),
        next(spec for spec in specs if spec is not None and spec.name == "hard_emg_4ep"),
    ]
    loss_weights = {
        name: float(value) for name, value in diffusion["loss_weights"].items()
    }
    history = []

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        refiner.train()
        totals = {}
        samples_seen = 0
        for batch_index, batch in enumerate(data_loader(dataset, config, shuffle=True)):
            natural_signals = batch["history_signals"].to(
                device=device, dtype=torch.float32
            )
            natural_present = batch["history_present"].to(
                device=device, dtype=torch.bool
            )
            spec = preferred_specs[(batch_index + epoch) % len(preferred_specs)]
            signals, present, _ = dynamic_view(
                natural_signals, natural_present, modalities, spec
            )
            target, valid = selected_targets(batch, device, maximum_samples)
            with torch.no_grad():
                base, probabilities, recent, output, route = decode_base(
                    model, adapter, po5_checkpoint, signals, present, horizons
                )
                structural_condition = build_structural_condition(
                    calibration_checkpoint["calibration_state"],
                    base.detach().float().cpu(),
                    cpu_tensor_tree(probabilities),
                    recent.detach().float().cpu(),
                    modalities,
                    sample_rate,
                    patch_samples,
                ).to(device=device, dtype=torch.float32)
                if refiner.structural_condition_channels > structural_condition.shape[2]:
                    _, _, burst_probability, _ = build_emg_calibrated_conditions(
                        calibration_checkpoint["calibration_state"],
                        base.detach().float().cpu(),
                        cpu_tensor_tree(probabilities),
                        recent.detach().float().cpu(),
                        modalities,
                        sample_rate,
                        patch_samples,
                    )
                    structural_condition = append_emg_burst_condition(
                        structural_condition,
                        burst_probability.to(device=device, dtype=torch.float32),
                        modalities,
                        patch_samples,
                        refiner.structural_condition_channels,
                    )
            mask = residual_strength_mask(
                base, modalities, diffusion["residual_strengths"]
            )
            predicted = refiner.training_prediction(
                base,
                recent,
                target,
                structural_condition,
                route["states"][:, 0],
                output["physiology_dynamics_states"][:, 0],
                active_modalities=train_modalities,
                residual_mask=mask,
            )
            active_valid = torch.zeros_like(valid)
            active_valid[:, active_indices] = valid[:, active_indices]
            losses = project3_joint_structure_loss(
                predicted,
                target,
                active_valid,
                modalities,
                sample_rate,
                patch_samples,
                tuple(int(value) for value in diffusion["multi_resolution_fft_sizes"]),
                loss_weights,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                trainable, float(config["train"].get("grad_clip", 1.0))
            )
            optimizer.step()
            count = len(signals)
            samples_seen += count
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * count
            if (batch_index + 1) % 50 == 0:
                print(
                    f"epoch={epoch} batches={batch_index + 1} "
                    f"loss={totals['loss'] / samples_seen:.6f}",
                    flush=True,
                )
        epoch_metrics = {
            name: value / max(1, samples_seen) for name, value in totals.items()
        }
        history.append({"epoch": epoch, "train": epoch_metrics})
        print(json.dumps(history[-1], indent=2), flush=True)
        save_checkpoint(
            output_dir / f"epoch_{epoch:02d}.pt",
            {
                "experiment_id": config["experiment"]["id"],
                "epoch": epoch,
                "refiner_state": refiner.state_dict(),
                "config": config,
                "po3_checkpoint": str(po3_path),
                "po5_checkpoint": str(po5_path),
                "initial_refiner_checkpoint": str(initial_refiner_path),
                "test_split_accessed": False,
            },
        )

    final_path = output_dir / "final.pt"
    save_checkpoint(
        final_path,
        {
            "experiment_id": config["experiment"]["id"],
            "epoch": int(config["train"]["epochs"]),
            "refiner_state": refiner.state_dict(),
            "config": config,
            "po3_checkpoint": str(po3_path),
            "po5_checkpoint": str(po5_path),
            "initial_refiner_checkpoint": str(initial_refiner_path),
            "test_split_accessed": False,
        },
    )
    metrics = {
        "history": history,
        "train_sequences": len(dataset),
        "final_checkpoint": str(final_path),
        "final_checkpoint_sha256": sha256(final_path),
        "test_split_accessed": False,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "resolved_config.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
