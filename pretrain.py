from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.nn.functional as F

from uniphysio_wm.config import load_config, with_manifest, with_normalization
from uniphysio_wm.engine import (
    AverageMeter,
    data_loader,
    epoch_dataset,
    prepare_run,
    resolve_device,
    save_checkpoint,
    seed_everything,
    write_json,
)
from uniphysio_wm.masking import random_modality_presence, random_span_mask
from uniphysio_wm.models import MaskedMultiModalModel, MultiModalEncoder, observation_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain the multimodal observation encoder.")
    parser.add_argument("--config", required=True, help="pretraining YAML configuration")
    parser.add_argument("--manifest", help="override data.manifest_path")
    parser.add_argument("--normalization", help="train-split normalization JSON")
    parser.add_argument("--smoke", action="store_true", help="run one epoch with no worker processes")
    parser.add_argument("--seed", type=int, help="override experiment seed")
    parser.add_argument("--epochs", type=int, help="override training epochs")
    parser.add_argument("--output-dir", help="override train.output_dir")
    return parser.parse_args()


def run_epoch(model, loader, device, config, optimizer=None, mask_seed=None):
    training = optimizer is not None
    model.train(training)
    total_meter = AverageMeter()
    reconstruction_meter = AverageMeter()
    waveform_meter = AverageMeter()
    latent_meter = AverageMeter()
    consistency_meter = AverageMeter()
    objective = config["objective"]
    waveform_weight = float(objective.get("waveform_reconstruction_weight", 1.0))
    latent_weight = float(objective.get("latent_reconstruction_weight", 0.0))
    consistency_weight = float(objective.get("consistency_weight", 0.0))
    teacher_momentum = float(objective.get("teacher_momentum", 0.99))
    modality_count = len(config["data"]["modalities"])
    generator = None
    if mask_seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(mask_seed))
    modality_meters = [AverageMeter() for _ in range(modality_count)]

    for step, batch in enumerate(loader):
        signals = batch["signals"].to(device=device, dtype=torch.float32)
        natural_present = batch["modality_present"].to(device=device, dtype=torch.bool)
        target_modality = step % modality_count
        patch_mask = random_span_mask(
            signals.shape[0],
            model.config.num_patches,
            float(objective["mask_ratio"]),
            int(objective["min_span"]),
            int(objective["max_span"]),
            device=device,
            generator=generator,
        )
        synthetic_present = random_modality_presence(
            signals.shape[0],
            modality_count,
            float(objective.get("auxiliary_modality_drop_probability", 0.0)),
            device,
            protected_modality=target_modality,
            generator=generator,
        )
        modality_present = natural_present & synthetic_present

        full_rows = torch.rand(signals.shape[0], device=device, generator=generator) < float(
            objective.get("full_modality_probability", 0.0)
        )
        other_available = natural_present.clone()
        other_available[:, target_modality] = False
        full_rows = full_rows & other_available.any(dim=1)
        if full_rows.any():
            modality_present[full_rows, target_modality] = False
            patch_mask[full_rows] = True

        target_latents = None
        full_representation = None
        if latent_weight > 0:
            target_latents = model.teacher_tokens(signals, natural_present)
            if consistency_weight > 0:
                full_representation = model.teacher_representation(signals, natural_present)
        elif consistency_weight > 0:
            encoder_training = model.encoder.training
            model.encoder.eval()
            with torch.no_grad():
                full_representation = model.encoder(signals, modality_present=natural_present)
            model.encoder.train(encoder_training)

        with torch.set_grad_enabled(training):
            target_latent = target_latents[:, target_modality] if target_latents is not None else None
            output = model(
                signals,
                target_modality,
                patch_mask,
                modality_present,
                target_latent=target_latent,
            )
            waveform_loss = output["reconstruction_loss"]
            latent_loss = output["latent_reconstruction_loss"]
            reconstruction_loss = waveform_weight * waveform_loss + latent_weight * latent_loss
            if consistency_weight > 0:
                if full_representation is None:
                    raise RuntimeError("consistency target was not computed")
                consistency_loss = 1.0 - F.cosine_similarity(
                    output["representation"], full_representation, dim=-1
                ).mean()
            else:
                consistency_loss = reconstruction_loss.new_zeros(())
            loss = reconstruction_loss + consistency_weight * consistency_loss
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_clip = float(config["train"].get("grad_clip", 0.0))
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if latent_weight > 0:
                    model.update_teacher(teacher_momentum)

        batch_size = signals.shape[0]
        total_meter.update(float(loss.detach().cpu()), batch_size)
        reconstruction_meter.update(float(reconstruction_loss.detach().cpu()), batch_size)
        waveform_meter.update(float(waveform_loss.detach().cpu()), batch_size)
        latent_meter.update(float(latent_loss.detach().cpu()), batch_size)
        consistency_meter.update(float(consistency_loss.detach().cpu()), batch_size)
        modality_meters[target_modality].update(float(reconstruction_loss.detach().cpu()), batch_size)
    return {
        "loss": total_meter.average,
        "reconstruction_loss": reconstruction_meter.average,
        "waveform_reconstruction_loss": waveform_meter.average,
        "latent_reconstruction_loss": latent_meter.average,
        "consistency_loss": consistency_meter.average,
        "reconstruction_by_modality": {
            name: modality_meters[index].average
            for index, name in enumerate(config["data"]["modalities"])
        },
    }


def main() -> int:
    args = parse_args()
    config = with_manifest(load_config(args.config), args.manifest)
    config = with_normalization(config, args.normalization)
    if config["experiment"]["kind"] != "pretrain":
        raise ValueError("pretrain.py requires experiment.kind=pretrain")
    if args.smoke:
        config = copy.deepcopy(config)
        config["train"]["epochs"] = 1
        config["train"]["num_workers"] = 0
    if args.seed is not None:
        config = copy.deepcopy(config)
        config["experiment"]["seed"] = int(args.seed)
    if args.epochs is not None:
        config = copy.deepcopy(config)
        config["train"]["epochs"] = int(args.epochs)
    if args.output_dir:
        config = copy.deepcopy(config)
        config["train"]["output_dir"] = str(Path(args.output_dir))

    seed_everything(int(config["experiment"]["seed"]))
    device = resolve_device(str(config["train"].get("device", "cuda")))
    run_dir = prepare_run(config)
    train_loader = data_loader(epoch_dataset(config, "train"), config, shuffle=True)
    val_loader = data_loader(epoch_dataset(config, "val"), config, shuffle=False)
    encoder = MultiModalEncoder(observation_config(config["data"], config["model"]))
    latent_prediction = float(config["objective"].get("latent_reconstruction_weight", 0.0)) > 0
    model = MaskedMultiModalModel(
        encoder,
        latent_prediction=latent_prediction,
        teacher_target=str(config["objective"].get("teacher_target", "tokenizer")),
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )

    best_loss = float("inf")
    best_metrics = {}
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, device, config, optimizer)
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            config,
            mask_seed=int(config["experiment"]["seed"]) + 1_000_003,
        )
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_recon={val_metrics['reconstruction_loss']:.6f} "
            f"val_wave={val_metrics['waveform_reconstruction_loss']:.6f} "
            f"val_latent={val_metrics['latent_reconstruction_loss']:.6f}"
        )
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_metrics = val_metrics
            save_checkpoint(
                run_dir / "best.pt",
                {
                    "experiment_id": config["experiment"]["id"],
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "encoder_state": model.pretrained_encoder_state(),
                    "student_encoder_state": model.encoder.state_dict(),
                    "encoder_source": "ema_teacher" if latent_prediction else "student",
                    "observation_config": model.config.to_dict(),
                    "validation_metrics": val_metrics,
                    "config": config,
                },
            )
    write_json(run_dir / "metrics.json", {"best_validation": best_metrics})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
