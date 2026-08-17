from __future__ import annotations

import torch

from .probabilistic_waveform import (
    probabilistic_waveform_loss,
    probabilistic_waveform_metrics,
)
from .waveform_metrics import waveform_forecast_metrics


def predict_mainline_cache(
    model,
    cache: dict,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    model.waveform_decoder.eval()
    waveforms = []
    accumulated = {}
    sample_count = len(cache["shared_state"])
    with torch.no_grad():
        for start in range(0, sample_count, int(batch_size)):
            stop = min(start + int(batch_size), sample_count)
            prediction, _, probabilities = model.waveform_decoder(
                cache["recent_waveform"][start:stop].to(device, torch.float32),
                cache["shared_state"][start:stop].to(device, torch.float32),
                cache["dynamics_state"][start:stop].to(device, torch.float32),
                return_structure=True,
                return_probabilities=True,
                modality_availability=cache["modality_availability"][start:stop].to(
                    device, torch.float32
                ),
            )
            waveforms.append(prediction.cpu())
            for modality, values in probabilities.items():
                accumulated.setdefault(modality, {})
                for name, value in values.items():
                    accumulated[modality].setdefault(name, []).append(value.cpu())
    probability_output = {
        modality: {
            name: torch.cat(values) for name, values in modality_values.items()
        }
        for modality, modality_values in accumulated.items()
    }
    return torch.cat(waveforms), probability_output


def evaluate_mainline_cache(
    model,
    cache: dict,
    config: dict,
    device: torch.device,
) -> dict:
    waveform = config["waveform"]
    prediction, probabilities = predict_mainline_cache(
        model, cache, int(config["train"]["batch_size"]), device
    )
    target = cache["target_waveform"].float()
    recent = cache["recent_waveform"].float()
    valid = cache["valid"]
    modalities = tuple(config["data"]["modalities"])
    sample_rate = int(config["data"]["sample_rate"])
    horizons = tuple(int(value) for value in waveform["horizons_seconds"])
    probability_losses = probabilistic_waveform_loss(
        probabilities,
        target,
        valid,
        modalities,
        sample_rate,
        horizons,
        int(waveform["patch_samples"]),
    )
    amplitude = {}
    for modality_index, modality in enumerate(modalities):
        selected = valid[:, modality_index]
        generated_rms = (
            prediction[selected, modality_index].square().mean(-1).sqrt().mean()
        )
        target_rms = target[selected, modality_index].square().mean(-1).sqrt().mean()
        amplitude[modality] = {
            "generated_rms": float(generated_rms),
            "target_rms": float(target_rms),
            "generated_to_target_ratio": float(
                generated_rms / target_rms.clamp_min(1e-8)
            ),
        }
    return {
        "model_waveform": waveform_forecast_metrics(
            prediction, target, valid, modalities, sample_rate, horizons
        ),
        "repeat_last_window": waveform_forecast_metrics(
            recent, target, valid, modalities, sample_rate, horizons
        ),
        "probability": probabilistic_waveform_metrics(
            probabilities,
            target,
            recent,
            valid,
            modalities,
            sample_rate,
            horizons,
            int(waveform["patch_samples"]),
        ),
        "mean_probability_loss": float(probability_losses["loss"]),
        "amplitude": amplitude,
    }
