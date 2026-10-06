from __future__ import annotations

"""Evaluate final SleepWM on complete, fixed-loss or retained-view conditions."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import yaml

from sleepwm.model import SleepWM
from sleepwm.protocols import dynamic_view
from sleepwm.protocols import average_precision, load_label_map
from sleepwm.training.latent_hazard_safe_adapter import evaluation_specs, transition_targets
from sleepwm.engine import data_loader, physio_feature_sequence_dataset
from sleepwm.metrics import classification_metrics
from sleepwm.masking import DynamicObservationSpec
from sleepwm.metrics import standardized_physiology_metrics


@torch.inference_mode()
def evaluate(model, dataset, config, device, spec):
    label_map = load_label_map(dataset)
    horizons = tuple(config["data"]["future_horizons"])
    fields = {k: [] for k in ("stage", "phys", "risk", "labels", "targets", "valid", "events")}
    for batch in data_loader(dataset, config, shuffle=False):
        x = batch["history_signals"].to(device, dtype=torch.float32)
        present = batch["history_present"].to(device, dtype=torch.bool)
        x, present, _ = dynamic_view(x, present, tuple(config["data"]["modalities"]), spec)
        out = model(x, present)
        for k, v in (("stage", out["stage_logits"]), ("phys", out["future_physiology"]), ("risk", out["transition_risk"]),
                     ("labels", batch["future_labels"]), ("targets", batch["future_physiology"]), ("valid", batch["future_physiology_valid"])):
            fields[k].append(v.cpu())
        fields["events"].append(transition_targets(batch, label_map, horizons, int(config["data"]["history_epochs"]), torch.device("cpu")))
    if not fields["stage"]:
        raise ValueError("No eligible histories in the selected split")
    v = {k: torch.cat(a) for k, a in fields.items()}
    phys = standardized_physiology_metrics(v["phys"], v["targets"], v["valid"], tuple(config["physiology"]["feature_names"]),
                                          {k: tuple(a) for k,a in config["physiology"]["feature_groups"].items()}, horizons)
    by_horizon = {}
    for i, h in enumerate(horizons):
        by_horizon[str(h)] = {
            "stage_macro_f1": classification_metrics(v["stage"][:,i], v["labels"][:,i], int(config["data"]["num_classes"]))["macro_f1"],
            "future_physiology_mae": phys["by_horizon"][str(h)]["mean_normalized_mae"],
            "transition_auprc": average_precision(v["events"][:,i].numpy().astype(bool), v["risk"][:,i].numpy()),
        }
    result = {k: float(np.mean([by_horizon[str(h)][k] for h in (1,2,4)])) for k in by_horizon["1"]}
    return {**result, "by_horizon": by_horizon, "histories": len(v["labels"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--data-config", help="Optional YAML overriding data/physiology path fields")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--protocol", choices=("primary", "multiview"), default="primary")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", help="Required for dataset evaluation")
    parser.add_argument("--synthetic", action="store_true", help="Check four synthetic histories without opening a dataset")
    args = parser.parse_args()
    model = SleepWM.from_checkpoints(json.loads(Path(args.checkpoints).read_text()), args.device)
    if args.synthetic:
        print(json.dumps(synthetic_check(model, args.device), indent=2))
        return
    if not args.output:
        parser.error("--output is required for dataset evaluation")
    config = model.config
    if args.data_config:
        overrides = yaml.safe_load(Path(args.data_config).read_text())
        for key in ("data", "physiology"):
            config[key].update(overrides.get(key, {}))
    config["train"].update(device=args.device, batch_size=args.batch_size, num_workers=args.workers)
    dataset = physio_feature_sequence_dataset(config, args.split)
    modalities = tuple(config["data"]["modalities"])
    specs = [None]
    if args.protocol == "primary":
        specs.extend(evaluation_specs(modalities))
    else:
        for retained in (("EEG",),("ECG",),("EMG",),("EEG","ECG"),("EEG","EMG"),("ECG","EMG")):
            for duration in (1,2,3,4,6,10):
                specs.append(DynamicObservationSpec("retain_"+"_".join(retained)+f"_{duration}ep", {m:duration for m in modalities if m not in retained}))
    conditions = {"full_observation" if spec is None else spec.name: evaluate(model, dataset, config, args.device, spec) for spec in specs}
    report = {"split":args.split, "test_split_accessed":args.split == "test", "protocol":args.protocol, "conditions":conditions}
    if args.protocol == "primary":
        report["dynamic"] = {k:float(np.mean([row[k] for name,row in conditions.items() if name != "full_observation"]))
                             for k in ("stage_macro_f1", "future_physiology_mae", "transition_auprc")}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2))
    print(path)


def synthetic_check(model, device="cpu"):
    torch.manual_seed(0)
    data = model.config["data"]
    signals = torch.randn(1, int(data["history_epochs"]), len(data["modalities"]),
                          int(data["sample_rate"] * data["epoch_seconds"]), device=device)
    reports = []
    for condition in ("complete", "all_absent_120s", "eeg_absent_120s", "return_after_gap"):
        present = torch.ones(signals.shape[:3], dtype=torch.bool, device=device)
        if condition == "all_absent_120s": present[:, -4:] = False
        if condition == "eeg_absent_120s": present[:, -4:, 0] = False
        if condition == "return_after_gap": present[:, -8:-4] = False
        out = model(signals * present.unsqueeze(-1), present)
        assert all(torch.isfinite(v).all() for v in out.values())
        risk = out["transition_risk"]
        assert ((risk >= 0) & (risk <= 1)).all()
        assert (risk[:, 1:] >= risk[:, :-1] - 1e-7).all()
        assert torch.allclose(out["stage_probabilities"].sum(-1), torch.ones_like(risk), atol=1e-6)
        reports.append({"condition":condition, "shapes":{k:list(v.shape) for k,v in out.items()}})
    return {"synthetic_only":True, "parameters":sum(p.numel() for p in model.parameters()), "checks":reports}


if __name__ == "__main__":
    main()
