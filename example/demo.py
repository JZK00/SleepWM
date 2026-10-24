"""Run the code on generated toy signals; no paper data or weights are used.

From the SleepWM directory after installing the package:
    python ../example/demo.py --output-dir ../demo_output --device cpu

This checks the software pipeline, not prediction quality or paper replication.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def make_recording(path: Path, seed: int, gap: bool = False) -> None:
    """Create 48 thirty-second epochs in volts, ordered EEG/ECG/EMG.

    These simple waveforms have no clinical interpretation. Labels deliberately
    cycle through W/N1/N2/N3/REM to exercise all five output classes.
    """
    rng = np.random.default_rng(seed)
    n_epochs, sample_rate, seconds = 48, 128, 30
    time = np.arange(sample_rate * seconds, dtype=np.float64) / sample_rate
    labels = ((np.arange(n_epochs) // 8) % 5).astype(np.int64)
    signals = np.empty((n_epochs, 3, time.size), dtype=np.float32)
    for epoch, stage in enumerate(labels):
        eeg_frequency = (10.0, 6.0, 4.0, 1.5, 7.0)[stage]
        eeg = (35e-6 * np.sin(2 * np.pi * eeg_frequency * time)
               + 9e-6 * np.sin(2 * np.pi * 18 * time)
               + rng.normal(0, 4e-6, time.size))
        rr = 0.85 + 0.025 * np.sin(epoch / 3.0) + 0.01 * seed
        phase = (time + 0.1) % rr
        ecg = (0.9e-3 * np.exp(-0.5 * ((phase - 0.20) / 0.012) ** 2)
               - 0.15e-3 * np.exp(-0.5 * ((phase - 0.23) / 0.015) ** 2)
               + 0.18e-3 * np.exp(-0.5 * ((phase - 0.43) / 0.050) ** 2)
               + rng.normal(0, 5e-6, time.size))
        emg_amplitude = (22e-6, 14e-6, 10e-6, 8e-6, 4e-6)[stage]
        emg = emg_amplitude * (np.sin(2 * np.pi * 26 * time)
                              + 0.5 * rng.normal(size=time.size))
        signals[epoch] = np.stack((eeg, ecg, emg))
    starts = np.arange(n_epochs, dtype=np.float64) * seconds
    if gap:
        # An omitted 30-second epoch: windows must not cross this gap.
        starts[26:] += seconds
    np.savez_compressed(
        path, signals=signals, labels=labels,
        modality_present=np.ones((n_epochs, 3), dtype=bool),
        epoch_start_seconds=starts,
    )


def run(script: Path, *arguments: object) -> None:
    command = [sys.executable, str(script), *(str(value) for value in arguments)]
    print("Running:", " ".join(command), flush=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, cwd=script.parent, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory for synthetic data and outputs; must not exist")
    parser.add_argument("--device", choices=("cpu",), default="cpu",
                        help="The small software demonstration runs on CPU")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    project = Path(__file__).resolve().parents[1] / "SleepWM"
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite existing directory: {output}")
    output.mkdir(parents=True)
    raw = output / "synthetic_raw"
    raw.mkdir()
    manifest = raw / "manifest.csv"
    rows = []
    for seed, (participant, split) in enumerate(
        (("train01", "train"), ("train02", "train"), ("val01", "val"), ("test01", "test")),
        start=1,
    ):
        record = f"synthetic_{participant}"
        make_recording(raw / f"{record}.npz", seed, gap=split == "val")
        rows.append({"record_id": record, "subject": f"synthetic:{participant}",
                     "split": split, "npz_path": f"{record}.npz"})
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["record_id", "subject", "split", "npz_path"])
        writer.writeheader()
        writer.writerows(rows)

    prepared, prepared_test = output / "prepared", output / "prepared_test"
    statistics = prepared / "statistics.json"
    run(project / "prepare_data.py", "--manifest", manifest, "--output-dir", prepared)
    from sleepwm.data import SleepDataset
    validation = SleepDataset(prepared / "manifest.csv", statistics, "val")
    # At stride 8, starts 0/8/16/24 are candidates. Only start 0 stays on one
    # continuous segment; the other three cross the deliberately inserted gap.
    if validation.index != [(0, 0)]:
        raise RuntimeError(f"Unexpected validation windows across the time gap: {validation.index}")
    # Test feature preparation reuses the training statistics; it never refits.
    run(project / "prepare_data.py", "--manifest", manifest, "--output-dir", prepared_test,
        "--splits", "test", "--statistics", statistics)
    if json.loads(statistics.read_text(encoding="utf-8")) != json.loads(
        (prepared_test / "statistics.json").read_text(encoding="utf-8")
    ):
        raise RuntimeError("Preparing test data changed the frozen training statistics")
    run_dir = output / "run"
    run(project / "train.py", "--mode", "supervised", "--manifest", prepared / "manifest.csv", "--statistics", statistics,
        "--output-dir", run_dir, "--epochs", 1, "--batch-size", 2,
        "--device", args.device, "--compact", "--max-batches", 2)
    metrics = output / "synthetic_validation.json"
    run(project / "evaluate.py", "--checkpoint", run_dir / "best.pt",
        "--manifest", prepared / "manifest.csv", "--statistics", statistics,
        "--split", "val", "--device", args.device, "--output", metrics)
    summary = {
        "scope": "Synthetic software smoke test; not clinical evidence or paper replication",
        "data": "48 epochs per participant, 30 seconds per epoch, 128 Hz, volts",
        "subjects": {"train": 2, "val": 1, "test": 1},
        "validation_time_gap": "30 seconds missing before epoch index 26",
        "gap_check": "One valid validation window; three windows crossing the gap excluded",
        "model": "Random initialization, compact dimensions, one supervised epoch, at most two training batches",
        "frozen_statistics_reused_for_test_preparation": True,
        "evaluation_split": "val",
        "checkpoint": str(run_dir / "best.pt"),
        "metrics": str(metrics),
    }
    (output / "demo_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nSynthetic smoke test completed. Validation output: {metrics}")
    print("These toy-data metrics do not estimate clinical accuracy or reproduce the paper.")


if __name__ == "__main__":
    main()
