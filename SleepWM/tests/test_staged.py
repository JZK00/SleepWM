"""Check staged checkpoint boundaries and that adaptation cannot update the backbone."""
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
import numpy as np
import torch

from sleepwm import default_config
from sleepwm.builders import build_mainline_model
from sleepwm.stages import partial_observation_filter as carry
from sleepwm.stages import recursive_belief_filter as belief
from sleepwm.features import FEATURE_NAMES
from sleepwm.staged_support import PhysioFeatureSequenceDataset, average_precision, load_label_map
from sleepwm import staged


@pytest.fixture
def reference(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(19)
    config = default_config(compact=True)
    path = tmp_path / "reference.pt"
    model = build_mainline_model(config)
    torch.save({"config": config, "model_state": model.state_dict()}, path)
    config["baseline"] = {"checkpoint_path": str(path)}
    return config


def assert_frozen_reference(teacher):
    assert not any(parameter.requires_grad for parameter in teacher.parameters())
    assert not any(module.training for module in teacher.modules())


def check_update_is_confined_to(student, prefixes, exact_names, state_key):
    """A real backward/update must change an adapter and leave all frozen state intact."""
    expected = {name for name, _ in student.named_parameters()
                if name in exact_names or name.startswith(prefixes)}
    actual = {name for name, parameter in student.named_parameters() if parameter.requires_grad}
    assert actual == expected and actual
    before = {name: value.detach().clone() for name, value in student.state_dict().items()}
    optimizer = torch.optim.AdamW(
        (parameter for parameter in student.parameters() if parameter.requires_grad), lr=1e-3
    )
    signals = torch.randn(2, 20, 3, 3840)
    present = torch.ones(2, 20, 3, dtype=torch.bool)
    present[0, -4:, 0] = False
    present[1, -6:-2, :] = False
    signals = signals.masked_fill(~present.unsqueeze(-1), 0.0)
    output = student.rollout_context(signals, present)
    loss = output[state_key].square().mean() + output["recursive_log_variance"].square().mean()
    assert torch.isfinite(loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
               for parameter in student.parameters() if parameter.requires_grad)
    assert all(parameter.grad is None for parameter in student.parameters() if not parameter.requires_grad)
    optimizer.step()
    after = student.state_dict()
    assert any(not torch.equal(before[name], after[name]) for name in actual)
    for name, value in before.items():
        if name not in actual:
            assert torch.equal(value, after[name]), f"Frozen parameter/buffer changed: {name}"


def test_carry_training_changes_only_carry_and_uncertainty(reference):
    teacher, student, _ = carry.load_models(reference, torch.device("cpu"))
    assert_frozen_reference(teacher)
    carry.set_partial_train_mode(student)
    assert not student.encoder.training
    assert student.carry_state_filter.training
    assert student.partial_uncertainty_head.training
    check_update_is_confined_to(
        student,
        ("carry_state_filter.", "partial_uncertainty_head.", "partial_uncertainty_state_norm."),
        {"log_freshness_decay", "uncertainty_bias", "uncertainty_age_log_scale",
         "uncertainty_horizon_log_scale"},
        "predicted_states",
    )


def test_belief_training_keeps_reference_and_carry_frozen(reference, tmp_path):
    _, carry_student, _ = carry.load_models(reference, torch.device("cpu"))
    path = tmp_path / "carry.pt"
    torch.save({"config": reference, "model_state": carry_student.state_dict()}, path)
    reference["belief_filter"]["initial_checkpoint"] = str(path)
    teacher, student, _ = belief.load_models(reference, torch.device("cpu"))
    assert_frozen_reference(teacher)
    belief.set_belief_train_mode(student)
    assert not student.encoder.training
    assert not student.carry_state_filter.training
    assert student.belief_transition.training and student.belief_correction_gate.training
    check_update_is_confined_to(
        student, ("belief_transition.", "belief_correction_gate."), set(), "belief_trajectory"
    )


def test_belief_initialization_rejects_unrelated_checkpoint_parameters(reference, tmp_path):
    _, student, _ = carry.load_models(reference, torch.device("cpu"))
    state = copy.deepcopy(student.state_dict())
    state["unrelated_model.weight"] = torch.zeros(1)
    path = tmp_path / "wrong_carry.pt"
    torch.save({"config": reference, "model_state": state}, path)
    reference["belief_filter"]["initial_checkpoint"] = str(path)
    with pytest.raises(ValueError, match="initialization mismatch"):
        belief.load_models(reference, torch.device("cpu"))


def test_staged_choices_can_be_listed_without_data_or_checkpoints():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "train.py"), "--mode", "staged", "--list"],
        cwd=root, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for stage in ("carry_align", "carry_final", "belief", "outcome", "direct",
                  "direct_event", "fusion", "event", "assemble"):
        assert stage in result.stdout


def test_staged_data_adapter_preserves_time_targets_and_reads_only_its_split(tmp_path):
    count = 28
    onsets = np.arange(count) * 30.0
    raw = tmp_path / "train.npz"
    features = tmp_path / "train_features.npz"
    signals = np.broadcast_to(np.arange(count)[:, None, None] * 1e-5,
                              (count, 3, 3840)).astype(np.float32).copy()
    np.savez(raw, signals=signals, labels=np.arange(count) % 5,
             modality_present=np.ones((count, 3), dtype=bool), epoch_start_seconds=onsets)
    np.savez(features, features=np.broadcast_to(np.arange(count)[:, None], (count, 11)),
             valid=np.ones((count, 11), dtype=bool), epoch_start_seconds=onsets)
    manifest = tmp_path / "manifest.csv"
    fields = ["record_id", "subject", "split", "npz_path", "features_path"]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(record_id="train", subject="synthetic:train", split="train",
                             npz_path=str(raw), features_path=str(features)))
        # If either dataset construction or event-label loading opens another split,
        # these deliberately absent files make the test fail.
        writer.writerow(dict(record_id="test", subject="synthetic:test", split="test",
                             npz_path="unavailable_test.npz", features_path="unavailable_features.npz"))
    statistics = {"schema_version": 1, "sample_rate": 128, "epoch_seconds": 30,
                  "feature_names": list(FEATURE_NAMES),
                  "signal_mean": [0.] * 3, "signal_std": [1.] * 3, "signal_count": [count] * 3,
                  "feature_mean": [0.] * 11, "feature_std": [1.] * 11,
                  "feature_count": [count] * 11}
    stats_path = tmp_path / "statistics.json"
    stats_path.write_text(json.dumps(statistics), encoding="utf-8")
    config = default_config(compact=True)
    config["data"].update(manifest_path=str(manifest), normalization_path=str(stats_path),
                          sequence_stride=8, sequence_start_offset=2)
    dataset = PhysioFeatureSequenceDataset(config, "train")
    assert set(load_label_map(dataset)) == {"train"}
    assert dataset.index == [(0, 2)]
    sample = dataset[0]
    assert sample["history_signals"].shape == (20, 3, 3840)
    assert sample["future_signals"].shape == (3, 3, 3840)
    assert sample["future_labels"].tolist() == [2, 3, 0]  # Raw indices 22, 23, 25.
    assert sample["history_labels"][-1].item() == 1  # Raw index 21.
    assert sample["current_physiology"].tolist() == [21.] * 11
    assert sample["future_physiology"][:, 0].tolist() == [22., 23., 25.]
    assert dataset.future_label_counts.tolist() == [1, 0, 1, 1, 0]


def test_staged_event_selection_handles_ties_and_event_free_samples():
    target = np.array([1, 0])
    score = np.array([0.5, 0.5])
    assert average_precision(target, score) == 0.5
    assert average_precision(target[::-1], score[::-1]) == 0.5
    # Staged selection uses zero for an event-free subset, as documented by the API.
    assert average_precision(np.zeros(2, dtype=int), score) == 0.0


@pytest.fixture
def normalization(tmp_path):
    stats = {"schema_version": 1, "sample_rate": 128, "epoch_seconds": 30,
             "feature_names": list(FEATURE_NAMES),
             "signal_mean": [0.] * 3, "signal_std": [1.] * 3, "signal_count": [20] * 3,
             "feature_mean": [0.] * 11, "feature_std": [1.] * 11,
             "feature_count": [20] * 11}
    path = tmp_path / "reference_statistics.json"
    path.write_text(json.dumps(stats), encoding="utf-8")
    return stats, path


def test_staged_requires_and_matches_reference_normalization(normalization):
    stats, path = normalization
    with pytest.raises(ValueError, match="normalization metadata"):
        staged.check_normalization({}, stats)
    staged.check_normalization({}, stats, path)
    staged.check_normalization({"normalization": stats}, stats, path)
    refitted = copy.deepcopy(stats)
    refitted["signal_std"][0] = 2.0
    with pytest.raises(ValueError, match="Normalization mismatch"):
        staged.check_normalization({"normalization": stats}, refitted)
    with pytest.raises(ValueError, match="Reference checkpoint and reference statistics"):
        staged.check_normalization({"normalization": refitted}, refitted, path)


def test_staged_refuses_to_overwrite_existing_stage(reference, normalization, tmp_path):
    _, statistics_path = normalization
    output = tmp_path / "run"
    previous = output / "carry_align" / "best.pt"
    previous.parent.mkdir(parents=True)
    previous.write_bytes(b"existing selected checkpoint")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("record_id,subject,split,npz_path\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite stage directory"):
        staged.main([
            "--stage", "carry_align", "--reference", reference["baseline"]["checkpoint_path"],
            "--reference-statistics", str(statistics_path), "--statistics", str(statistics_path),
            "--manifest", str(manifest), "--output-dir", str(output), "--device", "cpu",
        ])
    assert previous.read_bytes() == b"existing selected checkpoint"
    assert not (output / "run.json").exists()


def test_staged_dependency_checks_selected_bytes_seed_and_horizons(normalization, tmp_path):
    stats, _ = normalization
    config = default_config(compact=True)
    config["experiment"] = {"seed": 42}
    reference_data = copy.deepcopy(config["data"])
    path = staged.checkpoint_path(tmp_path, "belief", 42)
    path.parent.mkdir(parents=True)
    payload = {"config": config, "normalization": stats, "epoch": 1}
    def save_and_record():
        torch.save(payload, path)
        return {"stages": {"belief": {"sha256": staged.fingerprint(path)}}}
    record = save_and_record()
    assert staged.verified_dependency(tmp_path, "belief", 42, stats, record, reference_data)["epoch"] == 1
    # A replaced file must fail even if it is still a valid, same-shaped checkpoint.
    payload["epoch"] = 2
    torch.save(payload, path)
    with pytest.raises(ValueError, match="has changed since selection"):
        staged.verified_dependency(tmp_path, "belief", 42, stats, record, reference_data)
    payload["config"]["experiment"]["seed"] = 43
    record = save_and_record()
    with pytest.raises(ValueError, match="different seed"):
        staged.verified_dependency(tmp_path, "belief", 42, stats, record, reference_data)
    payload["config"]["experiment"]["seed"] = 42
    payload["config"]["data"]["future_horizons"] = [1, 2, 4, 10]
    record = save_and_record()
    with pytest.raises(ValueError, match="different future_horizons"):
        staged.verified_dependency(tmp_path, "belief", 42, stats, record, reference_data)
