from pathlib import Path

import torch

from uniphysio_wm.baseline_models import CausalSequenceBaseline
from uniphysio_wm.config import load_config
from uniphysio_wm.models import AttentiveCNNSleepClassifier, CNNBiLSTMSleepClassifier


ROOT = Path(__file__).resolve().parents[1]


def test_conventional_sleep_baseline_shapes() -> None:
    signals = torch.randn(2, 1, 256)
    present = torch.ones(2, 1, dtype=torch.bool)
    bilstm = CNNBiLSTMSleepClassifier(1, channels=32, tokens=8, hidden_dim=16)
    attention = AttentiveCNNSleepClassifier(1, channels=32, tokens=8, layers=1, heads=4)
    assert bilstm(signals, present).shape == (2, 5)
    assert attention(signals, present).shape == (2, 5)


def test_sequence_baselines_are_history_only_and_shape_stable() -> None:
    signals = torch.randn(2, 4, 3, 256)
    present = torch.ones(2, 4, 3, dtype=torch.bool)
    for architecture in ("gru", "transformer"):
        model = CausalSequenceBaseline(
            modalities=3,
            horizons=(1, 2, 4),
            architecture=architecture,
            d_model=32,
            layers=1,
            heads=4,
        )
        output = model(signals, present)
        assert output["current_logits"].shape == (2, 5)
        assert output["future_logits"].shape == (2, 3, 5)
        assert output["state"].shape == (2, 32)


def test_paper_baseline_configs_are_valid() -> None:
    names = (
        "eeg_tcn.yaml",
        "multimodal_tcn.yaml",
        "multimodal_transformer.yaml",
        "eeg_cnn_bilstm.yaml",
        "eeg_attentive_cnn.yaml",
        "sequence_gru.yaml",
        "sequence_transformer.yaml",
    )
    for name in names:
        config = load_config(ROOT / "configs" / "baseline" / name)
        assert config["experiment"]["seed"] == 20260804
