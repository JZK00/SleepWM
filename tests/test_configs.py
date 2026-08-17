from pathlib import Path

import yaml

from uniphysio_wm.config import load_config


def test_all_experiment_configs_are_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "configs").rglob("*.yaml"))
    assert paths
    identifiers = []
    for path in paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict), path
        identifier = payload.get("experiment", {}).get("id", payload.get("protocol"))
        assert isinstance(identifier, str), path
        identifiers.append(identifier)
        if all(section in payload for section in ("data", "model", "train")):
            load_config(path)
    assert len(identifiers) == len(set(identifiers))
