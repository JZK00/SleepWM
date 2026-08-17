from __future__ import annotations

import sys
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from uniphysio_wm.config import load_config  # noqa: E402


def main() -> int:
    paths = sorted((REPOSITORY_ROOT / "configs").rglob("*.yaml"))
    if not paths:
        raise RuntimeError("no YAML experiment configurations found")
    for path in paths:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError(f"configuration is not a mapping: {path}")
        identifier = config.get("experiment", {}).get("id", config.get("protocol"))
        if not isinstance(identifier, str):
            raise ValueError(f"configuration has no experiment id or protocol: {path}")
        if all(section in config for section in ("experiment", "data", "model", "train")):
            load_config(path)
        print(f"ok {path.relative_to(REPOSITORY_ROOT)} [{identifier}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
