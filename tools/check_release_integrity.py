from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED = (
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "pyproject.toml",
    "train_sleepwm.py",
    "test_sleepwm.py",
    "assets/sleepwm_architecture.png",
    "configs/sleepwm/train.yaml",
    "configs/sleepwm/evaluate.yaml",
    "docs/MODEL_CONTRACT.md",
    "docs/EXPERIMENT_PROTOCOL.md",
    "docs/REPRODUCIBILITY.md",
    "checkpoints/MANIFEST.json",
    "src/sleepwm/__init__.py",
)

PUBLIC_TEXT = (
    "README.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "data/README.md",
    "checkpoints/README.md",
    "docs/MODEL_CONTRACT.md",
    "docs/EXPERIMENT_PROTOCOL.md",
    "docs/RELEASE_STATUS.md",
    "docs/REPRODUCIBILITY.md",
)

FORBIDDEN_PUBLIC_PATTERNS = {
    "private server path": re.compile(r"/mnt/data/wurui|C:\\Users\\", re.I),
    "private server address": re.compile(r"\b10\.91\.\d{1,3}\.\d{1,3}\b"),
    "legacy public model name": re.compile(r"UniPhysio-WM", re.I),
    "development stage name": re.compile(r"\bPO\d+\b|\bproject3\b", re.I),
}


def main() -> int:
    errors: list[str] = []

    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required release file: {relative}")

    for forbidden_directory in ("paper", "results", "docs/development"):
        if (ROOT / forbidden_directory).exists():
            errors.append(f"private publication material present: {forbidden_directory}")

    for relative in PUBLIC_TEXT:
        path = ROOT / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for label, pattern in FORBIDDEN_PUBLIC_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label} in public file: {relative}")

    for path in ROOT.rglob("*"):
        if path.is_file() and ".git" not in path.parts and path.stat().st_size > 95 * 1024 * 1024:
            errors.append(f"file exceeds GitHub source limit: {path.relative_to(ROOT)}")

    manifest_path = ROOT / "checkpoints" / "MANIFEST.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest.get("files", []):
            relative = entry.get("path")
            expected = entry.get("sha256")
            checkpoint = ROOT / relative if isinstance(relative, str) else None
            if checkpoint is None or not checkpoint.is_file():
                errors.append(f"missing checkpoint from manifest: {relative}")
                continue
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if digest != expected:
                errors.append(f"checkpoint hash mismatch: {relative}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("SleepWM source release integrity: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
