"""Run a selected SleepWM training stage from the consolidated configuration."""
import argparse
import copy
import importlib
import os
from pathlib import Path
import sys
import tempfile

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", help="Stage name from --list")
    parser.add_argument("--list", action="store_true", help="List stages and required earlier checkpoints")
    parser.add_argument("--stage-help", action="store_true", help="Show the selected trainer's options")
    parser.add_argument("--stages-file", default="configs/training.yaml")
    parser.add_argument("--config", help="Use an edited, exported YAML instead of the stored stage config")
    parser.add_argument("--export-config", help="Write this stage's resolved config and exit")
    parser.add_argument("--device", help="Override train.device, e.g. cpu or cuda:0")
    args, forwarded = parser.parse_known_args()
    stages = yaml.safe_load(Path(args.stages_file).read_text(encoding="utf-8"))["stages"]
    if args.list:
        for name, stage in stages.items():
            print(f"{name}: requires {', '.join(stage['requires']) or '(none)'}")
        return
    if args.stage not in stages:
        parser.error("select a valid --stage; use --list to see the names")
    stage = stages[args.stage]
    config = copy.deepcopy(stage["config"])
    if args.config:
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.device:
        config["train"]["device"] = args.device
    if args.export_config:
        path = Path(args.export_config)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(path)
        return
    if not args.stage_help:
        device = str(config["train"].get("device", "cpu"))
        if device.startswith("cuda") and "CUDA_VISIBLE_DEVICES" not in os.environ:
            parser.error("set CUDA_VISIBLE_DEVICES explicitly for a GPU run")
    module = importlib.import_module(stage["module"])
    old_argv = sys.argv[:]
    try:
        if args.stage_help:
            sys.argv = [f"train.py --stage {args.stage}", "--help"]
            module.main()
            return
        # Temporary files are closed before the trainer opens them (Windows compatible).
        with tempfile.TemporaryDirectory(prefix="sleepwm-config-") as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            if stage.get("interface") == "checkpoints":
                defaults = []
                for key, value in config["checkpoints"].items():
                    defaults.extend(["--" + key.replace("_", "-") + "-checkpoint", str(value)])
                defaults.extend(["--seed", str(config["experiment"]["seed"]),
                                 "--device", str(config["train"]["device"]),
                                 "--epochs", str(config["train"]["epochs"]),
                                 "--output-dir", str(config["train"]["output_dir"])])
            else:
                defaults = ["--config", str(config_path)]
            sys.argv = [f"train.py --stage {args.stage}", *defaults, *forwarded]
            module.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
