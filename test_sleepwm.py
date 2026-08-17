"""Public evaluation entrypoint for SleepWM."""

import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parent / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from evaluate_belief_outcomes import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
