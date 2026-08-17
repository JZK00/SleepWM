#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON=${PYTHON:-python3}

"$PYTHON" tools/make_toy_data.py --output-dir .toy_data
"$PYTHON" tools/validate_configs.py
"$PYTHON" tools/check_release_integrity.py
PYTHONPATH=src:. "$PYTHON" -m pytest -q \
  tests/test_public_api.py \
  tests/test_causality.py \
  tests/test_partial_observation.py \
  tests/test_recursive_belief_filter.py
