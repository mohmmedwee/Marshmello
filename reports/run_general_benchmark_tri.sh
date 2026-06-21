#!/usr/bin/env bash
# Phase 18K: build eval set and compare the three standard checkpoints.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

python "$ROOT/18K_general_benchmark/build_general_benchmark.py"
python "$ROOT/18K_general_benchmark/compare_checkpoints.py"

echo "Reports: $ROOT/18K_general_benchmark/reports/"
