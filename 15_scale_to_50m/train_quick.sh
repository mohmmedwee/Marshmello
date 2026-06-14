#!/usr/bin/env bash
# Phase 15 — smoke test: benchmark + short 50M training run
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

echo "=== 50M benchmark ==="
python 15_scale_to_50m/benchmark.py --quick

echo
echo "=== 50M training smoke test (300 steps) ==="
python 13_gpt_pretraining/training/trainer.py --config large_50m --steps 300 --quick
