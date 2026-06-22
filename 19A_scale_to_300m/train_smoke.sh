#!/usr/bin/env bash
# Phase 19A: Marshmello-300M smoke test.
# Runs ONLY 20 optimizer steps to prove the large_300m config trains end-to-end
# (forward + backward + checkpoint) on this machine. This is NOT real training.
#
# Expect this to be slow and memory-hungry compared to large_50m — see README.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

echo "=== Phase 19A: Marshmello-300M smoke train (20 steps) ==="
echo "Project root: $ROOT"
echo

python 13_gpt_pretraining/training/trainer.py \
  --config large_300m \
  --corpus 13_gpt_pretraining/data/corpus_chat_only.txt \
  --steps 20 \
  --lr 1e-4

echo
echo "Smoke run complete. Checkpoints (if written): 13_gpt_pretraining/checkpoints/large_300m/"
echo "Benchmark the config:  python 13_gpt_pretraining/benchmark.py --config large_300m"
