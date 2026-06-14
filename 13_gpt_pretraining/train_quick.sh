#!/usr/bin/env bash
# Phase 13: prepare tech corpus → train BPE → quick GPT pretrain (300 steps)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

echo "=== Phase 13 quick train (tech corpus) ==="
echo "Project root: $ROOT"
echo

python 13_gpt_pretraining/data/prepare_corpus.py --mix tech --repeat-per-domain 20 --force
python 13_gpt_pretraining/tokenizer/train_bpe.py
python 13_gpt_pretraining/training/trainer.py --quick

echo
echo "Done. Generate text:"
echo '  python 13_gpt_pretraining/generate.py --prompt "Database" --domain-hint databases'
