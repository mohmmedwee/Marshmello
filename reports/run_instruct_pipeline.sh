#!/usr/bin/env bash
# Marshmello instruct pipeline: chat base → teacher → routing → core SFT
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

CHAT_BASE="13_gpt_pretraining/checkpoints/large_50m/latest.pt"
TEACHER_CKPT="18E_tiny_teacher_sft/checkpoints/teacher_latest.pt"
ROUTING_CKPT="18I_routing_teacher_fix/checkpoints/routing_latest.pt"
CORE_CKPT="18B_marshmello_instruct/checkpoints/latest.pt"
LOG="reports/pipeline_run.log"

mkdir -p reports

echo "=== Marshmello instruct pipeline ===" | tee "$LOG"
echo "Started: $(date -u)" | tee -a "$LOG"

if [[ ! -f "$CHAT_BASE" ]]; then
  echo "Missing chat-adapted base: $CHAT_BASE" | tee -a "$LOG"
  echo "Run Phase 18H chat-only pretrain first." | tee -a "$LOG"
  exit 1
fi

echo "" | tee -a "$LOG"
echo "--- Step 2: Teacher SFT (18E) ---" | tee -a "$LOG"
PYTHONUNBUFFERED=1 python 18B_marshmello_instruct/train_instruct.py \
  --mode teacher \
  --config large_50m \
  --base-checkpoint "$CHAT_BASE" \
  --steps 500 \
  --lr 5e-6 \
  --eval-generation-only-warn 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "--- Step 3: Routing fix (18I) ---" | tee -a "$LOG"
PYTHONUNBUFFERED=1 python 18B_marshmello_instruct/train_instruct.py \
  --mode routing \
  --config large_50m \
  --base-checkpoint "$TEACHER_CKPT" \
  --steps 300 \
  --eval-generation-only-warn 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "--- Step 4: Core SFT (18J) ---" | tee -a "$LOG"
PYTHONUNBUFFERED=1 python 18B_marshmello_instruct/train_instruct.py \
  --mode train \
  --config large_50m \
  --base-checkpoint "$ROUTING_CKPT" \
  --data 18J_marshmello_core_sft/data/marshmello_core_sft.jsonl \
  --steps 800 \
  --lr 1e-6 \
  --eval-generation-only-warn 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Finished training: $(date -u)" | tee -a "$LOG"
echo "Checkpoints:" | tee -a "$LOG"
echo "  Chat base:  $CHAT_BASE" | tee -a "$LOG"
echo "  Teacher:    $TEACHER_CKPT" | tee -a "$LOG"
echo "  Routing:    $ROUTING_CKPT" | tee -a "$LOG"
echo "  Core SFT:   $CORE_CKPT" | tee -a "$LOG"
