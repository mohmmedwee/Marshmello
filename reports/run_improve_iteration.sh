#!/usr/bin/env bash
# Iteration: teacher 800 steps + 18J eval on all teacher checkpoints; optional core SFT.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

RUN_ID="${1:-run_$(date -u +%Y%m%dT%H%M%SZ)}"
ARCHIVE="$ROOT/reports/archive/$RUN_ID"
mkdir -p "$ARCHIVE"
LOG="$ROOT/reports/improve_loop.log"
BASELINE_ROUTING="${BASELINE_ROUTING:-0.10}"

echo "" | tee -a "$LOG"
echo "=== Iteration $RUN_ID: chat eval + teacher 800 ===" | tee -a "$LOG"

python 18H_chat_only_pretraining/eval_chat_only.py --no-strict 2>&1 | tee "$ARCHIVE/chat_only_eval.txt" | tee -a "$LOG"

PYTHONUNBUFFERED=1 python 18B_marshmello_instruct/train_instruct.py \
  --mode teacher \
  --config large_50m \
  --base-checkpoint 13_gpt_pretraining/checkpoints/large_50m/latest.pt \
  --steps 800 \
  --lr 5e-6 \
  --eval-generation-only-warn 2>&1 | tee "$ARCHIVE/teacher_train.log" | tee -a "$LOG"

BEST_ROUTING=0
BEST_CKPT=""
BEST_REPORT=""

for CKPT in \
  18E_tiny_teacher_sft/checkpoints/teacher_best_score.pt \
  18E_tiny_teacher_sft/checkpoints/teacher_best_val.pt \
  18E_tiny_teacher_sft/checkpoints/teacher_latest.pt; do
  NAME=$(basename "$CKPT" .pt)
  JSON="$ARCHIVE/18j_${NAME}.json"
  MD="$ARCHIVE/18j_${NAME}.md"
  echo "Eval 18J: $CKPT" | tee -a "$LOG"
  python 18J_marshmello_core_sft/evaluate_core_routing.py \
    --checkpoint "$CKPT" \
    --no-baseline \
    --results-json "$JSON" \
    --report "$MD" 2>&1 | tee -a "$LOG" | tail -6
  ROUTING=$(python -c "import json; d=json.load(open('$JSON')); print(d['trained']['routing_accuracy'])")
  if python -c "import sys; sys.exit(0 if float('$ROUTING') > float('$BEST_ROUTING') else 1)"; then
    BEST_ROUTING="$ROUTING"
    BEST_CKPT="$CKPT"
    BEST_REPORT="$MD"
  fi
done

echo "Best teacher checkpoint: $BEST_CKPT routing=$BEST_ROUTING (baseline=$BASELINE_ROUTING)" | tee -a "$LOG"

if python -c "import sys; sys.exit(0 if float('$BEST_ROUTING') > float('$BASELINE_ROUTING') else 1)"; then
  cp -f "$BEST_CKPT" 18B_marshmello_instruct/checkpoints/best_18j_routing.pt
  echo "Updated best_18j_routing.pt from $BEST_CKPT" | tee -a "$LOG"
  if python -c "import sys; sys.exit(0 if float('$BEST_ROUTING') >= 0.12 else 1)"; then
    echo "=== Short core SFT from best teacher ===" | tee -a "$LOG"
    PYTHONUNBUFFERED=1 python 18B_marshmello_instruct/train_instruct.py \
      --mode train \
      --config large_50m \
      --base-checkpoint "$BEST_CKPT" \
      --data 18J_marshmello_core_sft/data/marshmello_core_sft.jsonl \
      --steps 200 \
      --lr 5e-7 \
      --first-token-weight 30 \
      --eval-generation-only-warn 2>&1 | tee "$ARCHIVE/core_sft_200.log" | tee -a "$LOG"
    python 18J_marshmello_core_sft/evaluate_core_routing.py \
      --checkpoint 18B_marshmello_instruct/checkpoints/latest.pt \
      --no-baseline \
      --results-json "$ARCHIVE/18j_core_after_teacher.json" \
      --report "$ARCHIVE/18j_core_after_teacher.md" 2>&1 | tee -a "$LOG" | tail -6
    CORE_R=$(python -c "import json; d=json.load(open('$ARCHIVE/18j_core_after_teacher.json')); print(d['trained']['routing_accuracy'])")
    if python -c "import sys; sys.exit(0 if float('$CORE_R') > float('$BEST_ROUTING') else 1)"; then
      cp -f 18B_marshmello_instruct/checkpoints/latest.pt 18B_marshmello_instruct/checkpoints/best_18j_routing.pt
      BEST_ROUTING="$CORE_R"
      BEST_CKPT="18B_marshmello_instruct/checkpoints/latest.pt"
      BEST_REPORT="$ARCHIVE/18j_core_after_teacher.md"
    fi
  fi
fi

python reports/archive_reports.py --label "iteration_$RUN_ID" --run-id "$RUN_ID"

python - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone

root = Path("$ROOT")
archive = root / "reports/archive/$RUN_ID"
rows = []
best = {"routing_accuracy": 0, "checkpoint": "none", "archive": "$RUN_ID"}
for p in sorted(archive.glob("18j_*.json")):
    d = json.loads(p.read_text())
    t = d["trained"]
    row = {
        "label": p.stem,
        "checkpoint": d.get("checkpoint", "?"),
        "routing_accuracy": t["routing_accuracy"],
        "concept_accuracy": t["concept_accuracy"],
        "hallucination_rate": t["hallucination_rate"],
    }
    rows.append(row)
    if t["routing_accuracy"] > best["routing_accuracy"]:
        best = {**row, "archive": "$RUN_ID"}

summary = root / "reports/latest_eval_summary.md"
lines = [
    "# Marshmello — Latest Eval Summary",
    "",
    f"- Updated: {datetime.now(timezone.utc).isoformat()}",
    f"- Baseline beaten: {float('$BASELINE_ROUTING')*100:.1f}% → **{best['routing_accuracy']*100:.1f}%**",
    f"- Best checkpoint: \`{best.get('checkpoint', '?')}\`",
    "",
    "## This iteration ($RUN_ID)",
    "",
    "| Eval | Routing | Concept | Hallucination |",
    "|---|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r['label']} | {r['routing_accuracy']*100:.1f}% | "
        f"{r['concept_accuracy']*100:.1f}% | {r['hallucination_rate']*100:.1f}% |"
    )
lines += ["", f"Archive: \`reports/archive/$RUN_ID/\`"]
summary.write_text("\\n".join(lines) + "\\n")
print(f"Best routing: {best['routing_accuracy']*100:.1f}%")
PY

echo "Done iteration $RUN_ID" | tee -a "$LOG"
