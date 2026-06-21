# Phase 18E — Tiny Teacher SFT

Phase 18E builds a small, direct-answer **teacher** dataset before broad SFT or core
experiments. The goal is to teach Marshmello basic assistant behavior first:

- answer the user directly
- keep responses short (12–25 words in extended sets)
- stay on topic
- stop cleanly
- avoid repeated loops such as `computer is a computer`

## Dataset builders

| Script | Output | Notes |
|--------|--------|-------|
| `build_teacher_data.py` | `data/teacher.jsonl` | Original ~250 examples |
| `build_teacher_extended.py` | `data/teacher_extended.jsonl` | Longer multi-sentence teacher |
| `build_teacher_extended_short.py` | `data/teacher_extended_short.jsonl` | **Primary train set** — single-sentence answers |
| `build_teacher_weak_boost.py` | merged into extended short | +168 db/transformers rows (experiment) |
| `build_teacher_transformers_routing.py` | `data/teacher_transformers_routing.jsonl` | 40-row micro routing set |
| `build_teacher_math.py` | `data/teacher_math.jsonl` | **216 math rows** (add/sub/mul) |

### Build all extended short data (recommended)

```bash
python 18E_tiny_teacher_sft/build_teacher_extended_short.py
python 18E_tiny_teacher_sft/build_teacher_math.py   # merges into extended short
```

Current merged train file:

```text
18E_tiny_teacher_sft/data/teacher_extended_short.jsonl   # 1590 rows (incl. math_basics)
```

JSONL schema:

```json
{"instruction":"...","response":"...","domain":"...","concept":"..."}
```

Domain highlights in `teacher_extended_short.jsonl`:

- Core: `ai_basics`, `machine_learning`, `databases`, `software_engineering`, `python`, `transformers_llms`
- **`math_basics`**: 216 rows — e.g. `"can you do sum 12+14="` → `"The sum of 12 and 14 is 26, ..."`

Reports: `18E_tiny_teacher_sft/reports/teacher_*_report.json`

## Train

Retrain teacher from **chat base** (not from `best_18j_routing` unless experimenting):

```bash
python 18B_marshmello_instruct/train_instruct.py \
  --mode teacher \
  --config large_50m \
  --base-checkpoint 13_gpt_pretraining/checkpoints/large_50m/latest.pt \
  --data 18E_tiny_teacher_sft/data/teacher_extended_short.jsonl \
  --steps 800 \
  --lr 5e-6 \
  --eval-generation-only-warn
```

Default teacher checkpoint:

```text
18E_tiny_teacher_sft/checkpoints/teacher_latest.pt
```

## Evaluate

Teacher quality (shortness, keywords, repetition):

```bash
python 18E_tiny_teacher_sft/eval_teacher.py
```

Core **routing** (18J) — must not collapse vs deploy gate:

```bash
python 18J_marshmello_core_sft/evaluate_core_routing.py \
  --checkpoint 18E_tiny_teacher_sft/checkpoints/teacher_latest.pt --no-baseline
```

General assistant quality (18K):

```bash
python 18K_general_benchmark/evaluate_general.py \
  --label teacher_latest \
  --checkpoint 18E_tiny_teacher_sft/checkpoints/teacher_latest.pt
```

## Chat smoke test

```bash
python 18B_marshmello_instruct/chat.py \
  --checkpoint 18E_tiny_teacher_sft/checkpoints/teacher_latest.pt \
  --prompt "can you do sum 12+14=" \
  --greedy
```

## Notes

- **45M regresses** on large heterogeneous SFT (core 1200-row, broad 9300-row, micro-patches).
- Best known **18J routing** (~18%) came from teacher short data → `best_18j_routing.pt`.
- Math rows are in data only until you retrain teacher on `teacher_extended_short.jsonl`.
