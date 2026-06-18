# Phase 18E - Tiny Teacher SFT

Phase 18E builds a small, direct-answer teacher dataset before broad SFT.

The goal is to teach Marshmello the basic assistant behavior first:

- answer the user directly
- keep responses short
- stay on topic
- stop cleanly
- avoid repeated loops such as `computer is a computer`

## Build Data

```bash
python 18E_tiny_teacher_sft/build_teacher_data.py
```

Output:

```text
18E_tiny_teacher_sft/data/teacher.jsonl
```

The JSONL schema is:

```json
{"instruction":"...","response":"...","domain":"..."}
```

Domain counts:

- `ai_basics`: 60
- `machine_learning`: 40
- `databases`: 40
- `software_engineering`: 40
- `python`: 40
- `transformers_llms`: 30

## Train

```bash
python 18B_marshmello_instruct/train_instruct.py \
  --mode teacher \
  --config large_50m \
  --base-checkpoint 13_gpt_pretraining/checkpoints/large_50m/step_001000.pt \
  --steps 500 \
  --lr 5e-6
```

Default teacher checkpoint:

```text
18E_tiny_teacher_sft/checkpoints/teacher_latest.pt
```

## Evaluate

```bash
python 18E_tiny_teacher_sft/eval_teacher.py
```

The evaluator checks shortness, topic keywords, repeated 3-grams, and the
known `computer is a computer` collapse loop.
