# Marshmello Core SFT Experiment Plan

## Hypothesis

Marshmello's current failures are caused primarily by either:

1. weak or ambiguous SFT data and question-to-concept routing, or
2. insufficient capacity in the existing 55M-parameter `large_50m` model.

## Fixed controls

- Model configuration: `large_50m`
- Approximate parameter count: 55M
- Base checkpoint: `13_gpt_pretraining/checkpoints/large_50m/latest.pt`
- Architecture changes: none
- Tokenizer changes: none
- Training steps: 800
- Learning rate: `1e-6`
- Training data: only `18J_marshmello_core_sft/data/marshmello_core_sft.jsonl`
- Evaluation data: `18J_marshmello_core_sft/data/marshmello_core_eval.jsonl`
- Decoding: greedy

## Data controls

- 1,200 SFT examples
- 400 examples in each of `ai_basics`, `transformers_llms`, and `databases`
- 66 concepts total, 22 per domain
- Six balanced question types
- Normal, paraphrased, and contrast routing coverage for every concept
- Three explicit incorrect concepts per hard-negative record
- 100 unique held-out evaluation questions with zero normalized question overlap

## Required training command

```bash
python 18B_marshmello_instruct/train_instruct.py \
  --mode train \
  --config large_50m \
  --base-checkpoint 13_gpt_pretraining/checkpoints/large_50m/latest.pt \
  --data 18J_marshmello_core_sft/data/marshmello_core_sft.jsonl \
  --steps 800 \
  --lr 1e-6
```

## Run log

| Field | Value |
|---|---|
| Run date | |
| Git commit | |
| Device | |
| Base checkpoint SHA-256 | |
| Fine-tuned checkpoint SHA-256 | |
| Final train loss | |
| Final validation loss | |
| Runtime | |

## Evaluation result

| Metric | Baseline | Fine-tuned | Delta |
|---|---:|---:|---:|
| Concept accuracy | | | |
| Routing accuracy | | | |
| Concept confusion rate | | | |
| Exact answer overlap | | | |
| Keyword overlap | | | |
| Hallucination rate | | | |

## Decision

- Routing above 80% and confusion reduced by at least 10 absolute percentage points:
  data/training was the primary bottleneck.
- Routing at or below 80%: model capacity is likely the bottleneck and scaling to
  300M–500M is justified.
- Otherwise: inconclusive; do not attribute the result to capacity without another
  controlled run.
