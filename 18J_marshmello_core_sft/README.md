# Marshmello Core SFT Experiment

This experiment isolates data quality and routing from model capacity. It keeps the
existing `large_50m` model, architecture, and tokenizer unchanged and fine-tunes only
on three focused domains.

## Files

- `18J_marshmello_core_sft/data/marshmello_core_sft.jsonl` — 1,200 SFT examples, exactly 400 per domain.
- `18J_marshmello_core_sft/data/marshmello_core_negatives.jsonl` — 396 routing hard-negative records.
- `18J_marshmello_core_sft/data/marshmello_core_eval.jsonl` — 100 held-out questions: 40 AI, 30 transformers,
  and 30 databases.
- `18J_marshmello_core_sft/build_marshmello_core_data.py` — deterministic dataset builder and validator.
- `18J_marshmello_core_sft/evaluate_core_routing.py` — checkpoint evaluator and report generator.
- `18J_marshmello_core_sft/data/reports/marshmello_core_dataset_report.md` — generated dataset audit.
- `18J_marshmello_core_sft/data/reports/marshmello_core_experiment_plan.md` — experiment controls and run log.

The SFT file contains only:

- `ai_basics`
- `transformers_llms`
- `databases`

Every concept has all six question types and all three routing variants: normal,
paraphrased, and contrast. The final 5% of the SFT JSONL is deliberately balanced
at 20 examples per domain because the current trainer uses the file tail as its
validation split.

## 1. Rebuild and validate data

Run from the repository root:

```bash
python3 18J_marshmello_core_sft/build_marshmello_core_data.py
python3 18J_marshmello_core_sft/evaluate_core_routing.py --validate-only
```

The builder fails if counts, question balance, routing coverage, hard-negative
coverage, or the held-out split are invalid.

## 2. Train

Use only this training command:

```bash
python 18B_marshmello_instruct/train_instruct.py \
  --mode train \
  --config large_50m \
  --base-checkpoint 13_gpt_pretraining/checkpoints/large_50m/latest.pt \
  --data 18J_marshmello_core_sft/data/marshmello_core_sft.jsonl \
  --steps 800 \
  --lr 1e-6
```

The trainer writes `18B_marshmello_instruct/checkpoints/latest.pt`. Preserve any
existing checkpoint at that path before the run if it is needed elsewhere.

Do not add the hard-negative JSONL to the training command. Contrast examples in
the SFT file provide the trainable routing signal; the separate negative file is
for explicit audit and evaluation metadata.

## 3. Evaluate

```bash
python3 18J_marshmello_core_sft/evaluate_core_routing.py
```

Defaults:

- fine-tuned checkpoint: `18B_marshmello_instruct/checkpoints/latest.pt`
- baseline checkpoint: `13_gpt_pretraining/checkpoints/large_50m/latest.pt`
- deterministic greedy decoding
- full 100-question held-out set

Outputs:

- `18J_marshmello_core_sft/data/reports/marshmello_core_eval_results.json`
- `18J_marshmello_core_sft/data/reports/marshmello_core_eval_report.md`
- `18J_marshmello_core_sft/data/reports/marshmello_core_predictions.jsonl`
- `18J_marshmello_core_sft/data/reports/marshmello_core_baseline_predictions.jsonl`

The JSON report contains the complete concept and domain confusion matrices plus
per-example scores. The Markdown report contains aggregate metrics, domain
breakdowns, the largest confusions, and the experiment conclusion.

For a quick evaluator smoke test:

```bash
python3 18J_marshmello_core_sft/evaluate_core_routing.py --limit 3 --no-baseline
```

## Metrics

- **Concept accuracy:** answer contains at least one expected concept signature.
- **Routing accuracy:** expected concept has the highest weighted signature score.
- **Exact answer overlap:** normalized generated answer exactly equals the reference.
- **Keyword overlap:** average recall of expected concept keywords.
- **Hallucination rate:** deterministic proxy for empty, repetitive, ungrounded, or
  hard-negative-routed answers.
- **Confusion matrix:** expected concept versus predicted concept.

Contrast questions may legitimately mention another concept. The evaluator discounts
the declared contrast concept when selecting the routing target.

## Decision rule

The default definition of a significant confusion reduction is an absolute drop of
at least 10 percentage points from the base checkpoint.

- If routing accuracy is greater than 80% and concept confusion drops significantly:
  **Data/training was the primary bottleneck.**
- If routing accuracy remains at or below 80%:
  **Model capacity is likely the bottleneck and scaling to 300M–500M is justified.**
- If routing clears 80% without a significant baseline-relative confusion drop, the
  result is inconclusive and should not be used to justify scaling by itself.
