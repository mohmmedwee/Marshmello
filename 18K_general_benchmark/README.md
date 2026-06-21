# Phase 18K — Marshmello General Benchmark

Phase **18K** adds a **second benchmark** that measures **general assistant quality**
separately from Phase **18J** core concept routing.

## Two benchmarks, two questions

| Phase | Question | Use when |
|---|---|---|
| **18J** | Does the model route to the correct **core concept** (AI, transformer, database index, …)? | Judging teacher SFT, routing fixes, core discrimination |
| **18K** | Does the model answer **general assistant** questions well (SQL, programming, daily life, …)? | Judging broad SFT, assistant usefulness, general QA |

**Do not use 18J alone to judge `marshmello_all_sft.jsonl` or broad SFT.**

## Files

| File | Purpose |
|---|---|
| `build_general_benchmark.py` | Build 500 held-out questions from `data/marshmello_all_sft.jsonl` |
| `data/general_eval.jsonl` | Stratified held-out eval (500) |
| `data/general_train.jsonl` | Non-overlapping train pool (~9300) for future broad SFT |
| `evaluate_general.py` | Evaluate one checkpoint |
| `compare_checkpoints.py` | Compare base, `best_18j_routing`, `teacher_latest` |
| `reports/` | Markdown + JSON reports |

## Eval buckets (100 each)

1. **AI / ML** — ai_basics, machine_learning, deep_learning, nlp, transformers_llms, …
2. **Databases / SQL** — databases, sql, data_structures
3. **Programming** — python, javascript, algorithms, web_development, …
4. **System Design / DevOps** — system_design, devops, linux, networking, …
5. **General knowledge** — daily_life, writing_communication, general_science, …

Each eval row includes:

- `instruction`
- `reference_response`
- `domain`, `concept`
- `expected_keywords`
- `benchmark_bucket`

Guarantees:

- no duplicate instructions in eval
- no instruction overlap between eval and train pool

## Metrics (18K)

- **Keyword recall** — expected keywords found in the answer
- **Reference token overlap** — token F1 vs reference answer
- **Response length sanity** — not empty, not extremely short/long vs reference
- **Repetition rate** — repeated 3-grams
- **Empty response rate**
- **Hallucination proxy** — empty/repetitive or long low-overlap answers
- **Domain score** — composite per example and per bucket
- **10 sample outputs** per checkpoint in reports

## Quick start

```bash
# Build held-out eval + train pool
python 18K_general_benchmark/build_general_benchmark.py

# Compare the three standard checkpoints
python 18K_general_benchmark/compare_checkpoints.py

# Or evaluate one checkpoint
python 18K_general_benchmark/evaluate_general.py \
  --label best_18j_routing \
  --checkpoint 18B_marshmello_instruct/checkpoints/best_18j_routing.pt
```

Reports are written to `18K_general_benchmark/reports/`.

## Standard checkpoints

1. `13_gpt_pretraining/checkpoints/large_50m/latest.pt` — chat base (**base**)
2. `18B_marshmello_instruct/checkpoints/best_18j_routing.pt` — best 18J routing deploy
3. `18E_tiny_teacher_sft/checkpoints/teacher_latest.pt` — latest teacher SFT

## Decision guide

### Broad SFT

Judge broad SFT with **18K**, not 18J.

Core SFT can hurt 18J routing even when answers sound better. Broad SFT should improve **18K domain score** without collapsing 18J below your deploy gate.

### 300M scaling

Consider 300M when:

- **18J** routing plateaus after better teacher data on 45M, **and**
- **18K** general quality is still poor after broad SFT on 45M

Or when both benchmarks stay weak after a full pipeline on 45M.

### Hybrid success example

```text
18J routing  = 18%   (core teacher works a little)
18K domain score = good (general assistant useful)
→ You built a helpful assistant; routing can improve separately.
```

```text
18J routing  = 18%
18K domain score = poor
→ General capacity or broad training is the bottleneck; consider 300M or broad SFT first.
```

## Relationship to 18J

- **18J is unchanged.** Phase 18K does not modify 18J eval, training, or checkpoints.
- **18K complements 18J** for the hybrid evaluation strategy (Path C).
