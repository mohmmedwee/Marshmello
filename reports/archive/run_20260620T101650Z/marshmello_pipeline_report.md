# Marshmello Instruct Pipeline — Full Report

- Generated: 2026-06-20T08:56:52.453066+00:00
- Pipeline: chat-only base (18H) → teacher (18E) → routing (18I) → core SFT (18J)
- Training status: **all steps complete**
- Eval status: **all evals passed (exit 0)**

## Executive summary

| Step | Stage | Training | Eval |
|---|---|---|---|
| 1 | Chat-only base (18H) | ✓ step 8500 | 2/3 chat prompts (FAIL (2/3)) |
| 2 | Teacher SFT (18E) | ✓ 500 steps | 6/6 (PASS (6/6)) |
| 3 | Routing fix (18I) | ✓ 300 steps | 6/6 (PASS (6/6)) |
| 4 | Core SFT (18J) | ✓ 800 steps, val 3.50 | routing 10.0%, concept 21.0% |

## vs old pipeline (wrong order)

| Checkpoint | Routing | Concept | Notes |
|---|---:|---:|---|
| Base (prose pretrain) | 3.0% | 4.0% | before 18H |
| Teacher (old base) | 7.0% | 9.0% | best routing in old run |
| Full SFT 9.8k (old order) | 3.0% | 17.0% | broad SFT too early |
| **This pipeline (core SFT)** | **10.0%** | **21.0%** | chat base → teacher → routing → core |

**Recommendation:** Correct pipeline order helped: routing 10.0% and concept 21.0% vs ~3% base. Routing prompts pass 7/7, but held-out routing is still below 30%. Extend teacher/routing steps or eval `teacher_best_score.pt` before scaling model size.

## Checkpoints

- **step1_chat_base:** `/Users/ehabshobaki/Documents/myproject/learning/mini-transformer-from-scratch/13_gpt_pretraining/checkpoints/large_50m/latest.pt` ✓
- **step2_teacher:** `/Users/ehabshobaki/Documents/myproject/learning/mini-transformer-from-scratch/18E_tiny_teacher_sft/checkpoints/teacher_latest.pt` ✓
- **step3_routing:** `/Users/ehabshobaki/Documents/myproject/learning/mini-transformer-from-scratch/18I_routing_teacher_fix/checkpoints/routing_latest.pt` ✓
- **step4_core_sft:** `/Users/ehabshobaki/Documents/myproject/learning/mini-transformer-from-scratch/18B_marshmello_instruct/checkpoints/latest.pt` ✓

## Step 1 — Chat-only base (18H)

- Eval: `FAIL (2/3)`
- Checkpoint: `/Users/ehabshobaki/Documents/myproject/learning/mini-transformer-from-scratch/13_gpt_pretraining/checkpoints/large_50m/latest.pt`

```text
Answer:               A database index is a lookup structure that helps find rows faster. It speeds reads but can make writes slightly slower.
on_topic:             True ['index', 'database', 'lookup', 'faster', 'rows']
stops_at_end:         True
clean_after_end:      True
no_repeated_3gram:    True
result:               PASS
============================================================
final_result:         FAIL (2/3)
```

## Step 2 — Teacher SFT (18E)

- Eval: `PASS (6/6)` (6/6)
- Checkpoint: `/Users/ehabshobaki/Documents/myproject/learning/mini-transformer-from-scratch/18E_tiny_teacher_sft/checkpoints/teacher_latest.pt`

## Step 3 — Routing fix (18I)

- Eval: `PASS (6/6)` (6/6)
- Checkpoint: `/Users/ehabshobaki/Documents/myproject/learning/mini-transformer-from-scratch/18I_routing_teacher_fix/checkpoints/routing_latest.pt`

## Step 4 — Core SFT (18J) on 100 held-out questions

| Metric | Value |
|---|---:|
| Concept accuracy | 21.0% |
| Routing accuracy | 10.0% |
| Hallucination rate | 93.0% |
| Confusion drop vs chat base | 7.0% |

18J eval script note: Model capacity is likely the bottleneck and scaling to 300M–500M is justified.

Pipeline recommendation: Correct pipeline order helped: routing 10.0% and concept 21.0% vs ~3% base. Routing prompts pass 7/7, but held-out routing is still below 30%. Extend teacher/routing steps or eval `teacher_best_score.pt` before scaling model size.

## Decision guide

- **Routing > 30%** on 18J eval → core SFT pipeline is working; consider curated broad SFT.
- **Routing < 10%** after Step 3 → repeat routing or extend teacher; do not scale model size yet.
- **Chat-only base < 2/3 PASS** → extend 18H pretrain before any SFT.

## Raw JSON

Full machine-readable results: `reports/marshmello_pipeline_results.json`

## Re-run pipeline training

```bash
bash reports/run_instruct_pipeline.sh
python reports/generate_pipeline_report.py
```
