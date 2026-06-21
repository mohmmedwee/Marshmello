# Marshmello Core Routing — Checkpoint Comparison

- Updated: 2026-06-20 (teacher_extended_short run)
- Eval: 100 held-out questions, greedy decode

## Summary

| Run | Routing | Concept | Best checkpoint |
|---|---:|---:|---|
| Baseline (250 ex teacher) | 10.0% | 13% | teacher_best_score |
| Extended multi-sentence | 13.0% | 28% | teacher_latest |
| **Short single-sentence** | **18.0%** | **33%** | **teacher_latest** |

## Current best (deploy)

```
18E_tiny_teacher_sft/checkpoints/teacher_latest.pt
18B_marshmello_instruct/checkpoints/best_18j_routing.pt
```

## Dataset pipeline

1. `build_teacher_extended.py` → `teacher_extended.jsonl` (1422)
2. `build_teacher_extended_short.py` → `teacher_extended_short.jsonl` (1206)
   - First sentence only, 12–35 words, cross-domain filter, no filler

## Archive

`reports/archive/run_teacher_short_20260620T120454Z/`
