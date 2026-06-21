# Marshmello Core Routing — Checkpoint Comparison

- Updated: 2026-06-20 (fix run)
- Eval set: `marshmello_core_eval.jsonl` (100 held-out questions)
- Decoding: greedy

## Summary table (18J held-out)

| Checkpoint | Routing | Concept | Hallucination | Notes |
|---|---:|---:|---:|---|
| Chat base | 3.0% | 3–4% | 99% | step 8500 |
| `routing_latest` (old, from teacher_latest) | 3.0% | 6.0% | 94% | 18I 6/6 but 18J fails |
| Core SFT `latest` (800 steps, from routing) | 6.0% | 17.0% | 98% | val loss ↓, routing mediocre |
| **`teacher_best_score.pt`** | **10.0%** | **13.0%** | **89%** | **best single checkpoint** |
| Routing v2 (from teacher_best_score) | 2.0% | 3.0% | 99% | routing stage **hurts** 18J |
| Core SFT fix (400 steps, lr 5e-7, wt 30, from teacher_best_score) | 8.0% | 13.0% | 97% | better than old core, worse than teacher alone |

## Conclusions from fix run

1. **Use `teacher_best_score.pt` for best 18J routing (10%)** — do not assume `latest.pt` is best.
2. **18I routing eval (6/6) does not predict 18J** — routing retrain from best teacher dropped 18J routing 10% → 2%.
3. **Core SFT from teacher_best_score** (400 steps, `--first-token-weight 30`, `--lr 5e-7`) reached **8%** routing — improvement over old 6% core run, but **still below teacher alone**.
4. **Next lever:** ship `teacher_best_score.pt` as instruct checkpoint; extend **teacher** training (not routing) on 18J-style paraphrases; optional very short core SFT (≤200 steps) with early stop on 18J.

## Checkpoint paths

| Label | Path |
|---|---|
| Best 18J routing | `18E_tiny_teacher_sft/checkpoints/teacher_best_score.pt` |
| Latest core SFT (fix run) | `18B_marshmello_instruct/checkpoints/latest.pt` |
| Routing v2 | `18I_routing_teacher_fix/checkpoints/routing_latest.pt` |
| Backups | `routing_latest_backup.pt`, `latest_pre_fix_backup.pt` |

## Per-checkpoint reports

| Checkpoint | Report |
|---|---|
| teacher_best_score | [marshmello_core_eval_teacher_best_score_report.md](marshmello_core_eval_teacher_best_score_report.md) |
| Core SFT fix | [marshmello_core_eval_fix_report.md](marshmello_core_eval_fix_report.md) |
| Routing v2 | [marshmello_core_eval_routing_v2_report.md](marshmello_core_eval_routing_v2_report.md) |
| Routing (old) | [marshmello_core_eval_report.md](marshmello_core_eval_report.md) |
