# Marshmello Core Routing Evaluation

- Generated: 2026-06-20T08:14:06.390183+00:00
- Fine-tuned checkpoint: `18E_tiny_teacher_sft/checkpoints/teacher_latest.pt`
- Baseline: not evaluated
- Decoding: greedy

## Fine-tuned metrics

| Metric | Value |
|---|---:|
| Concept accuracy | 9.0% |
| Routing accuracy | 7.0% |
| Concept confusion rate | 93.0% |
| Exact answer overlap | 0.0% |
| Keyword overlap | 3.2% |
| Reference token overlap | 17.0% |
| Hallucination rate | 93.0% |

## Baseline metrics

| Metric | Value |
|---|---:|
| Concept accuracy | 4.0% |
| Routing accuracy | 3.0% |
| Concept confusion rate | 97.0% |
| Exact answer overlap | 0.0% |
| Keyword overlap | 2.0% |
| Reference token overlap | 13.5% |
| Hallucination rate | 97.0% |

Absolute concept-confusion drop: 4.0%.

## Decision

Model capacity is likely the bottleneck and scaling to 300M–500M is justified.

The configured success rule requires routing accuracy above 80.0% and an absolute confusion-rate drop of at least 10.0%.

## Domain breakdown

| Domain | Count | Concept accuracy | Routing accuracy | Hallucination rate |
|---|---:|---:|---:|---:|
| ai_basics | 40 | 15.0% | 10.0% | 90.0% |
| databases | 30 | 6.7% | 6.7% | 93.3% |
| transformers_llms | 30 | 3.3% | 3.3% | 96.7% |

## Largest concept confusions

| Expected | Predicted | Count |
|---|---|---:|
| database_column | database | 2 |
| database_table | training_data | 2 |
| inference | machine_learning | 2 |
| model_evaluation | machine_learning | 2 |
| overfitting | machine_learning | 2 |
| primary_key | __unknown__ | 2 |
| training_data | __unknown__ | 2 |
| acid | database_constraint | 1 |
| attention_head | database_constraint | 1 |
| b_tree_index | database | 1 |
| bias_variance_tradeoff | training_data | 1 |
| causal_mask | token | 1 |
| classification | __unknown__ | 1 |
| classification | database | 1 |
| context_window | training_data | 1 |

## Metric definitions

- Concept accuracy: the answer contains at least one expected concept signature.
- Routing accuracy: the expected concept has the highest weighted signature score.
- Exact answer overlap: normalized generated text exactly matches the reference answer.
- Keyword overlap: mean recall of expected concept keywords.
- Hallucination rate: deterministic proxy for empty/repetitive, ungrounded, or hard-negative-routed answers.
- The complete concept and domain confusion matrices are stored in the JSON results file.
