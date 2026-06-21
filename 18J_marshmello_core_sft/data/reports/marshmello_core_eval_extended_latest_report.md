# Marshmello Core Routing Evaluation

- Generated: 2026-06-20T10:58:40.188555+00:00
- Fine-tuned checkpoint: `18E_tiny_teacher_sft/checkpoints/teacher_latest.pt`
- Baseline: not evaluated
- Decoding: greedy

## Fine-tuned metrics

| Metric | Value |
|---|---:|
| Concept accuracy | 28.0% |
| Routing accuracy | 13.0% |
| Concept confusion rate | 87.0% |
| Exact answer overlap | 0.0% |
| Keyword overlap | 10.2% |
| Reference token overlap | 22.9% |
| Hallucination rate | 90.0% |

## Decision

Model capacity is likely the bottleneck and scaling to 300M–500M is justified.

The configured success rule requires routing accuracy above 80.0% and an absolute confusion-rate drop of at least 10.0%.

## Domain breakdown

| Domain | Count | Concept accuracy | Routing accuracy | Hallucination rate |
|---|---:|---:|---:|---:|
| ai_basics | 40 | 27.5% | 25.0% | 80.0% |
| databases | 30 | 43.3% | 3.3% | 96.7% |
| transformers_llms | 30 | 13.3% | 6.7% | 96.7% |

## Largest concept confusions

| Expected | Predicted | Count |
|---|---|---:|
| database | database_index | 2 |
| database_column | database_index | 2 |
| database_table | database_index | 2 |
| loss_function | database_index | 2 |
| relational_database | database_index | 2 |
| supervised_learning | underfitting | 2 |
| vocabulary | database_index | 2 |
| artificial_intelligence | precision | 1 |
| attention_head | database_index | 1 |
| b_tree_index | database_index | 1 |
| bias_variance_tradeoff | database_index | 1 |
| causal_mask | database_index | 1 |
| classification | database | 1 |
| classification | database_index | 1 |
| context_window | database_index | 1 |

## Metric definitions

- Concept accuracy: the answer contains at least one expected concept signature.
- Routing accuracy: the expected concept has the highest weighted signature score.
- Exact answer overlap: normalized generated text exactly matches the reference answer.
- Keyword overlap: mean recall of expected concept keywords.
- Hallucination rate: deterministic proxy for empty/repetitive, ungrounded, or hard-negative-routed answers.
- The complete concept and domain confusion matrices are stored in the JSON results file.
