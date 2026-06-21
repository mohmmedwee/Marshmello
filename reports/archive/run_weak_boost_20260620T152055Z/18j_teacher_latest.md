# Marshmello Core Routing Evaluation

- Generated: 2026-06-20T15:24:46.748721+00:00
- Fine-tuned checkpoint: `18E_tiny_teacher_sft/checkpoints/teacher_latest.pt`
- Baseline: not evaluated
- Decoding: greedy

## Fine-tuned metrics

| Metric | Value |
|---|---:|
| Concept accuracy | 31.0% |
| Routing accuracy | 17.0% |
| Concept confusion rate | 83.0% |
| Exact answer overlap | 0.0% |
| Keyword overlap | 12.0% |
| Reference token overlap | 22.9% |
| Hallucination rate | 79.0% |

## Decision

Model capacity is likely the bottleneck and scaling to 300M–500M is justified.

The configured success rule requires routing accuracy above 80.0% and an absolute confusion-rate drop of at least 10.0%.

## Domain breakdown

| Domain | Count | Concept accuracy | Routing accuracy | Hallucination rate |
|---|---:|---:|---:|---:|
| ai_basics | 40 | 30.0% | 25.0% | 77.5% |
| databases | 30 | 36.7% | 16.7% | 80.0% |
| transformers_llms | 30 | 26.7% | 6.7% | 80.0% |

## Largest concept confusions

| Expected | Predicted | Count |
|---|---|---:|
| database_column | database_index | 2 |
| loss_function | database_index | 2 |
| token | transformer | 2 |
| unsupervised_learning | supervised_learning | 2 |
| attention_head | database_index | 1 |
| b_tree_index | database_index | 1 |
| bias_variance_tradeoff | transformer | 1 |
| causal_mask | database_index | 1 |
| classification | database | 1 |
| classification | database_index | 1 |
| context_window | database_index | 1 |
| database | database_index | 1 |
| database_constraint | database_index | 1 |
| database_index | b_tree_index | 1 |
| database_join | database_index | 1 |

## Metric definitions

- Concept accuracy: the answer contains at least one expected concept signature.
- Routing accuracy: the expected concept has the highest weighted signature score.
- Exact answer overlap: normalized generated text exactly matches the reference answer.
- Keyword overlap: mean recall of expected concept keywords.
- Hallucination rate: deterministic proxy for empty/repetitive, ungrounded, or hard-negative-routed answers.
- The complete concept and domain confusion matrices are stored in the JSON results file.
