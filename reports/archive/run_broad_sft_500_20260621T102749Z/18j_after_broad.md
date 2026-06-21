# Marshmello Core Routing Evaluation

- Generated: 2026-06-21T10:33:20.698541+00:00
- Fine-tuned checkpoint: `18B_marshmello_instruct/checkpoints/latest.pt`
- Baseline: not evaluated
- Decoding: greedy

## Fine-tuned metrics

| Metric | Value |
|---|---:|
| Concept accuracy | 25.0% |
| Routing accuracy | 17.0% |
| Concept confusion rate | 83.0% |
| Exact answer overlap | 0.0% |
| Keyword overlap | 8.5% |
| Reference token overlap | 19.4% |
| Hallucination rate | 91.0% |

## Decision

Model capacity is likely the bottleneck and scaling to 300M–500M is justified.

The configured success rule requires routing accuracy above 80.0% and an absolute confusion-rate drop of at least 10.0%.

## Domain breakdown

| Domain | Count | Concept accuracy | Routing accuracy | Hallucination rate |
|---|---:|---:|---:|---:|
| ai_basics | 40 | 32.5% | 30.0% | 85.0% |
| databases | 30 | 30.0% | 13.3% | 90.0% |
| transformers_llms | 30 | 10.0% | 3.3% | 100.0% |

## Largest concept confusions

| Expected | Predicted | Count |
|---|---|---:|
| database_table | database | 2 |
| embedding | token | 2 |
| foreign_key | database | 2 |
| supervised_learning | underfitting | 2 |
| transformer | database | 2 |
| acid | __unknown__ | 1 |
| attention_head | loss_function | 1 |
| bias_variance_tradeoff | primary_key | 1 |
| causal_mask | database_index | 1 |
| classification | database | 1 |
| classification | database_index | 1 |
| context_window | database | 1 |
| database_column | database | 1 |
| database_column | database_transaction | 1 |
| database_constraint | database | 1 |

## Metric definitions

- Concept accuracy: the answer contains at least one expected concept signature.
- Routing accuracy: the expected concept has the highest weighted signature score.
- Exact answer overlap: normalized generated text exactly matches the reference answer.
- Keyword overlap: mean recall of expected concept keywords.
- Hallucination rate: deterministic proxy for empty/repetitive, ungrounded, or hard-negative-routed answers.
- The complete concept and domain confusion matrices are stored in the JSON results file.
