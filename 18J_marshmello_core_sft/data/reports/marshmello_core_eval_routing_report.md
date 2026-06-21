# Marshmello Core Routing Evaluation

- Generated: 2026-06-20T08:13:16.724587+00:00
- Fine-tuned checkpoint: `18I_routing_teacher_fix/checkpoints/routing_latest.pt`
- Baseline: not evaluated
- Decoding: greedy

## Fine-tuned metrics

| Metric | Value |
|---|---:|
| Concept accuracy | 6.0% |
| Routing accuracy | 2.0% |
| Concept confusion rate | 98.0% |
| Exact answer overlap | 0.0% |
| Keyword overlap | 1.8% |
| Reference token overlap | 15.5% |
| Hallucination rate | 97.0% |

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

Absolute concept-confusion drop: -1.0%.

## Decision

Model capacity is likely the bottleneck and scaling to 300M–500M is justified.

The configured success rule requires routing accuracy above 80.0% and an absolute confusion-rate drop of at least 10.0%.

## Domain breakdown

| Domain | Count | Concept accuracy | Routing accuracy | Hallucination rate |
|---|---:|---:|---:|---:|
| ai_basics | 40 | 5.0% | 5.0% | 95.0% |
| databases | 30 | 13.3% | 0.0% | 96.7% |
| transformers_llms | 30 | 0.0% | 0.0% | 100.0% |

## Largest concept confusions

| Expected | Predicted | Count |
|---|---|---:|
| database | database_index | 2 |
| database_column | database_index | 2 |
| foreign_key | database_index | 2 |
| gradient_descent | database_index | 2 |
| machine_learning | __unknown__ | 2 |
| relational_database | database_index | 2 |
| tokenizer | database_index | 2 |
| underfitting | __unknown__ | 2 |
| vocabulary | database_index | 2 |
| acid | database_index | 1 |
| attention_head | database_index | 1 |
| b_tree_index | database_index | 1 |
| bias_variance_tradeoff | __unknown__ | 1 |
| causal_mask | database_index | 1 |
| classification | database | 1 |

## Metric definitions

- Concept accuracy: the answer contains at least one expected concept signature.
- Routing accuracy: the expected concept has the highest weighted signature score.
- Exact answer overlap: normalized generated text exactly matches the reference answer.
- Keyword overlap: mean recall of expected concept keywords.
- Hallucination rate: deterministic proxy for empty/repetitive, ungrounded, or hard-negative-routed answers.
- The complete concept and domain confusion matrices are stored in the JSON results file.
