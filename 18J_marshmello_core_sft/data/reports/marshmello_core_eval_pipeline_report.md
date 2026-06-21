# Marshmello Core Routing Evaluation

- Generated: 2026-06-20T09:43:28.674284+00:00
- Fine-tuned checkpoint: `18B_marshmello_instruct/checkpoints/latest.pt`
- Baseline checkpoint: `13_gpt_pretraining/checkpoints/large_50m/latest.pt`
- Decoding: greedy

## Fine-tuned metrics

| Metric | Value |
|---|---:|
| Concept accuracy | 17.0% |
| Routing accuracy | 6.0% |
| Concept confusion rate | 94.0% |
| Exact answer overlap | 0.0% |
| Keyword overlap | 6.2% |
| Reference token overlap | 19.4% |
| Hallucination rate | 98.0% |

## Baseline metrics

| Metric | Value |
|---|---:|
| Concept accuracy | 3.0% |
| Routing accuracy | 3.0% |
| Concept confusion rate | 97.0% |
| Exact answer overlap | 0.0% |
| Keyword overlap | 1.8% |
| Reference token overlap | 14.0% |
| Hallucination rate | 99.0% |

Absolute concept-confusion drop: 3.0%.

## Decision

Model capacity is likely the bottleneck and scaling to 300M–500M is justified.

The configured success rule requires routing accuracy above 80.0% and an absolute confusion-rate drop of at least 10.0%.

## Domain breakdown

| Domain | Count | Concept accuracy | Routing accuracy | Hallucination rate |
|---|---:|---:|---:|---:|
| ai_basics | 40 | 15.0% | 10.0% | 95.0% |
| databases | 30 | 33.3% | 6.7% | 100.0% |
| transformers_llms | 30 | 3.3% | 0.0% | 100.0% |

## Largest concept confusions

| Expected | Predicted | Count |
|---|---|---:|
| database_column | database_index | 2 |
| database_row | database_index | 2 |
| database_table | database_index | 2 |
| gradient_descent | database_index | 2 |
| loss_function | database_index | 2 |
| model_evaluation | database_index | 2 |
| relational_database | database_index | 2 |
| self_attention | database_index | 2 |
| sql | database_index | 2 |
| underfitting | database | 2 |
| vocabulary | database_index | 2 |
| acid | database_index | 1 |
| artificial_intelligence | transformer | 1 |
| attention_head | database_index | 1 |
| b_tree_index | database_index | 1 |

## Metric definitions

- Concept accuracy: the answer contains at least one expected concept signature.
- Routing accuracy: the expected concept has the highest weighted signature score.
- Exact answer overlap: normalized generated text exactly matches the reference answer.
- Keyword overlap: mean recall of expected concept keywords.
- Hallucination rate: deterministic proxy for empty/repetitive, ungrounded, or hard-negative-routed answers.
- The complete concept and domain confusion matrices are stored in the JSON results file.
