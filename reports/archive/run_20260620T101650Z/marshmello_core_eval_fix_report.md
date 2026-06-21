# Marshmello Core Routing Evaluation

- Generated: 2026-06-20T09:55:30.535393+00:00
- Fine-tuned checkpoint: `18B_marshmello_instruct/checkpoints/latest.pt`
- Baseline checkpoint: `13_gpt_pretraining/checkpoints/large_50m/latest.pt`
- Decoding: greedy

## Fine-tuned metrics

| Metric | Value |
|---|---:|
| Concept accuracy | 13.0% |
| Routing accuracy | 8.0% |
| Concept confusion rate | 92.0% |
| Exact answer overlap | 0.0% |
| Keyword overlap | 5.5% |
| Reference token overlap | 18.5% |
| Hallucination rate | 97.0% |

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

Absolute concept-confusion drop: 5.0%.

## Decision

Model capacity is likely the bottleneck and scaling to 300M–500M is justified.

The configured success rule requires routing accuracy above 80.0% and an absolute confusion-rate drop of at least 10.0%.

## Domain breakdown

| Domain | Count | Concept accuracy | Routing accuracy | Hallucination rate |
|---|---:|---:|---:|---:|
| ai_basics | 40 | 20.0% | 17.5% | 92.5% |
| databases | 30 | 16.7% | 3.3% | 100.0% |
| transformers_llms | 30 | 0.0% | 0.0% | 100.0% |

## Largest concept confusions

| Expected | Predicted | Count |
|---|---|---:|
| classification | database | 2 |
| database_table | database_index | 2 |
| loss_function | database_index | 2 |
| model_evaluation | database_index | 2 |
| sql | database_index | 2 |
| acid | database | 1 |
| artificial_intelligence | neural_network | 1 |
| attention_head | database | 1 |
| b_tree_index | database_index | 1 |
| bias_variance_tradeoff | database_constraint | 1 |
| causal_mask | database_index | 1 |
| context_window | training_data | 1 |
| database | database_index | 1 |
| database_column | database | 1 |
| database_column | database_index | 1 |

## Metric definitions

- Concept accuracy: the answer contains at least one expected concept signature.
- Routing accuracy: the expected concept has the highest weighted signature score.
- Exact answer overlap: normalized generated text exactly matches the reference answer.
- Keyword overlap: mean recall of expected concept keywords.
- Hallucination rate: deterministic proxy for empty/repetitive, ungrounded, or hard-negative-routed answers.
- The complete concept and domain confusion matrices are stored in the JSON results file.
