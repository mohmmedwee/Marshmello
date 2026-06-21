# Marshmello Core Routing Evaluation

- Generated: 2026-06-20T10:22:19.671758+00:00
- Fine-tuned checkpoint: `18B_marshmello_instruct/checkpoints/latest.pt`
- Baseline: not evaluated
- Decoding: greedy

## Fine-tuned metrics

| Metric | Value |
|---|---:|
| Concept accuracy | 11.0% |
| Routing accuracy | 6.0% |
| Concept confusion rate | 94.0% |
| Exact answer overlap | 0.0% |
| Keyword overlap | 4.5% |
| Reference token overlap | 17.6% |
| Hallucination rate | 96.0% |

## Decision

Model capacity is likely the bottleneck and scaling to 300M–500M is justified.

The configured success rule requires routing accuracy above 80.0% and an absolute confusion-rate drop of at least 10.0%.

## Domain breakdown

| Domain | Count | Concept accuracy | Routing accuracy | Hallucination rate |
|---|---:|---:|---:|---:|
| ai_basics | 40 | 15.0% | 12.5% | 92.5% |
| databases | 30 | 13.3% | 3.3% | 96.7% |
| transformers_llms | 30 | 3.3% | 0.0% | 100.0% |

## Largest concept confusions

| Expected | Predicted | Count |
|---|---|---:|
| database_column | database_index | 2 |
| loss_function | database_index | 2 |
| model_evaluation | database_index | 2 |
| relational_database | database_index | 2 |
| acid | database_index | 1 |
| artificial_intelligence | neural_network | 1 |
| attention_head | database_index | 1 |
| b_tree_index | database_index | 1 |
| bias_variance_tradeoff | __unknown__ | 1 |
| causal_mask | database_index | 1 |
| classification | __unknown__ | 1 |
| classification | database | 1 |
| context_window | __unknown__ | 1 |
| database | database_index | 1 |
| database_constraint | database_index | 1 |

## Metric definitions

- Concept accuracy: the answer contains at least one expected concept signature.
- Routing accuracy: the expected concept has the highest weighted signature score.
- Exact answer overlap: normalized generated text exactly matches the reference answer.
- Keyword overlap: mean recall of expected concept keywords.
- Hallucination rate: deterministic proxy for empty/repetitive, ungrounded, or hard-negative-routed answers.
- The complete concept and domain confusion matrices are stored in the JSON results file.
