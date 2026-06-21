# Marshmello Core Routing Evaluation

- Generated: 2026-06-21T09:41:54.387625+00:00
- Fine-tuned checkpoint: `18B_marshmello_instruct/checkpoints/latest.pt`
- Baseline: not evaluated
- Decoding: greedy

## Fine-tuned metrics

| Metric | Value |
|---|---:|
| Concept accuracy | 27.0% |
| Routing accuracy | 15.0% |
| Concept confusion rate | 85.0% |
| Exact answer overlap | 0.0% |
| Keyword overlap | 9.8% |
| Reference token overlap | 20.8% |
| Hallucination rate | 78.0% |

## Decision

Model capacity is likely the bottleneck and scaling to 300M–500M is justified.

The configured success rule requires routing accuracy above 80.0% and an absolute confusion-rate drop of at least 10.0%.

## Domain breakdown

| Domain | Count | Concept accuracy | Routing accuracy | Hallucination rate |
|---|---:|---:|---:|---:|
| ai_basics | 40 | 35.0% | 27.5% | 65.0% |
| databases | 30 | 33.3% | 3.3% | 83.3% |
| transformers_llms | 30 | 10.0% | 10.0% | 90.0% |

## Largest concept confusions

| Expected | Predicted | Count |
|---|---|---:|
| classification | database_index | 2 |
| database | database_index | 2 |
| database_column | database_index | 2 |
| database_table | database_index | 2 |
| foreign_key | database_index | 2 |
| loss_function | database_index | 2 |
| relational_database | database_index | 2 |
| self_attention | token | 2 |
| sql | database_index | 2 |
| tokenization | database_index | 2 |
| tokenizer | database_index | 2 |
| artificial_intelligence | transformer | 1 |
| attention_head | database_index | 1 |
| b_tree_index | database_index | 1 |
| bias_variance_tradeoff | transformer | 1 |

## Metric definitions

- Concept accuracy: the answer contains at least one expected concept signature.
- Routing accuracy: the expected concept has the highest weighted signature score.
- Exact answer overlap: normalized generated text exactly matches the reference answer.
- Keyword overlap: mean recall of expected concept keywords.
- Hallucination rate: deterministic proxy for empty/repetitive, ungrounded, or hard-negative-routed answers.
- The complete concept and domain confusion matrices are stored in the JSON results file.
