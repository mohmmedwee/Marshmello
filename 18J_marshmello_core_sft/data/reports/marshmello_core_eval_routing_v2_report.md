# Marshmello Core Routing Evaluation

- Generated: 2026-06-20T09:52:33.690669+00:00
- Fine-tuned checkpoint: `18I_routing_teacher_fix/checkpoints/routing_latest.pt`
- Baseline: not evaluated
- Decoding: greedy

## Fine-tuned metrics

| Metric | Value |
|---|---:|
| Concept accuracy | 3.0% |
| Routing accuracy | 2.0% |
| Concept confusion rate | 98.0% |
| Exact answer overlap | 0.0% |
| Keyword overlap | 1.2% |
| Reference token overlap | 13.2% |
| Hallucination rate | 99.0% |

## Decision

Model capacity is likely the bottleneck and scaling to 300M–500M is justified.

The configured success rule requires routing accuracy above 80.0% and an absolute confusion-rate drop of at least 10.0%.

## Domain breakdown

| Domain | Count | Concept accuracy | Routing accuracy | Hallucination rate |
|---|---:|---:|---:|---:|
| ai_basics | 40 | 5.0% | 5.0% | 97.5% |
| databases | 30 | 3.3% | 0.0% | 100.0% |
| transformers_llms | 30 | 0.0% | 0.0% | 100.0% |

## Largest concept confusions

| Expected | Predicted | Count |
|---|---|---:|
| database_column | __unknown__ | 2 |
| database_row | __unknown__ | 2 |
| tokenizer | database_index | 2 |
| acid | transformer | 1 |
| attention_head | __unknown__ | 1 |
| b_tree_index | database_index | 1 |
| bias_variance_tradeoff | __unknown__ | 1 |
| causal_mask | database_index | 1 |
| classification | database | 1 |
| classification | machine_learning | 1 |
| context_window | database_index | 1 |
| database | __unknown__ | 1 |
| database | machine_learning | 1 |
| database_constraint | supervised_learning | 1 |
| database_index | training_data | 1 |

## Metric definitions

- Concept accuracy: the answer contains at least one expected concept signature.
- Routing accuracy: the expected concept has the highest weighted signature score.
- Exact answer overlap: normalized generated text exactly matches the reference answer.
- Keyword overlap: mean recall of expected concept keywords.
- Hallucination rate: deterministic proxy for empty/repetitive, ungrounded, or hard-negative-routed answers.
- The complete concept and domain confusion matrices are stored in the JSON results file.
