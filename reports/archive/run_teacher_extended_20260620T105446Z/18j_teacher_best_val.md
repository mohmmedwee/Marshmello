# Marshmello Core Routing Evaluation

- Generated: 2026-06-20T10:57:56.069163+00:00
- Fine-tuned checkpoint: `18E_tiny_teacher_sft/checkpoints/teacher_best_val.pt`
- Baseline: not evaluated
- Decoding: greedy

## Fine-tuned metrics

| Metric | Value |
|---|---:|
| Concept accuracy | 13.0% |
| Routing accuracy | 11.0% |
| Concept confusion rate | 89.0% |
| Exact answer overlap | 0.0% |
| Keyword overlap | 5.2% |
| Reference token overlap | 17.3% |
| Hallucination rate | 91.0% |

## Decision

Model capacity is likely the bottleneck and scaling to 300M–500M is justified.

The configured success rule requires routing accuracy above 80.0% and an absolute confusion-rate drop of at least 10.0%.

## Domain breakdown

| Domain | Count | Concept accuracy | Routing accuracy | Hallucination rate |
|---|---:|---:|---:|---:|
| ai_basics | 40 | 25.0% | 22.5% | 82.5% |
| databases | 30 | 6.7% | 6.7% | 93.3% |
| transformers_llms | 30 | 3.3% | 0.0% | 100.0% |

## Largest concept confusions

| Expected | Predicted | Count |
|---|---|---:|
| database_table | training_data | 2 |
| foreign_key | __unknown__ | 2 |
| generalization | __unknown__ | 2 |
| gradient_descent | __unknown__ | 2 |
| inference | machine_learning | 2 |
| loss_function | __unknown__ | 2 |
| model_evaluation | machine_learning | 2 |
| positional_encoding | __unknown__ | 2 |
| acid | __unknown__ | 1 |
| attention_head | __unknown__ | 1 |
| b_tree_index | next_token_prediction | 1 |
| bias_variance_tradeoff | __unknown__ | 1 |
| causal_mask | token | 1 |
| classification | __unknown__ | 1 |
| context_window | machine_learning | 1 |

## Metric definitions

- Concept accuracy: the answer contains at least one expected concept signature.
- Routing accuracy: the expected concept has the highest weighted signature score.
- Exact answer overlap: normalized generated text exactly matches the reference answer.
- Keyword overlap: mean recall of expected concept keywords.
- Hallucination rate: deterministic proxy for empty/repetitive, ungrounded, or hard-negative-routed answers.
- The complete concept and domain confusion matrices are stored in the JSON results file.
