# Marshmello Core Routing Evaluation

- Generated: 2026-06-20T09:50:27.607594+00:00
- Fine-tuned checkpoint: `18E_tiny_teacher_sft/checkpoints/teacher_best_score.pt`
- Baseline: not evaluated
- Decoding: greedy

## Fine-tuned metrics

| Metric | Value |
|---|---:|
| Concept accuracy | 13.0% |
| Routing accuracy | 10.0% |
| Concept confusion rate | 90.0% |
| Exact answer overlap | 0.0% |
| Keyword overlap | 4.2% |
| Reference token overlap | 16.0% |
| Hallucination rate | 89.0% |

## Decision

Model capacity is likely the bottleneck and scaling to 300M–500M is justified.

The configured success rule requires routing accuracy above 80.0% and an absolute confusion-rate drop of at least 10.0%.

## Domain breakdown

| Domain | Count | Concept accuracy | Routing accuracy | Hallucination rate |
|---|---:|---:|---:|---:|
| ai_basics | 40 | 20.0% | 17.5% | 85.0% |
| databases | 30 | 13.3% | 10.0% | 86.7% |
| transformers_llms | 30 | 3.3% | 0.0% | 96.7% |

## Largest concept confusions

| Expected | Predicted | Count |
|---|---|---:|
| database_table | training_data | 2 |
| loss_function | __unknown__ | 2 |
| model_evaluation | machine_learning | 2 |
| neural_network | machine_learning | 2 |
| positional_encoding | __unknown__ | 2 |
| tokenizer | __unknown__ | 2 |
| acid | __unknown__ | 1 |
| attention_head | __unknown__ | 1 |
| b_tree_index | __unknown__ | 1 |
| bias_variance_tradeoff | training_data | 1 |
| causal_mask | machine_learning | 1 |
| classification | model_evaluation | 1 |
| context_window | __unknown__ | 1 |
| database | __unknown__ | 1 |
| database_column | __unknown__ | 1 |

## Metric definitions

- Concept accuracy: the answer contains at least one expected concept signature.
- Routing accuracy: the expected concept has the highest weighted signature score.
- Exact answer overlap: normalized generated text exactly matches the reference answer.
- Keyword overlap: mean recall of expected concept keywords.
- Hallucination rate: deterministic proxy for empty/repetitive, ungrounded, or hard-negative-routed answers.
- The complete concept and domain confusion matrices are stored in the JSON results file.
