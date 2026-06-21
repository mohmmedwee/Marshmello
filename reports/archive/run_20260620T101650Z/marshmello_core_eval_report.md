# Marshmello Core Routing Evaluation

- Generated: 2026-06-20T09:48:14.205318+00:00
- Fine-tuned checkpoint: `18I_routing_teacher_fix/checkpoints/routing_latest.pt`
- Baseline: not evaluated
- Decoding: greedy

## Fine-tuned metrics

| Metric | Value |
|---|---:|
| Concept accuracy | 6.0% |
| Routing accuracy | 3.0% |
| Concept confusion rate | 97.0% |
| Exact answer overlap | 0.0% |
| Keyword overlap | 1.5% |
| Reference token overlap | 12.5% |
| Hallucination rate | 94.0% |

## Decision

Model capacity is likely the bottleneck and scaling to 300M–500M is justified.

The configured success rule requires routing accuracy above 80.0% and an absolute confusion-rate drop of at least 10.0%.

## Domain breakdown

| Domain | Count | Concept accuracy | Routing accuracy | Hallucination rate |
|---|---:|---:|---:|---:|
| ai_basics | 40 | 5.0% | 5.0% | 95.0% |
| databases | 30 | 6.7% | 0.0% | 93.3% |
| transformers_llms | 30 | 6.7% | 3.3% | 93.3% |

## Largest concept confusions

| Expected | Predicted | Count |
|---|---|---:|
| machine_learning | __unknown__ | 2 |
| positional_encoding | __unknown__ | 2 |
| primary_key | supervised_learning | 2 |
| self_attention | transformer | 2 |
| tokenizer | database_index | 2 |
| transformer | __unknown__ | 2 |
| underfitting | __unknown__ | 2 |
| acid | machine_learning | 1 |
| attention_head | database_index | 1 |
| b_tree_index | next_token_prediction | 1 |
| bias_variance_tradeoff | __unknown__ | 1 |
| causal_mask | token | 1 |
| classification | __unknown__ | 1 |
| classification | database | 1 |
| context_window | machine_learning | 1 |

## Metric definitions

- Concept accuracy: the answer contains at least one expected concept signature.
- Routing accuracy: the expected concept has the highest weighted signature score.
- Exact answer overlap: normalized generated text exactly matches the reference answer.
- Keyword overlap: mean recall of expected concept keywords.
- Hallucination rate: deterministic proxy for empty/repetitive, ungrounded, or hard-negative-routed answers.
- The complete concept and domain confusion matrices are stored in the JSON results file.
