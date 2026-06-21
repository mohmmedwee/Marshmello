# Marshmello Core Routing Evaluation

- Generated: 2026-06-20T08:18:42.779290+00:00
- Fine-tuned checkpoint: `18B_marshmello_instruct/checkpoints/latest.pt`
- Baseline: not evaluated
- Decoding: greedy

## Fine-tuned metrics

| Metric | Value |
|---|---:|
| Concept accuracy | 18.0% |
| Routing accuracy | 2.0% |
| Concept confusion rate | 98.0% |
| Exact answer overlap | 0.0% |
| Keyword overlap | 6.5% |
| Reference token overlap | 18.8% |
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
| ai_basics | 40 | 10.0% | 2.5% | 95.0% |
| databases | 30 | 36.7% | 3.3% | 100.0% |
| transformers_llms | 30 | 10.0% | 0.0% | 96.7% |

## Largest concept confusions

| Expected | Predicted | Count |
|---|---|---:|
| database_column | database_index | 2 |
| database_table | database_index | 2 |
| gradient_descent | database_index | 2 |
| model_evaluation | database | 2 |
| neural_network | database | 2 |
| primary_key | database | 2 |
| regression | database | 2 |
| reinforcement_learning | database | 2 |
| training_data | database | 2 |
| underfitting | database | 2 |
| unsupervised_learning | database | 2 |
| acid | database_index | 1 |
| artificial_intelligence | neural_network | 1 |
| attention_head | database_index | 1 |
| b_tree_index | database_index | 1 |

## Metric definitions

- Concept accuracy: the answer contains at least one expected concept signature.
- Routing accuracy: the expected concept has the highest weighted signature score.
- Exact answer overlap: normalized generated text exactly matches the reference answer.
- Keyword overlap: mean recall of expected concept keywords.
- Hallucination rate: deterministic proxy for empty/repetitive, ungrounded, or hard-negative-routed answers.
- The complete concept and domain confusion matrices are stored in the JSON results file.

---

## All checkpoints compared

See **[marshmello_core_eval_comparison.md](marshmello_core_eval_comparison.md)**.

| Checkpoint | Routing acc | Concept acc |
|---|---:|---:|
| Base (pretrain) | 3.0% | 4.0% |
| Full SFT (9.8k mix) | 3.0% | 17.0% |
| Core SFT (18J, 800 steps) | 2.0% | 18.0% |
| Routing (18I) | 2.0% | 6.0% |
| Teacher (18E) | 7.0% | 9.0% |
