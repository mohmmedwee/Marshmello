# Marshmello General Benchmark Evaluation

- Generated: 2026-06-21T10:16:00.271747+00:00
- Checkpoint: `18B_marshmello_instruct/checkpoints/best_18j_routing.pt`
- Questions: 500
- Decoding: greedy

## Overall metrics

| Metric | Value |
|---|---:|
| Keyword overlap | 2.2% |
| Reference token F1 | 10.5% |
| Answer correctness | 1.0% |
| Response quality | 11.8% |
| Hallucination rate | 68.2% |
| Exact answer overlap | 0.0% |

## By benchmark bucket

| Bucket | Count | Keyword | Token F1 | Correctness | Quality | Hallucination |
|---|---:|---:|---:|---:|---:|---:|
| ai | 100 | 4.0% | 12.1% | 2.0% | 14.1% | 64.0% |
| databases | 100 | 5.1% | 12.8% | 2.0% | 15.1% | 62.0% |
| general_knowledge | 100 | 0.7% | 9.6% | 1.0% | 10.2% | 72.0% |
| programming | 100 | 0.4% | 8.7% | 0.0% | 9.6% | 72.0% |
| system_design | 100 | 0.7% | 9.2% | 0.0% | 10.2% | 71.0% |

## Metric definitions

- Keyword overlap: fraction of reference keywords present in the generated answer.
- Reference token F1: token overlap between generated and reference answers.
- Answer correctness: exact match or token F1 ≥ 35%.
- Response quality: weighted blend of keyword overlap, token F1, and non-hallucination.
- Hallucination rate: empty/repetitive answers or very low overlap with long outputs.

