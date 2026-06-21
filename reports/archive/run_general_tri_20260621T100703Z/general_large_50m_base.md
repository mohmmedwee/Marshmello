# Marshmello General Benchmark Evaluation

- Generated: 2026-06-21T10:11:30.824071+00:00
- Checkpoint: `13_gpt_pretraining/checkpoints/large_50m/latest.pt`
- Questions: 500
- Decoding: greedy

## Overall metrics

| Metric | Value |
|---|---:|
| Keyword overlap | 3.0% |
| Reference token F1 | 9.9% |
| Answer correctness | 0.6% |
| Response quality | 8.7% |
| Hallucination rate | 84.2% |
| Exact answer overlap | 0.0% |

## By benchmark bucket

| Bucket | Count | Keyword | Token F1 | Correctness | Quality | Hallucination |
|---|---:|---:|---:|---:|---:|---:|
| ai | 100 | 3.7% | 10.4% | 0.0% | 10.2% | 79.0% |
| databases | 100 | 4.9% | 10.9% | 3.0% | 10.8% | 79.0% |
| general_knowledge | 100 | 2.0% | 9.6% | 0.0% | 6.6% | 92.0% |
| programming | 100 | 2.0% | 9.8% | 0.0% | 8.5% | 83.0% |
| system_design | 100 | 2.5% | 9.0% | 0.0% | 7.3% | 88.0% |

## Metric definitions

- Keyword overlap: fraction of reference keywords present in the generated answer.
- Reference token F1: token overlap between generated and reference answers.
- Answer correctness: exact match or token F1 ≥ 35%.
- Response quality: weighted blend of keyword overlap, token F1, and non-hallucination.
- Hallucination rate: empty/repetitive answers or very low overlap with long outputs.

