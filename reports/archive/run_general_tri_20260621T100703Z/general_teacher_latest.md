# Marshmello General Benchmark Evaluation

- Generated: 2026-06-21T10:13:15.071586+00:00
- Checkpoint: `18E_tiny_teacher_sft/checkpoints/teacher_latest.pt`
- Questions: 500
- Decoding: greedy

## Overall metrics

| Metric | Value |
|---|---:|
| Keyword overlap | 2.2% |
| Reference token F1 | 10.8% |
| Answer correctness | 0.6% |
| Response quality | 12.8% |
| Hallucination rate | 64.2% |
| Exact answer overlap | 0.0% |

## By benchmark bucket

| Bucket | Count | Keyword | Token F1 | Correctness | Quality | Hallucination |
|---|---:|---:|---:|---:|---:|---:|
| ai | 100 | 3.5% | 11.2% | 0.0% | 13.6% | 63.0% |
| databases | 100 | 5.1% | 13.0% | 3.0% | 17.4% | 51.0% |
| general_knowledge | 100 | 0.6% | 10.3% | 0.0% | 12.0% | 64.0% |
| programming | 100 | 1.2% | 9.2% | 0.0% | 10.4% | 71.0% |
| system_design | 100 | 0.7% | 10.3% | 0.0% | 10.5% | 72.0% |

## Metric definitions

- Keyword overlap: fraction of reference keywords present in the generated answer.
- Reference token F1: token overlap between generated and reference answers.
- Answer correctness: exact match or token F1 ≥ 35%.
- Response quality: weighted blend of keyword overlap, token F1, and non-hallucination.
- Hallucination rate: empty/repetitive answers or very low overlap with long outputs.

