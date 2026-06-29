# Marshmello Model Family

Marshmello is an educational open-source language model family built from scratch in PyTorch. The family is intentionally small enough to study end to end while still covering the practical workflow of pretraining, instruction tuning, evaluation, and scaling experiments.

## Summary

| Model | Parameters | Config | Purpose | Status |
| ----- | ---------: | ------ | ------- | ------ |
| Marshmello-8M | 8M | `default` | Small baseline and fast learning checkpoint | Stable |
| Marshmello-55M | 55M | `large_50m` | Main stable baseline for SFT and evaluation | Stable baseline |
| Marshmello-300M | 268,834,816 | `large_300m` | Capacity scaling experiment after the 55M plateau | Phase 19A smoke/benchmark |

Parameter counts for the 8M and 55M classes depend on the trained BPE vocabulary size. The 55M label refers to the 8000-token vocabulary target used by the current large configuration. The exact Phase 19A 300M count is 268,834,816 parameters.

## Marshmello-8M

| Field | Detail |
| ----- | ------ |
| Parameter count | About 8M, vocabulary-dependent |
| Config key | `default` |
| Architecture | Decoder-only GPT, 4 layers, 384 hidden size, 6 attention heads |
| Context length | 256 tokens |
| Tokenizer | Project BPE tokenizer |
| Status | Stable |

Purpose:

Marshmello-8M is the fast baseline for learning and smoke testing. It is small enough to run quickly and useful for understanding the training stack before moving to larger models.

Benchmark notes:

Marshmello-8M is not the current benchmark target. It is mainly used as a small control in earlier evaluation phases.

Recommended usage:

- Learn the GPT pretraining path.
- Test tokenizer, training, checkpoint, and generation code.
- Compare scaling behavior against the 55M baseline.

## Marshmello-55M

| Field | Detail |
| ----- | ------ |
| Parameter count | About 55M at the 8000-token vocabulary target |
| Config key | `large_50m` |
| Architecture | Decoder-only GPT, 6 layers, 768 hidden size, 12 attention heads |
| Context length | 512 tokens |
| Tokenizer | Project BPE tokenizer |
| Status | Stable baseline |

Purpose:

Marshmello-55M is the main stable checkpoint family for teacher SFT, routing experiments, and internal evaluation. Older project docs may call this the 45M or 50M class because the exact count changes with vocabulary size.

Benchmark notes:

| Benchmark | Result |
| --------- | -----: |
| 18J Core Routing | 18% |
| 18K General Domain Score | 22.5% |
| 18K Hallucination | 64.2% |

These are educational internal benchmarks for comparing Marshmello checkpoints, not commercial LLM benchmarks.

Recommended usage:

- Use as the main educational baseline.
- Compare SFT recipes with 18J and 18K.
- Keep benchmark regressions visible before replacing a checkpoint.
- Do not treat it as a production assistant.

## Marshmello-300M

| Field | Detail |
| ----- | ------ |
| Parameter count | 268,834,816 |
| Config key | `large_300m` |
| Architecture | Decoder-only GPT, 20 layers, 1024 hidden size, 16 attention heads |
| Context length | 512 tokens |
| Tokenizer | BPE v2 / 8000 vocabulary target |
| Status | Phase 19A smoke/benchmark |

Purpose:

Marshmello-300M exists to test whether model capacity is the bottleneck after the 55M line plateaued on internal benchmarks. Phase 19A is a scale-up validation phase, not a completed trained model release.

Benchmark notes:

The current Phase 19A result is a smoke/throughput benchmark:

| Metric | Result |
| ------ | -----: |
| Training tokens/sec | 3,552 |
| Forward tokens/sec | 14,813 |
| Generation tokens/sec | 11 |
| Peak memory | 5.5GB |
| Checkpoint size | 3.0GB |
| Model weights | 1.0GB |

Recommended usage:

- Validate that the 300M configuration trains end to end.
- Use for controlled smoke testing before long training.
- Evaluate with 18J and 18K only after a real training run.
- Do not publish as a stable quality checkpoint until it beats the 55M baseline honestly.
